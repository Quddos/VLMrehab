"""Export the trained model to two ONNX graphs + vocab for the browser demo."""
import argparse
import json
import numpy as np
import onnxruntime as ort
import torch

from evaluate import load
from rehabvlm.pose import KP, MP_INDEX, sample, synth_keypoints

# example patients for the demo page (target angles: [right, left])
DEMOS = {
    "Healthy arm raise": dict(sh=[170, 172], el=[5, 5], hp=[5, 5], lean=0),
    "Right hemiparesis": dict(sh=[60, 165], el=[80, 5], hp=[3, 8], lean=-8),
    "Left frozen shoulder": dict(sh=[168, 85], el=[5, 10], hp=[5, 5], lean=4),
    "Resting stance": dict(sh=[12, 15], el=[15, 10], hp=[4, 6], lean=0),
}


def demo_poses():
    rng = np.random.default_rng(7)
    out = {}
    for name, prof in DEMOS.items():
        k = synth_keypoints(rng, {k: np.array(v, float) if isinstance(v, list) else v
                                  for k, v in prof.items()})
        k = (k - k.min(0)) / (k.max(0) - k.min(0)).max() * 0.8 + 0.1   # like MediaPipe [0,1]
        out[name] = np.round(k, 4).tolist()
    return out


class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        return self.m.encode_image(image)


class Dec(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img_tokens, ids):
        return self.m.decode(img_tokens, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--out", default="web/model")
    args = ap.parse_args()
    model, tok = load(args.ckpt)

    img = torch.zeros(1, 3, 96, 96)
    mem = model.encode_image(img)
    ids = torch.tensor([[1, 5, 6, 7]])
    torch.onnx.export(Enc(model), (img,), f"{args.out}/encoder.onnx", input_names=["image"],
                      output_names=["img_tokens"], opset_version=17, dynamo=False)
    torch.onnx.export(Dec(model), (mem, ids), f"{args.out}/decoder.onnx",
                      input_names=["img_tokens", "ids"], output_names=["logits"],
                      dynamic_axes={"ids": {1: "T"}, "logits": {1: "T"}},
                      opset_version=17, dynamo=False)
    with open(f"{args.out}/vocab.json", "w") as f:
        json.dump({"vocab": tok.itos, "keypoints": KP, "mediapipe_index": MP_INDEX}, f)
    with open(f"{args.out}/demo_poses.json", "w") as f:
        json.dump(demo_poses(), f)

    # parity check against PyTorch on a random sample
    x = torch.from_numpy(sample(np.random.default_rng(0))[0])[None]
    e = ort.InferenceSession(f"{args.out}/encoder.onnx")
    d = ort.InferenceSession(f"{args.out}/decoder.onnx")
    m_onnx = e.run(None, {"image": x.numpy()})[0]
    ids = torch.tensor([[1] + tok.encode("describe the posture .") + [2]])
    l_onnx = d.run(None, {"img_tokens": m_onnx, "ids": ids.numpy()})[0]
    with torch.no_grad():
        l_pt = model(x, ids).numpy()
    print("max |logit diff|:", float(np.abs(l_onnx - l_pt).max()))


if __name__ == "__main__":
    main()
