"""Export the trained model to two ONNX graphs + vocab for the browser demo."""
import argparse
import json
import os
import numpy as np
import onnxruntime as ort
import torch

from evaluate import load
from rehabvlm.domains import DOMAINS
from rehabvlm.pose import KP, MP_INDEX, synth_keypoints

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


HAND_DEMOS = {   # (MCP, PIP, DIP) per finger, thumb, spread
    "Open hand": ({f: [5, 5, 3] for f in ("index", "middle", "ring", "pinky")},
                  dict(opp=0.1, mcp=5, ip=5), 30.0),
    "Full fist": ({f: [85, 100, 70] for f in ("index", "middle", "ring", "pinky")},
                  dict(opp=0.8, mcp=45, ip=50), 0.0),
    "Stiff ring finger": ({**{f: [80, 100, 70] for f in ("index", "middle", "pinky")}, "ring": [15, 20, 10]},
                          dict(opp=0.8, mcp=40, ip=45), 0.0),
    "Weak grip (stroke)": ({f: [35, 45, 25] for f in ("index", "middle", "ring", "pinky")},
                           dict(opp=0.4, mcp=20, ip=15), 5.0),
    "Pinch": ({"index": [45, 70, 20], "middle": [30, 40, 20], "ring": [35, 45, 25], "pinky": [40, 50, 30]},
              dict(opp=0.7, mcp=10, ip=30), 5.0),
}


def demo_hands():
    from rehabvlm.hand import synth_canonical
    rng = np.random.default_rng(3)
    tilt = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    out = {}
    for name, (f, thumb, spread) in HAND_DEMOS.items():
        k = synth_canonical(rng, ({n: np.array(v, float) for n, v in f.items()}, thumb, spread))
        k = (k @ tilt.T) * np.array([1, -1, 1]) * 0.09     # metres, y down like MediaPipe
        out[name] = np.round(k, 5).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--out", default=None, help="default: web/model (body) or web/model/hand")
    args = ap.parse_args()
    model, tok, domain = load(args.ckpt)
    args.out = args.out or ("web/model" if domain == "body" else "web/model/hand")
    os.makedirs(args.out, exist_ok=True)
    dom = DOMAINS[domain]

    img = torch.zeros(1, 3, 96, 96)
    mem = model.encode_image(img)
    ids = torch.tensor([[1, 5, 6, 7]])
    torch.onnx.export(Enc(model).eval(), (img,), f"{args.out}/encoder.onnx", input_names=["image"],
                      output_names=["img_tokens"], opset_version=17, dynamo=False)
    torch.onnx.export(Dec(model).eval(), (mem, ids), f"{args.out}/decoder.onnx",
                      input_names=["img_tokens", "ids"], output_names=["logits"],
                      dynamic_axes={"ids": {1: "T"}, "logits": {1: "T"}},
                      opset_version=17, dynamo=False)
    with open(f"{args.out}/vocab.json", "w") as f:
        if domain == "body":
            json.dump({"vocab": tok.itos, "keypoints": KP, "mediapipe_index": MP_INDEX}, f)
        else:
            json.dump({"vocab": tok.itos, "landmarks": 21}, f)
    with open(f"{args.out}/{'demo_poses' if domain == 'body' else 'demo_hands'}.json", "w") as f:
        json.dump(demo_poses() if domain == "body" else demo_hands(), f)

    # parity check against PyTorch on a random sample
    x = torch.from_numpy(dom.sample(np.random.default_rng(0))[0])[None]
    e = ort.InferenceSession(f"{args.out}/encoder.onnx")
    d = ort.InferenceSession(f"{args.out}/decoder.onnx")
    m_onnx = e.run(None, {"image": x.numpy()})[0]
    ids = torch.tensor([[1] + tok.encode(dom.questions["describe"]) + [2]])
    l_onnx = d.run(None, {"img_tokens": m_onnx, "ids": ids.numpy()})[0]
    with torch.no_grad():
        l_pt = model(x, ids)[0].numpy()
    print("max |logit diff|:", float(np.abs(l_onnx - l_pt).max()))


if __name__ == "__main__":
    main()
