"""Evaluate RehabVLM on fresh synthetic poses: angle error of generated text + exact-match.

    python evaluate.py --n 300
"""
import argparse
import re
import numpy as np
import torch

from rehabvlm.pose import sample
from rehabvlm.text import QUESTIONS, Tokenizer, answer, BOS, SEP, EOS
from rehabvlm.model import TinyVLM

FIELDS = [("right shoulder abduction", "r_shoulder_abd"), ("left shoulder abduction", "l_shoulder_abd"),
          ("right elbow flexion", "r_elbow_flex"), ("left elbow flexion", "l_elbow_flex"),
          ("right hip abduction", "r_hip_abd"), ("left hip abduction", "l_hip_abd")]


def load(path):
    ck = torch.load(path, map_location="cpu")
    tok = Tokenizer(ck["vocab"])
    model = TinyVLM(len(tok.itos))
    model.load_state_dict(ck["model"])
    return model.eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    model, tok = load(args.ckpt)
    rng = np.random.default_rng(args.seed)

    errs = {k: [] for _, k in FIELDS}
    exact = {t: [] for t in QUESTIONS}
    for i in range(args.n):
        img, m = sample(rng, width=2.2)
        x = torch.from_numpy(img)[None]
        for task in QUESTIONS:
            prompt = torch.tensor([[BOS] + tok.encode(QUESTIONS[task]) + [SEP]])
            out = tok.decode(model.generate(x, prompt, EOS)[0].tolist())
            ref = tok.decode(tok.encode(answer(task, m)))
            exact[task].append(out == ref)
            if task == "describe":
                for phrase, key in FIELDS:
                    hit = re.search(phrase + r" (\d+) degrees", out)
                    errs[key].append(abs(int(hit.group(1)) - m[key]) if hit else 180)
            if i == 0:
                print(f"[{task}] {out}")
    print("\nmean absolute error vs. true angle (degrees, 'describe' task):")
    for _, k in FIELDS:
        print(f"  {k:16s} {np.mean(errs[k]):5.1f}")
    print(f"  {'overall':16s} {np.mean([e for v in errs.values() for e in v]):5.1f}")
    print("\nexact-match with reference answer (5-degree bins):")
    for t, v in exact.items():
        print(f"  {t:10s} {100 * np.mean(v):5.1f}%")


if __name__ == "__main__":
    main()
