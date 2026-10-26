"""Evaluate RehabVLM on fresh synthetic samples: angle error of generated text + exact-match.

    python evaluate.py --n 300
    python evaluate.py --ckpt checkpoints/rehabvlm-hand.pt --n 300
"""
import argparse
import numpy as np
import torch

from rehabvlm.answers import parse_describe
from rehabvlm.domains import DOMAINS
from rehabvlm.text import Tokenizer, BOS, SEP, EOS
from rehabvlm.model import TinyVLM

# measurement keys scored from the 'describe' answer, per domain
FIELDS = {
    "body": ["r_shoulder_abd", "l_shoulder_abd", "r_elbow_flex", "l_elbow_flex", "r_hip_abd", "l_hip_abd"],
    "hand": [f"{f}_flex" for f in ["thumb", "index", "middle", "ring", "pinky"]],
    "wrist": ["wrist_flex", "wrist_dev"],
}


def load(path):
    ck = torch.load(path, map_location="cpu")
    tok = Tokenizer(ck["vocab"])
    model = TinyVLM(len(tok.itos), n_targets=ck.get("n_targets", 7))
    model.load_state_dict(ck["model"])
    return model.eval(), tok, ck.get("domain", "body")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    model, tok, domain = load(args.ckpt)
    dom = DOMAINS[domain]
    rng = np.random.default_rng(args.seed)

    errs = {k: [] for k in FIELDS[domain]}
    exact = {t: [] for t in dom.questions}
    for i in range(args.n):
        img, m = dom.sample(rng, width=2.2 if domain == "body" else 2.0)
        x = torch.from_numpy(img)[None]
        for task, question in dom.questions.items():
            prompt = torch.tensor([[BOS] + tok.encode(question) + [SEP]])
            out = tok.decode(model.generate(x, prompt, EOS)[0].tolist())
            ref = tok.decode(tok.encode(dom.answer(task, m)))
            exact[task].append(out == ref)
            if task == "describe":
                said = parse_describe(domain, out)
                for key in FIELDS[domain]:
                    errs[key].append(abs(said[key] - m[key]) if key in said else 180)
            if i == 0:
                print(f"[{task}] {out}")
    print(f"\n[{domain}] mean absolute error vs. true angle (degrees, 'describe' task):")
    for k in FIELDS[domain]:
        print(f"  {k:16s} {np.mean(errs[k]):5.1f}")
    print(f"  {'overall':16s} {np.mean([e for v in errs.values() for e in v]):5.1f}")
    print("\nexact-match with reference answer (5-degree bins):")
    for t, v in exact.items():
        print(f"  {t:10s} {100 * np.mean(v):5.1f}%")


if __name__ == "__main__":
    main()
