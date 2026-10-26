"""Fine-tune a trained RehabVLM model on real clinical recordings (see rehabvlm/realdata.py).

    python finetune.py --base checkpoints/rehabvlm-hand.pt --data clinic.jsonl \\
                       --out checkpoints/rehabvlm-hand-ft.pt

Batches mix real records (augmented) with fresh synthetic samples so the model adapts to
real trackers and clinician labels without forgetting the full range of postures. Whole
subjects are held out for validation, and the report compares three readings against the
clinician labels: plain landmark geometry, the base model and the fine-tuned model.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load
from rehabvlm import realdata
from rehabvlm.answers import parse_describe
from rehabvlm.domains import DOMAINS
from rehabvlm.text import BOS, SEP, EOS
from train import collate


@torch.no_grad()
def evaluate_real(model, tok, dom, domain, records):
    """Mean absolute error vs clinician labels: (geometry, model) per label key."""
    model.eval()
    prompt = torch.tensor([[BOS] + tok.encode(dom.questions["describe"]) + [SEP]])
    geo, vlm = {}, {}
    for rec in records:
        img, measured = realdata.render_and_measure(rec)
        said = parse_describe(domain, tok.decode(model.generate(torch.from_numpy(img)[None], prompt, EOS)[0].tolist()))
        for key, label in rec.get("labels", {}).items():
            if key in said:
                geo.setdefault(key, []).append(abs(measured[key] - label))
                vlm.setdefault(key, []).append(abs(said[key] - label))
    return {k: (float(np.mean(geo[k])), float(np.mean(vlm[k])), len(vlm[k])) for k in sorted(vlm)}


def report(title, res):
    print(f"\n{title}")
    print(f"  {'label':16s} {'geometry':>9s} {'model':>7s}   n")
    for k, (g, v, n) in res.items():
        print(f"  {k:16s} {g:8.1f}° {v:6.1f}°  {n}")
    if res:
        print(f"  {'mean':16s} {np.mean([g for g, _, _ in res.values()]):8.1f}° "
              f"{np.mean([v for _, v, _ in res.values()]):6.1f}°")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--real-frac", type=float, default=0.5, help="share of each batch from real records")
    ap.add_argument("--val-frac", type=float, default=0.2, help="share of subjects held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, tok, domain = load(args.base)
    dom = DOMAINS[domain]
    records, rejected = realdata.load(args.data, domain)
    for where, why in rejected[:20]:
        print(f"rejected {where}: {why}")
    if rejected:
        print(f"{len(rejected)} record(s) rejected")
    if not records:
        raise SystemExit(f"no usable {domain} records")
    train_recs, val_recs = realdata.split_by_subject(records, args.val_frac, args.seed)
    n_sub = len({r["subject"] for r in records})
    print(f"{domain}: {len(records)} records from {n_sub} subjects -> "
          f"train {len(train_recs)}, validation {len(val_recs)} (held-out subjects)")
    if not val_recs:
        print("warning: only one subject, so validation reuses the training records")
        val_recs = train_recs

    report("before fine-tuning (validation subjects)", evaluate_real(model, tok, dom, domain, val_recs))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_real = max(1, int(args.batch * args.real_frac))
    for step in range(1, args.steps + 1):
        model.train()
        items = []
        for i in rng.integers(len(train_recs), size=n_real):
            rec = train_recs[i]
            img, measured = realdata.render_and_measure(rec, rng)
            items.append((img, realdata.targets(rec, measured)))
        items += [dom.sample(rng) for _ in range(args.batch - n_real)]
        img, ids, tgt, ang = collate(rng, tok, dom, items)
        logits, ang_pred = model(img, ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)) + F.mse_loss(ang_pred, ang)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}", flush=True)

    report("after fine-tuning (validation subjects)", evaluate_real(model, tok, dom, domain, val_recs))
    torch.save({"model": model.state_dict(), "vocab": tok.itos, "domain": domain,
                "n_targets": dom.n_targets, "base": args.base,
                "finetuned_on": {"records": len(train_recs), "subjects": n_sub}}, args.out)
    print(f"\nsaved {args.out}  (export with: python export_onnx.py --ckpt {args.out})")


if __name__ == "__main__":
    main()
