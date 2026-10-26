"""Train RehabVLM on procedurally generated skeleton images (CPU friendly).

    python train.py --steps 4000
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from rehabvlm.domains import DOMAINS
from rehabvlm.text import Tokenizer, PAD, BOS, SEP, EOS
from rehabvlm.model import TinyVLM


def make_batch(rng, tok, n, dom, max_len=64):
    return collate(rng, tok, dom, [dom.sample(rng) for _ in range(n)], max_len)


def collate(rng, tok, dom, items, max_len=64):
    """items: [(image, measurements)] -> tensors; each item gets a random question."""
    imgs, ids, tgt, ang = [], [], [], []
    for img, m in items:
        tasks = list(dom.questions)
        task = tasks[rng.integers(len(tasks))]
        q = tok.encode(dom.questions[task])
        a = tok.encode(dom.answer(task, m)) + [EOS]
        seq = [BOS] + q + [SEP] + a
        lab = [-100] * (len(q) + 2) + a       # loss only on the answer
        seq, lab = seq[:-1], lab[1:]
        pad = max_len - len(seq)
        imgs.append(img)
        ang.append(dom.targets(m))
        ids.append(seq + [PAD] * pad)
        tgt.append(lab[:len(seq)] + [-100] * pad)
    return (torch.from_numpy(np.stack(imgs)), torch.tensor(ids), torch.tensor(tgt),
            torch.tensor(ang, dtype=torch.float32))


class Stream(torch.utils.data.IterableDataset):
    """Endless stream of fresh synthetic batches, generated in a worker process."""

    def __init__(self, seed, batch, domain):
        self.seed, self.batch, self.domain = seed, batch, domain

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973)
        dom = DOMAINS[self.domain]
        tok = Tokenizer(dom.build_vocab())
        while True:
            yield make_batch(rng, tok, self.batch, dom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", choices=list(DOMAINS), default="body")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dom = DOMAINS[args.domain]
    tok = Tokenizer(dom.build_vocab())
    model = TinyVLM(len(tok.itos), n_targets=dom.n_targets)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(Stream(args.seed, args.batch, args.domain), batch_size=None,
                                              num_workers=args.workers, prefetch_factor=4))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        img, ids, tgt, ang = next(loader)
        logits, ang_pred = model(img, ids)
        text_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        aux_loss = F.mse_loss(ang_pred, ang)   # -1 targets (not measurable) are learnt as-is
        loss = text_loss + aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step:5d}  text {text_loss.item():.4f}  angles {aux_loss.item():.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == args.steps:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": model.state_dict(), "vocab": tok.itos, "domain": args.domain,
                        "n_targets": dom.n_targets}, args.out)


if __name__ == "__main__":
    main()
