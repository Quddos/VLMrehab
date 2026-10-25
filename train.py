"""Train RehabVLM on procedurally generated skeleton images (CPU friendly).

    python train.py --steps 4000
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from rehabvlm.pose import ANGLE_KEYS, sample
from rehabvlm.text import QUESTIONS, Tokenizer, answer, PAD, BOS, SEP, EOS
from rehabvlm.model import TinyVLM

TASKS = list(QUESTIONS)


def make_batch(rng, tok, n, max_len=64):
    imgs, ids, tgt, ang = [], [], [], []
    for _ in range(n):
        img, m = sample(rng)
        task = TASKS[rng.integers(len(TASKS))]
        q = tok.encode(QUESTIONS[task])
        a = tok.encode(answer(task, m)) + [EOS]
        seq = [BOS] + q + [SEP] + a
        lab = [-100] * (len(q) + 2) + a       # loss only on the answer
        seq, lab = seq[:-1], lab[1:]
        pad = max_len - len(seq)
        imgs.append(img)
        ang.append([m[k] / 90 for k in ANGLE_KEYS])
        ids.append(seq + [PAD] * pad)
        tgt.append(lab[:len(seq)] + [-100] * pad)
    return (torch.from_numpy(np.stack(imgs)), torch.tensor(ids), torch.tensor(tgt),
            torch.tensor(ang, dtype=torch.float32))


class Stream(torch.utils.data.IterableDataset):
    """Endless stream of fresh synthetic batches, generated in a worker process."""

    def __init__(self, seed, batch):
        self.seed, self.batch = seed, batch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973)
        tok = Tokenizer()
        while True:
            yield make_batch(rng, tok, self.batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/rehabvlm.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = Tokenizer()
    model = TinyVLM(len(tok.itos))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(Stream(args.seed, args.batch), batch_size=None,
                                              num_workers=1, prefetch_factor=4))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        img, ids, tgt, ang = next(loader)
        logits, ang_pred = model(img, ids)
        text_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        aux_loss = F.mse_loss(ang_pred, ang)
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
            torch.save({"model": model.state_dict(), "vocab": tok.itos}, args.out)


if __name__ == "__main__":
    main()
