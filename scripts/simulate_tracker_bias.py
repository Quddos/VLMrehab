"""SIMULATED clinic data, for testing the fine-tuning pipeline only (not patient data).

Mimics a known real-world error: hand trackers under-read finger curl, especially at the
PIP/DIP joints, while a clinician's goniometer reads the true angle. Each simulated subject
gets their own tracker bias. Output records follow rehabvlm/realdata.py and are marked
"source": "simulated".

    python scripts/simulate_tracker_bias.py --subjects 30 --per-subject 20 --out sim_hand.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rehabvlm import hand  # noqa: E402


def biased(profile, mcp_gain, ip_gain):
    f, thumb, spread = profile
    f = {n: np.array([a[0] * mcp_gain, a[1] * ip_gain, a[2] * ip_gain]) for n, a in f.items()}
    thumb = dict(thumb, mcp=thumb["mcp"] * ip_gain, ip=thumb["ip"] * ip_gain)
    return f, thumb, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--out", default="sim_hand.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    with open(args.out, "w") as out:
        for s in range(args.subjects):
            mcp_gain, ip_gain = rng.uniform(0.8, 0.95), rng.uniform(0.65, 0.85)
            for _ in range(args.per_subject):
                profile = hand._profile(rng)
                seed = int(rng.integers(1 << 31))
                true = hand.synth_canonical(np.random.default_rng(seed), profile)
                seen = hand.synth_canonical(np.random.default_rng(seed), biased(profile, mcp_gain, ip_gain))
                if rng.random() < 0.5:
                    seen[:, 0] *= -1
                seen = seen @ hand.random_rotation(rng).T * 0.09 + rng.normal(0, 0.3, 3)
                seen += rng.normal(0, 0.0008, seen.shape)
                labels = {k: round(v, 1) for k, v in hand.measure(true).items() if k.endswith("_flex")}
                out.write(json.dumps({"domain": "hand", "subject": f"sim-{s:02d}", "consent": True,
                                      "source": "simulated", "hand": np.round(seen, 5).tolist(),
                                      "labels": labels}) + "\n")
    print(f"wrote {args.subjects * args.per_subject} simulated records to {args.out}")


if __name__ == "__main__":
    main()
