"""Real (clinical) data: record format, validation and conversion to training samples.

One JSON object per line (JSONL). A record is an API request body plus clinician labels:

  {"domain": "hand",                       # "body" | "hand" | "wrist"
   "subject": "site1-017",                 # pseudonymous id, used for subject-wise splits
   "consent": true,                        # records without documented consent are rejected
   "hand": [[x, y, z], ... 21],            # hand / wrist: MediaPipe Hands order
   "elbow": [x, y, z], "handedness": "left",   # wrist only (elbow only needs the same axes)
   "joints": {...} | "mediapipe": [[x, y], ... 33],   # body
   "space": "image" | "unity",
   "labels": {"index_flex": 182, ...},     # goniometer / mocap values, any subset of the keys
   "source": "goniometer"}

Labels override what geometry measures from the landmarks, so fine-tuning teaches the model
to correct systematic tracker error (e.g. hand trackers under-reading finger curl). Keys
without a label fall back to the landmark geometry. No images are stored, only landmarks.
"""
import json

import numpy as np

from rehabvlm import hand, pose, wrist

LABEL_KEYS = {"body": pose.ANGLE_KEYS, "hand": hand.ANGLE_KEYS, "wrist": wrist.ANGLE_KEYS}


def _body_points(rec):
    if "mediapipe" in rec:
        mp = rec["mediapipe"]
        if len(mp) < 29:
            raise ValueError("mediapipe needs the 33 pose landmarks")
        pts = [mp[i] for i in pose.MP_INDEX]
    elif "joints" in rec:
        missing = [n for n in pose.KP if n not in rec["joints"]]
        if missing:
            raise ValueError("missing joints: " + ", ".join(missing))
        pts = [rec["joints"][n] for n in pose.KP]
    else:
        raise ValueError("body record needs `joints` or `mediapipe`")
    k = np.array([p[:2] for p in pts], float)
    if rec.get("space") == "unity":
        k = -k                                      # matches keypointsFromInput in web/core.mjs
    return k


def _hand_points(rec):
    h = np.array(rec["hand"], float)
    if h.shape != (21, 3):
        raise ValueError("`hand` must be 21 landmarks [x, y, z]")
    if rec.get("space") == "unity":
        h[:, 2] *= -1
    return h


def _elbow(rec):
    e = np.array(rec["elbow"], float)
    if e.shape != (3,):
        raise ValueError("`elbow` must be [x, y, z]")
    if rec.get("space") == "unity":
        e[2] *= -1
    return e


def validate(rec):
    """Raise ValueError if a record can't be used; return the domain."""
    if rec.get("consent") is not True:
        raise ValueError("record has no documented consent")
    domain = rec.get("domain")
    if domain not in LABEL_KEYS:
        raise ValueError(f"unknown domain {domain!r}")
    if not rec.get("subject"):
        raise ValueError("record needs a pseudonymous `subject` id")
    unknown = set(rec.get("labels", {})) - set(LABEL_KEYS[domain])
    if unknown:
        raise ValueError(f"unknown label keys for {domain}: {sorted(unknown)}")
    render_and_measure(rec)
    return domain


def render_and_measure(rec, rng=None, width=None):
    """-> (image (3, 96, 96), geometry measurements from the landmarks).

    With an rng, applies light augmentation (landmark jitter, line width)."""
    domain = rec["domain"]
    jitter = (lambda a, s: a + rng.normal(0, s, a.shape)) if rng is not None else (lambda a, s: a)
    w = width if width is not None else (rng.uniform(1.6, 2.6) if rng is not None else (2.2 if domain == "body" else 2.0))
    if domain == "body":
        k = _body_points(rec)
        span = np.ptp(k, 0).max()
        k = jitter(k, 0.005 * span)
        return pose.rasterise(pose.normalise(k), width=w), pose.measure(k)
    h = _hand_points(rec)
    scale = np.linalg.norm(h[9] - h[0])
    h = jitter(h, 0.004 * scale)
    if domain == "hand":
        return hand.rasterise(hand.canonicalise(h), width=w), hand.measure(h)
    e, handed = _elbow(rec), rec.get("handedness", "right")
    return wrist.rasterise(wrist.canonicalise(h, e, handed), width=w), wrist.measure(h, e, handed)


def targets(rec, measured):
    """Measurements used to build the answer: clinician labels where given, else geometry."""
    out = dict(measured)
    out.update({k: float(v) for k, v in rec.get("labels", {}).items()})
    return out


def load(paths, domain=None):
    """Load and validate JSONL files. Returns (records, rejected[(line, reason)])."""
    records, rejected = [], []
    for path in paths if isinstance(paths, (list, tuple)) else [paths]:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    d = validate(rec)
                    if domain is None or d == domain:
                        records.append(rec)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                    rejected.append((f"{path}:{i}", str(e)))
    return records, rejected


def split_by_subject(records, val_frac=0.2, seed=0):
    """Hold out whole subjects so validation measures generalisation to new patients."""
    subjects = sorted({r["subject"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac))) if len(subjects) > 1 else 0
    val = set(subjects[:n_val])
    return [r for r in records if r["subject"] not in val], [r for r in records if r["subject"] in val]
