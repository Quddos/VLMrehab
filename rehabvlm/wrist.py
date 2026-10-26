"""Synthetic wrists: hand landmarks + elbow -> wrist flexion/extension and radial/ulnar deviation.

Input is the 21 MediaPipe-order hand landmarks plus one elbow point (from body tracking:
MediaPipe Pose, Meta Movement SDK, a Humanoid rig, Kinect...). Only the forearm *direction*
(elbow -> wrist) is used, so the elbow may come from a different tracker or frame origin as
long as the axes agree.

Clinical convention: the hand axis is the 3rd metacarpal (wrist -> middle MCP).
  flexion  > 0: hand bends towards the palm      extension < 0
  radial   > 0: hand bends towards the thumb     ulnar     < 0
The palm side is only known from handedness, so it is a required input.

Rendering uses a forearm frame (origin wrist, +y along the forearm, +z palm side, +x radial)
with the forearm drawn as a fixed-length stub, then the same three orthographic views as
hands: channel 0 front (x, y), 1 side (z, y), 2 top (x, z). Mirrored in web/core.mjs.
"""
import math
import numpy as np

from rehabvlm import hand as H
from rehabvlm.hand import IMG, SEGMENTS as HAND_SEGMENTS, _ink, _unit, random_rotation

ANGLE_KEYS = ["wrist_flex", "wrist_dev"]
STUB = 1.2                                  # forearm stub length in palm lengths
VIEWS = [(0, 1, 0.0, 0.3), (2, 1, 0.0, 0.3), (0, 2, 0.0, 0.0)]
PX = 28.0


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])      # y -> z (flexion towards palm)


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])      # y -> x (radial deviation)


def _profile(rng):
    kind = rng.choice(["neutral", "flexion", "extension", "radial", "ulnar", "combined",
                       "spastic", "stiff", "random"],
                      p=[0.1, 0.15, 0.15, 0.1, 0.1, 0.12, 0.08, 0.08, 0.12])
    flex, dev = rng.uniform(-80, 90), rng.uniform(-35, 25)
    if kind == "neutral":
        flex, dev = rng.normal(0, 6), rng.normal(0, 4)
    elif kind == "flexion":
        flex, dev = rng.uniform(5, 90), rng.normal(0, 6)
    elif kind == "extension":
        flex, dev = -rng.uniform(5, 80), rng.normal(0, 6)
    elif kind == "radial":
        flex, dev = rng.normal(0, 8), rng.uniform(3, 25)
    elif kind == "ulnar":
        flex, dev = rng.normal(0, 8), -rng.uniform(3, 35)
    elif kind == "spastic":                   # stroke: flexed, ulnar-deviated wrist
        flex, dev = rng.uniform(40, 85), -rng.uniform(5, 25)
    elif kind == "stiff":                     # post-fracture: little movement either way
        flex, dev = rng.uniform(-20, 25), rng.uniform(-8, 8)
    return float(np.clip(flex, -85, 90)), float(np.clip(dev, -40, 30))


def synth_canonical(rng, wrist=None, hand=None):
    """Right hand + elbow in the forearm frame. Returns (21, 3) hand, (3,) elbow."""
    flex, dev = wrist or _profile(rng)
    k = synth_canonical_hand(rng, hand)
    k = k @ (_rx(flex) @ _rz(dev)).T          # deviation in the hand plane, then flexion
    elbow = np.array([0.0, -rng.uniform(2.3, 2.8), 0.0])
    return k, elbow


def synth_canonical_hand(rng, hand=None):
    """Canonical hand rotated so the 3rd metacarpal lies exactly on +y (neutral wrist)."""
    k = H.synth_canonical(rng, hand)
    a = math.degrees(math.atan2(k[9, 0], k[9, 1]))
    return k @ _rz(-a).T


def synth_raw(rng, wrist=None, hand=None):
    """Forearm-frame sample -> random mirroring, orientation, scale and tracker noise.

    Returns (21, 3) hand, (3,) elbow, handedness ("right" | "left")."""
    k, elbow = synth_canonical(rng, wrist, hand)
    pts = np.vstack([k, elbow])
    handed = "right"
    if rng.random() < 0.5:
        pts[:, 0] *= -1
        handed = "left"
    pts = pts @ random_rotation(rng).T * rng.uniform(0.05, 0.3) + rng.normal(0, 1, 3)
    scale = np.linalg.norm(pts[9] - pts[0])
    pts[:21] += rng.normal(0, 0.008, (21, 3)) * scale
    pts[:21, 2] += rng.normal(0, 0.015, 21) * scale
    pts[21] += rng.normal(0, 0.06, 3) * scale       # elbow from body tracking is coarser
    return pts[:21], pts[21], handed


def _frame(k, elbow, handedness):
    """Forearm direction f, palm-side normal n and radial axis r (orthonormal)."""
    k = np.asarray(k, float)
    f = _unit(k[0] - np.asarray(elbow, float))
    s = -1.0 if str(handedness).lower() == "left" else 1.0
    n = s * np.cross(k[5] - k[0], k[17] - k[0])
    n = _unit(n - f * np.dot(n, f))
    r = s * np.cross(f, n)                 # towards the thumb side for either hand
    return f, n, r


def measure(k, elbow, handedness="right"):
    k = np.asarray(k, float)
    f, n, r = _frame(k, elbow, handedness)
    h = _unit(k[9] - k[0])
    return {"wrist_flex": math.degrees(math.atan2(np.dot(h, n), np.dot(h, f))),
            "wrist_dev": math.degrees(math.asin(max(-1.0, min(1.0, np.dot(h, r)))))}


def canonicalise(k, elbow, handedness="right"):
    """(22, 3): hand landmarks + forearm stub end, in the forearm frame (palm length = 1)."""
    k = np.asarray(k, float)
    f, n, r = _frame(k, elbow, handedness)
    rel = k - k[0]
    c = np.stack([rel @ r, rel @ f, rel @ n], 1) / (np.linalg.norm(rel[9]) + 1e-9)
    return np.vstack([c, [0.0, -STUB, 0.0]])


def rasterise(c, width=2.0, size=IMG):
    img = np.zeros((3, size, size), np.float32)
    for ch, (u, v, cu, cv) in enumerate(VIEWS):
        pts = np.stack([(c[:, u] - cu) * PX + size / 2, size / 2 - (c[:, v] - cv) * PX], 1)
        for a, b in HAND_SEGMENTS:
            _ink(img[ch], pts[a], pts[b], width, size)
        _ink(img[ch], pts[0], pts[21], 2 * width, size)          # forearm, drawn thicker
        for tip in (4, 8, 12, 16, 20):
            _ink(img[ch], pts[tip], pts[tip], 2 * width, size)
    return img


def sample(rng, width=None):
    k, elbow, handed = synth_raw(rng)
    w = width if width is not None else rng.uniform(1.6, 2.6)
    return rasterise(canonicalise(k, elbow, handed), width=w), measure(k, elbow, handed)
