"""Synthetic frontal-plane poses, joint-angle measurement and skeleton rasterisation.

Keypoints follow the MediaPipe Pose naming so that the browser demo can feed real
landmarks through exactly the same normalisation + rasteriser used in training.
"x" grows to the image right, "y" grows downwards. The subject faces the camera,
so the subject's LEFT side appears on the image RIGHT (+x).
"""
import math
import numpy as np

IMG = 96  # model input resolution

# order of the 13 keypoints we use (MediaPipe indices in comments)
KP = ["nose",            # 0
      "l_shoulder",      # 11
      "r_shoulder",      # 12
      "l_elbow",         # 13
      "r_elbow",         # 14
      "l_wrist",         # 15
      "r_wrist",         # 16
      "l_hip",           # 23
      "r_hip",           # 24
      "l_knee",          # 25
      "r_knee",          # 26
      "l_ankle",         # 27
      "r_ankle"]         # 28
MP_INDEX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
I = {n: i for i, n in enumerate(KP)}

# (a, b, channel)  channel 0 = subject's right side, 1 = left side, 2 = midline
SEGMENTS = [
    ("r_shoulder", "r_elbow", 0), ("r_elbow", "r_wrist", 0),
    ("r_hip", "r_knee", 0), ("r_knee", "r_ankle", 0),
    ("l_shoulder", "l_elbow", 1), ("l_elbow", "l_wrist", 1),
    ("l_hip", "l_knee", 1), ("l_knee", "l_ankle", 1),
    ("l_shoulder", "r_shoulder", 2), ("l_hip", "r_hip", 2),
    ("neck", "midhip", 2), ("neck", "nose", 2),
]

ANGLE_KEYS = ["r_shoulder_abd", "l_shoulder_abd", "r_elbow_flex", "l_elbow_flex",
              "r_hip_abd", "l_hip_abd", "trunk_lean"]


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _profile(rng):
    """Sample target joint angles from a mixture of clinically-motivated profiles."""
    kind = rng.choice(["normal", "hemi_r", "hemi_l", "frozen_r", "frozen_l", "random", "rest"],
                      p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.2, 0.1])
    sh = rng.uniform(0, 180, 2)          # right, left
    el = rng.uniform(0, 140, 2) * (rng.random(2) < 0.6)
    hp = rng.uniform(0, 45, 2) * (rng.random(2) < 0.7)
    lean = rng.normal(0, 6) if rng.random() < 0.7 else rng.uniform(-25, 25)
    if kind == "normal":
        sh = rng.uniform(150, 180, 2); el = rng.uniform(0, 15, 2)
    elif kind in ("hemi_r", "hemi_l"):
        k = 0 if kind == "hemi_r" else 1
        sh[k] = rng.uniform(0, 100); sh[1 - k] = rng.uniform(140, 180)
        el[k] = rng.uniform(30, 120); el[1 - k] = rng.uniform(0, 20)
        lean = rng.uniform(-15, 15)
    elif kind in ("frozen_r", "frozen_l"):
        k = 0 if kind == "frozen_r" else 1
        sh[k] = rng.uniform(40, 110); sh[1 - k] = rng.uniform(150, 180)
    elif kind == "rest":
        sh = rng.uniform(0, 25, 2); el = rng.uniform(0, 30, 2); hp = rng.uniform(0, 10, 2)
    return dict(sh=sh, el=el, hp=hp, lean=float(np.clip(lean, -30, 30)))


def synth_keypoints(rng, profile=None):
    """Return (13, 2) keypoints of a frontal pose (arbitrary scale).

    profile: optional dict(sh=[r, l], el=[r, l], hp=[r, l], lean=deg) of target angles.
    """
    p = profile or _profile(rng)
    T = 1.0
    lean = p["lean"]
    up = _rot(np.array([0.0, -1.0]), lean)      # trunk direction (hip -> neck)
    across = _rot(np.array([1.0, 0.0]), lean)   # towards subject's left (+x)
    s = rng.uniform(0.9, 1.1, 6)                # body proportion noise
    midhip = np.zeros(2)
    neck = midhip + T * up
    k = np.zeros((13, 2))
    k[I["nose"]] = neck + 0.32 * s[0] * _rot(up, rng.normal(0, 8))
    k[I["l_shoulder"]] = neck + 0.22 * s[1] * across
    k[I["r_shoulder"]] = neck - 0.22 * s[1] * across
    k[I["l_hip"]] = midhip + 0.14 * s[2] * across
    k[I["r_hip"]] = midhip - 0.14 * s[2] * across
    down = -up
    ua, fa, th, sn = 0.36 * s[3], 0.30 * s[4], 0.46 * s[5], 0.44 * s[5]
    for side, sign in (("r", -1), ("l", 1)):
        j = 0 if side == "r" else 1
        # positive rotation in image coords is clockwise on screen; abduction of the
        # left arm rotates "down" towards +x i.e. counter-clockwise -> negative angle
        arm = _rot(down, -sign * p["sh"][j])
        elbow = k[I[f"{side}_shoulder"]] + ua * arm
        fore = _rot(arm, rng.choice([-1, 1]) * p["el"][j])
        k[I[f"{side}_elbow"]] = elbow
        k[I[f"{side}_wrist"]] = elbow + fa * fore
        leg = _rot(down, -sign * p["hp"][j])
        knee = k[I[f"{side}_hip"]] + th * leg
        shin = _rot(leg, rng.normal(0, 4))
        k[I[f"{side}_knee"]] = knee
        k[I[f"{side}_ankle"]] = knee + sn * shin
    k += rng.normal(0, 0.012, k.shape)          # landmark-detector style jitter
    return k


def _ang(a, b):
    """Unsigned angle in degrees between 2-D vectors a and b."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def measure(k):
    """Joint angles (degrees) measured from keypoints - the ground-truth labels."""
    neck = (k[I["l_shoulder"]] + k[I["r_shoulder"]]) / 2
    midhip = (k[I["l_hip"]] + k[I["r_hip"]]) / 2
    down = midhip - neck
    out = {}
    for side in ("r", "l"):
        sh, el, wr = k[I[f"{side}_shoulder"]], k[I[f"{side}_elbow"]], k[I[f"{side}_wrist"]]
        hip, kn = k[I[f"{side}_hip"]], k[I[f"{side}_knee"]]
        out[f"{side}_shoulder_abd"] = _ang(el - sh, down)
        out[f"{side}_elbow_flex"] = 180 - _ang(sh - el, wr - el)
        out[f"{side}_hip_abd"] = _ang(kn - hip, down)
    up = neck - midhip
    # signed lean: positive = trunk tilts towards the subject's left (+x)
    out["trunk_lean"] = math.degrees(math.atan2(up[0], -up[1]))
    return out


def normalise(k, size=IMG, margin=0.1):
    """Fit keypoints into a size x size canvas (aspect preserving, centred)."""
    neck = (k[I["l_shoulder"]] + k[I["r_shoulder"]]) / 2
    midhip = (k[I["l_hip"]] + k[I["r_hip"]]) / 2
    pts = np.vstack([k, neck, midhip])
    lo, hi = pts.min(0), pts.max(0)
    scale = (1 - 2 * margin) * size / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    centre = (lo + hi) / 2
    return (pts - centre) * scale + size / 2   # (15, 2): 13 kps + neck + midhip


def rasterise(pts, width=2.2, size=IMG):
    """Draw anti-aliased skeleton. pts: (15, 2) from normalise(). Returns float32 (3, H, W).

    Mirrored 1:1 in web/app.js - keep both in sync.
    """
    names = KP + ["neck", "midhip"]
    idx = {n: i for i, n in enumerate(names)}
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    img = np.zeros((3, size, size), np.float32)
    for a, b, ch in SEGMENTS:
        p, q = pts[idx[a]], pts[idx[b]]
        d = q - p
        L2 = float(d @ d) + 1e-9
        t = np.clip(((xs - p[0]) * d[0] + (ys - p[1]) * d[1]) / L2, 0, 1)
        dist = np.hypot(xs - (p[0] + t * d[0]), ys - (p[1] + t * d[1]))
        img[ch] = np.maximum(img[ch], np.clip(width / 2 + 0.5 - dist, 0, 1))
    # head: ring around the nose
    n = pts[idx["nose"]]
    r = 0.06 * size
    dist = np.abs(np.hypot(xs - n[0], ys - n[1]) - r)
    img[2] = np.maximum(img[2], np.clip(width / 2 + 0.5 - dist, 0, 1))
    return img


def sample(rng, width=None):
    k = synth_keypoints(rng)
    w = width if width is not None else rng.uniform(1.6, 3.0)
    return rasterise(normalise(k), width=w), measure(k)
