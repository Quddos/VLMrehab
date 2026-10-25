"""Synthetic 3-D hands, finger-angle measurement and a 3-view skeleton rasteriser.

Landmarks follow the 21-point MediaPipe Hands layout (also easy to map from OpenXR /
Unity XR Hands): 0 wrist; thumb 1-4 (CMC, MCP, IP, tip); index 5-8, middle 9-12,
ring 13-16, pinky 17-20 (MCP, PIP, DIP, tip).

Pipeline: raw 3-D landmarks in any pose/orientation -> canonical hand frame
(origin wrist, +y towards the middle knuckle, +x towards the index side, z = palm
normal) -> three orthographic views, one per image channel:
  channel 0: front (x, y)   channel 1: side (z, y)   channel 2: top (x, z)
Which way z points depends on left/right hand and coordinate conventions, so training
randomly mirrors hands and the model learns to read curl in either direction.
canonicalise() and rasterise() are mirrored 1:1 in web/core.mjs.
"""
import math
import numpy as np

IMG = 96
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
SEGMENTS = [(c[i], c[i + 1]) for c in CHAINS for i in range(4)] + [(5, 9), (9, 13), (13, 17)]
ANGLE_KEYS = ["thumb_flex", "index_flex", "middle_flex", "ring_flex", "pinky_flex", "spread", "pinch_gap"]

# canonical right hand, palm length (wrist -> middle MCP) = 1, palm side = +z
MCP = {"index": (0.30, 0.95), "middle": (0.08, 1.0), "ring": (-0.13, 0.95), "pinky": (-0.32, 0.85)}
BONES = {"index": (0.46, 0.27, 0.20), "middle": (0.50, 0.31, 0.21),
         "ring": (0.47, 0.30, 0.21), "pinky": (0.37, 0.21, 0.18)}
SPREAD_DIR = {"index": 1.0, "middle": 0.2, "ring": -0.5, "pinky": -1.0}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _profile(rng):
    """Joint angles (degrees) for a clinically-motivated hand posture."""
    kind = rng.choice(["open", "fist", "partial", "pinch", "claw", "spastic", "stiff_finger", "random"],
                      p=[0.15, 0.12, 0.15, 0.15, 0.08, 0.08, 0.12, 0.15])
    f = {n: rng.uniform([0, 0, 0], [90, 110, 80]) for n in BONES}     # MCP, PIP, DIP
    thumb = dict(opp=rng.uniform(0, 1), mcp=rng.uniform(0, 60), ip=rng.uniform(0, 80))
    spread = rng.uniform(0, 25)
    if kind == "open":
        for n in f: f[n] = rng.uniform(0, [15, 15, 10])
        thumb = dict(opp=rng.uniform(0, 0.3), mcp=rng.uniform(0, 15), ip=rng.uniform(0, 15))
        spread = rng.uniform(0, 25)
    elif kind == "fist":
        for n in f: f[n] = rng.uniform([70, 90, 50], [90, 110, 80])
        thumb = dict(opp=rng.uniform(0.5, 1), mcp=rng.uniform(30, 60), ip=rng.uniform(30, 80))
        spread = rng.uniform(0, 5)
    elif kind == "partial":
        s = rng.uniform(0.3, 0.7)
        for n in f: f[n] = s * np.array([90, 110, 80]) * rng.uniform(0.8, 1.2, 3)
        spread = rng.uniform(0, 10)
    elif kind == "pinch":
        f["index"] = rng.uniform([35, 55, 5], [55, 85, 35])
        for n in ("middle", "ring", "pinky"): f[n] = rng.uniform(0, [70, 90, 60])
        thumb = dict(opp=rng.uniform(0.6, 0.85), mcp=rng.uniform(0, 25), ip=rng.uniform(15, 45))
    elif kind == "claw":
        for n in f: f[n] = np.array([rng.uniform(0, 10), rng.uniform(70, 110), rng.uniform(50, 80)])
    elif kind == "spastic":
        for n in f: f[n] = rng.uniform([60, 70, 40], [90, 110, 80])
        thumb = dict(opp=rng.uniform(0.7, 1), mcp=rng.uniform(30, 60), ip=rng.uniform(20, 60))
        spread = 0.0
    elif kind == "stiff_finger":
        for n in f: f[n] = rng.uniform([60, 80, 40], [90, 110, 80])
        stiff = rng.choice(list(BONES))
        f[stiff] = rng.uniform(0, [25, 30, 20])
        spread = rng.uniform(0, 8)
    return f, thumb, float(spread)


def synth_canonical(rng, profile=None):
    """Return (21, 3) landmarks of a right hand in the canonical frame."""
    f, thumb, spread = profile or _profile(rng)
    s = rng.uniform(0.92, 1.08, 3)
    k = np.zeros((21, 3))
    z = np.array([0.0, 0.0, 1.0])
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        base = 5 + 4 * i
        k[base] = [MCP[name][0] * s[0], MCP[name][1] * s[1], 0.0]
        abd = math.radians(SPREAD_DIR[name] * spread / 2 + rng.normal(0, 2))
        d = np.array([math.sin(abd), math.cos(abd), 0.0])
        theta = 0.0
        for j, (bone, ang) in enumerate(zip(BONES[name], f[name])):
            theta += math.radians(ang)
            k[base + j + 1] = k[base + j] + bone * s[2] * (d * math.cos(theta) + z * math.sin(theta))
    # thumb: metacarpal swings from beside the palm (opp=0) to in front of it (opp=1)
    k[1] = [0.22, 0.22, 0.04]
    meta = _unit((1 - thumb["opp"]) * np.array([0.75, 0.65, 0.05]) + thumb["opp"] * np.array([0.05, 0.65, 0.75]))
    bend = _unit(np.array([-1.0, 0.1, 0.4]) - meta * np.dot(np.array([-1.0, 0.1, 0.4]), meta))
    k[2] = k[1] + 0.45 * s[2] * meta
    theta = 0.0
    for j, (bone, ang) in enumerate(zip((0.33, 0.27), (thumb["mcp"], thumb["ip"]))):
        theta += math.radians(ang)
        k[3 + j] = k[2 + j] + bone * s[2] * (meta * math.cos(theta) + bend * math.sin(theta))
    return k


def random_rotation(rng):
    q = _unit(rng.normal(size=4))
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def synth_raw(rng, profile=None):
    """Canonical hand -> random orientation, scale, mirroring and tracker-like noise."""
    k = synth_canonical(rng, profile)
    if rng.random() < 0.5:
        k[:, 0] *= -1                       # a left hand is a mirrored right hand
    k = k @ random_rotation(rng).T * rng.uniform(0.05, 0.3) + rng.normal(0, 1, 3)
    scale = np.linalg.norm(k[9] - k[0])
    k += rng.normal(0, 0.008, k.shape) * scale
    k[:, 2] += rng.normal(0, 0.015, 21) * scale   # depth is the noisiest axis
    return k


def _ang(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def measure(k):
    """Rotation-invariant finger measurements from (21, 3) landmarks."""
    k = np.asarray(k, float)
    normal = _unit(np.cross(k[5] - k[0], k[17] - k[0]))
    out = {}
    for name, chain in zip(FINGERS, CHAINS):
        segs = [k[chain[i + 1]] - k[chain[i]] for i in range(4)]
        if name == "thumb":
            total = _ang(segs[1], segs[2]) + _ang(segs[2], segs[3])               # MCP + IP
        else:
            mcp = math.degrees(math.asin(min(1.0, abs(np.dot(_unit(segs[1]), normal)))))
            total = mcp + _ang(segs[1], segs[2]) + _ang(segs[2], segs[3])         # MCP+PIP+DIP
        out[f"{name}_flex"] = total
    proj = lambda v: v - normal * np.dot(v, normal)
    a, b = k[6] - k[5], k[18] - k[17]
    # spread is only meaningful while index and pinky stay roughly in the palm plane
    flat = min(np.linalg.norm(proj(a)) / np.linalg.norm(a), np.linalg.norm(proj(b)) / np.linalg.norm(b))
    out["spread"] = _ang(proj(a), proj(b)) if flat > 0.7 else -1.0
    width = np.linalg.norm(k[5] - k[17]) + 1e-9
    out["pinch_gap"] = 100 * np.linalg.norm(k[4] - k[8]) / width   # % of hand width
    return out


def canonicalise(k):
    """Express landmarks in the canonical hand frame (palm length = 1)."""
    k = np.asarray(k, float) - k[0]
    y = _unit(k[9])
    xt = k[5] - k[17]
    x = _unit(xt - y * np.dot(xt, y))
    z = np.cross(x, y)
    return np.stack([k @ x, k @ y, k @ z], 1) / (np.linalg.norm(k[9]) + 1e-9)


# (u axis, v axis, centre_u, centre_v) per view, image v grows downwards; 36 px per palm length
VIEWS = [(0, 1, 0.0, 1.0), (2, 1, 0.3, 1.0), (0, 2, 0.0, 0.3)]
PX = 36.0


def _ink(img, p, q, width, size):
    """Max-blend an anti-aliased segment p-q into img (only touching its bounding box)."""
    r = width / 2 + 1
    x0, x1 = max(0, int(min(p[0], q[0]) - r)), min(size, int(max(p[0], q[0]) + r) + 1)
    y0, y1 = max(0, int(min(p[1], q[1]) - r)), min(size, int(max(p[1], q[1]) + r) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    d = q - p
    L2 = float(d @ d) + 1e-9
    t = np.clip(((xs - p[0]) * d[0] + (ys - p[1]) * d[1]) / L2, 0, 1)
    dist = np.hypot(xs - (p[0] + t * d[0]), ys - (p[1] + t * d[1]))
    np.maximum(img[y0:y1, x0:x1], np.clip(width / 2 + 0.5 - dist, 0, 1), out=img[y0:y1, x0:x1])


def rasterise(c, width=2.0, size=IMG):
    """c: canonical (21, 3) landmarks -> float32 (3, size, size). Fingertips are drawn as
    dots (a zero-length segment of double width) so the direction of curl is readable."""
    img = np.zeros((3, size, size), np.float32)
    for ch, (u, v, cu, cv) in enumerate(VIEWS):
        pts = np.stack([(c[:, u] - cu) * PX + size / 2, size / 2 - (c[:, v] - cv) * PX], 1)
        for a, b in SEGMENTS:
            _ink(img[ch], pts[a], pts[b], width, size)
        for tip in (4, 8, 12, 16, 20):
            _ink(img[ch], pts[tip], pts[tip], 2 * width, size)
    return img


def sample(rng, width=None):
    raw = synth_raw(rng)
    w = width if width is not None else rng.uniform(1.6, 2.6)
    return rasterise(canonicalise(raw), width=w), measure(raw)
