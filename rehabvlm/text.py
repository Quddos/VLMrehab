"""Question set, answer templates and the closed word-level tokenizer."""
import json

QUESTIONS = {
    "describe": "describe the posture .",
    "shoulders": "assess the shoulders .",
    "elbows": "assess the elbows .",
    "hips": "assess the hips .",
    "symmetry": "check symmetry .",
    "weaker": "which side is weaker ?",
    "exercise": "suggest exercises .",
}

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
PAD, BOS, SEP, EOS, UNK = range(5)


def q5(x):
    """Round to the nearest 5 degrees (number tokens are 0, 5, ..., 180)."""
    return int(max(0, min(180, 5 * round(x / 5))))


def _sh_grade(a):
    return "full range" if a >= 150 else "partial range" if a >= 90 else "limited"


def _el_grade(a):
    return "extended" if a < 15 else "partially flexed" if a < 60 else "flexed"


def _hip_grade(a):
    return "neutral" if a < 10 else "mild abduction" if a < 25 else "wide abduction"


def _asym_grade(d):
    return "minimal" if d < 15 else "moderate" if d < 40 else "significant"


def _lean(l):
    if abs(l) < 5:
        return "trunk upright"
    side = "left" if l > 0 else "right"
    return f"trunk leans {q5(abs(l))} degrees to the {side}"


def answer(task, m):
    r_sh, l_sh = q5(m["r_shoulder_abd"]), q5(m["l_shoulder_abd"])
    r_el, l_el = q5(m["r_elbow_flex"]), q5(m["l_elbow_flex"])
    r_hp, l_hp = q5(m["r_hip_abd"]), q5(m["l_hip_abd"])
    if task == "describe":
        return (f"{_lean(m['trunk_lean'])} . right shoulder abduction {r_sh} degrees . "
                f"left shoulder abduction {l_sh} degrees . right elbow flexion {r_el} degrees . "
                f"left elbow flexion {l_el} degrees . right hip abduction {r_hp} degrees . "
                f"left hip abduction {l_hp} degrees .")
    if task == "shoulders":
        d = q5(abs(m["r_shoulder_abd"] - m["l_shoulder_abd"]))
        return (f"right shoulder abduction {r_sh} degrees , {_sh_grade(r_sh)} . "
                f"left shoulder abduction {l_sh} degrees , {_sh_grade(l_sh)} . "
                f"asymmetry {d} degrees , {_asym_grade(d)} .")
    if task == "elbows":
        return (f"right elbow flexion {r_el} degrees , {_el_grade(r_el)} . "
                f"left elbow flexion {l_el} degrees , {_el_grade(l_el)} .")
    if task == "hips":
        return (f"right hip abduction {r_hp} degrees , {_hip_grade(r_hp)} . "
                f"left hip abduction {l_hp} degrees , {_hip_grade(l_hp)} . {_lean(m['trunk_lean'])} .")
    if task == "symmetry":
        du = q5(abs(m["r_shoulder_abd"] - m["l_shoulder_abd"]))
        dl = q5(abs(m["r_hip_abd"] - m["l_hip_abd"]))
        return (f"upper limb asymmetry {du} degrees , {_asym_grade(du)} . "
                f"lower limb asymmetry {dl} degrees , {_asym_grade(dl)} . {_lean(m['trunk_lean'])} .")
    if task == "weaker":
        d = m["r_shoulder_abd"] - m["l_shoulder_abd"]
        if abs(d) < 15:
            return "no clear side difference in arm elevation ."
        side = "right" if d < 0 else "left"
        return f"the {side} side shows less range , {q5(abs(d))} degrees less shoulder abduction ."
    if task == "exercise":
        worst = min(("right", m["r_shoulder_abd"]), ("left", m["l_shoulder_abd"]), key=lambda t: t[1])
        if worst[1] < 90:
            return (f"{worst[0]} shoulder is limited . try pendulum swings , "
                    f"wall finger walks and assisted arm raises .")
        if worst[1] < 150:
            return (f"{worst[0]} shoulder has partial range . try wall slides , "
                    f"band assisted abduction and overhead stretches .")
        return "range looks functional . keep up maintenance stretching and strengthening ."
    raise KeyError(task)


def build_vocab():
    words = set()
    for q in QUESTIONS.values():
        words.update(q.split())
    probe = {"r_shoulder_abd": 30, "l_shoulder_abd": 170, "r_elbow_flex": 30, "l_elbow_flex": 90,
             "r_hip_abd": 20, "l_hip_abd": 40, "trunk_lean": 10}
    variants = [probe, {k: (0 if k != "trunk_lean" else -10) for k in probe},
                {k: 120 for k in probe} | {"trunk_lean": 0, "r_hip_abd": 0},
                {**probe, "r_shoulder_abd": 170, "l_shoulder_abd": 30},
                {**probe, "r_shoulder_abd": 120, "l_shoulder_abd": 170},
                {**probe, "r_shoulder_abd": 170, "l_shoulder_abd": 160}]
    for v in variants:
        for t in QUESTIONS:
            words.update(answer(t, v).split())
    words.update(str(n) for n in range(0, 185, 5))
    return SPECIALS + sorted(words)


class Tokenizer:
    def __init__(self, vocab=None):
        self.itos = vocab or build_vocab()
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi.get(w, UNK) for w in s.split()]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return " ".join(out).replace(" .", ".").replace(" ,", ",").replace(" ?", "?")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.itos, "questions": QUESTIONS}, f)
