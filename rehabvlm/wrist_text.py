"""Wrist questions, answer templates and vocabulary.

Grades assume the patient is actively moving as far as they can in the direction shown.
Typical active ranges: flexion 60-80, extension 55-70, radial deviation 15-20, ulnar 25-30 degrees.
"""
from rehabvlm.text import SPECIALS, q5

QUESTIONS = {
    "describe": "describe the wrist .",
    "bend": "assess wrist bending .",
    "side": "assess wrist side bending .",
    "exercise": "suggest wrist exercises .",
}


def _fe(m):
    """('flexion' | 'extension', degrees) of the current wrist posture."""
    f = m["wrist_flex"]
    return ("flexion", f) if f >= 0 else ("extension", -f)


def _dv(m):
    d = m["wrist_dev"]
    return ("radial", d) if d >= 0 else ("ulnar", -d)


def _fe_grade(kind, a):
    full, part = (60, 30) if kind == "flexion" else (55, 25)
    return "full range" if a >= full else "partial range" if a >= part else "limited"


def _dv_grade(kind, a):
    full, part = (15, 7) if kind == "radial" else (25, 12)
    return "full range" if a >= full else "partial range" if a >= part else "limited"


TYPICAL = {"flexion": "60 to 80", "extension": "55 to 70", "radial": "15 to 20", "ulnar": "25 to 30"}


def answer(task, m):
    fk, fa = _fe(m)
    dk, da = _dv(m)
    if task == "describe":
        return f"wrist {fk} {q5(fa)} degrees . {dk} deviation {q5(da)} degrees ."
    if task == "bend":
        return (f"wrist {fk} {q5(fa)} degrees , {_fe_grade(fk, fa)} . "
                f"typical {fk} is {TYPICAL[fk]} degrees .")
    if task == "side":
        return (f"{dk} deviation {q5(da)} degrees , {_dv_grade(dk, da)} . "
                f"typical {dk} deviation is {TYPICAL[dk]} degrees .")
    if task == "exercise":
        if fa >= 15:
            if _fe_grade(fk, fa) == "full range":
                return "wrist range looks functional . keep up gentle stretches and grip strengthening ."
            if fk == "flexion":
                return "wrist flexion is reduced . try active wrist curls , flexion stretches and towel wringing ."
            return "wrist extension is reduced . try reverse wrist curls , prayer stretches and tabletop extension stretches ."
        if da >= 5:
            if _dv_grade(dk, da) == "full range":
                return "wrist range looks functional . keep up gentle stretches and grip strengthening ."
            return f"{dk} deviation is reduced . try {dk} deviation slides and light hammer lifts ."
        return "wrist is near neutral . try active wrist circles and flexion and extension range exercises ."
    raise KeyError(task)


def build_vocab():
    words = set()
    for q in QUESTIONS.values():
        words.update(q.split())
    for f in (-70, -40, -10, 0, 10, 20, 40, 70):
        for d in (-30, -15, -8, -2, 0, 3, 10, 20):
            for t in QUESTIONS:
                words.update(answer(t, {"wrist_flex": f, "wrist_dev": d}).split())
    words.update(str(n) for n in range(0, 185, 5))
    return SPECIALS + sorted(words)
