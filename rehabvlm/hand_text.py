"""Hand questions, answer templates and vocabulary (same Tokenizer interface as text.py)."""
from rehabvlm.text import SPECIALS, q5

QUESTIONS = {
    "describe": "describe the hand .",
    "fist": "assess the fist .",
    "pinch": "assess the pinch .",
    "limited": "which finger is most limited ?",
    "spread": "check finger spread .",
    "exercise": "suggest hand exercises .",
}
LONG = ["index", "middle", "ring", "pinky"]


def n5(x):
    return int(max(0, min(300, 5 * round(x / 5))))


def _fist_grade(m):
    return ("full fist" if m >= 200 else "partial fist" if m >= 120
            else "slightly flexed fingers" if m >= 40 else "open hand")


def _pinch_grade(g):
    return "pinch achieved" if g < 25 else "close to pinch" if g < 60 else "pinch not achieved"


def _spread(m):
    if m["spread"] < 0:
        return "fingers are flexed , spread not assessed"
    s = q5(m["spread"])
    grade = "wide" if s >= 40 else "moderate" if s >= 20 else "reduced"
    return f"finger spread {s} degrees , {grade}"


def answer(task, m):
    flex = {f: n5(m[f"{f}_flex"]) for f in ["thumb"] + LONG}
    mean = sum(m[f"{f}_flex"] for f in LONG) / 4
    worst = min(LONG, key=lambda f: m[f"{f}_flex"])
    gap = n5(m["pinch_gap"])
    if task == "describe":
        return (f"thumb flexion {flex['thumb']} degrees . index flexion {flex['index']} degrees . "
                f"middle flexion {flex['middle']} degrees . ring flexion {flex['ring']} degrees . "
                f"pinky flexion {flex['pinky']} degrees . {_spread(m)} .")
    if task == "fist":
        return (f"average finger flexion {n5(mean)} degrees , {_fist_grade(mean)} . "
                f"least flexed is the {worst} finger at {flex[worst]} degrees .")
    if task == "pinch":
        return (f"thumb to index gap {gap} percent of hand width , {_pinch_grade(m['pinch_gap'])} . "
                f"thumb flexion {flex['thumb']} degrees .")
    if task == "limited":
        if m[f"{worst}_flex"] >= 200:
            return "all fingers reach full flexion ."
        return f"the {worst} finger has the least flexion , {flex[worst]} degrees ."
    if task == "spread":
        return f"{_spread(m)} ."
    if task == "exercise":
        if mean < 120:
            return "finger flexion is limited . try tendon glides , soft ball squeezes and fist making ."
        if m[f"{worst}_flex"] < 150:
            return (f"the {worst} finger lags behind . try isolated {worst} finger bends "
                    f"and blocking exercises .")
        if 0 <= m["spread"] < 20:
            return "finger spread is reduced . try rubber band finger extensions and finger fans ."
        return "hand range looks functional . keep up putty exercises and finger stretches ."
    raise KeyError(task)


def build_vocab():
    words = set()
    for q in QUESTIONS.values():
        words.update(q.split())
    base = {"thumb_flex": 50, "index_flex": 100, "middle_flex": 100, "ring_flex": 100,
            "pinky_flex": 100, "spread": 30, "pinch_gap": 50}
    variants = [base, {**base, "spread": -1}, {**base, "spread": 50}, {**base, "spread": 5},
                {**base, **{f"{f}_flex": 250 for f in LONG}}, {**base, "pinch_gap": 10},
                {**base, "pinch_gap": 100}, {**base, **{f"{f}_flex": 10 for f in LONG}},
                {**base, **{f"{f}_flex": 160 for f in LONG}, "ring_flex": 140},
                {**base, **{f"{f}_flex": 160 for f in LONG}},
                {**base, **{f"{f}_flex": 160 for f in LONG}, "spread": 5}]
    for f in LONG:
        variants.append({**base, **{g: 200 for g in [f"{x}_flex" for x in LONG]}, f"{f}_flex": 60})
    for v in variants:
        for t in QUESTIONS:
            words.update(answer(t, v).split())
    words.update(str(n) for n in range(0, 305, 5))
    return SPECIALS + sorted(words)
