"""Parse the numbers a model stated in its 'describe' answer back into measurement keys."""
import re


def parse_describe(domain, text):
    """Numbers the model stated in its 'describe' answer, keyed like the measurements."""
    out = {}
    if domain == "body":
        for side, s in (("right", "r"), ("left", "l")):
            for phrase, key in ((f"{side} shoulder abduction", f"{s}_shoulder_abd"),
                                (f"{side} elbow flexion", f"{s}_elbow_flex"),
                                (f"{side} hip abduction", f"{s}_hip_abd")):
                m = re.search(phrase + r" (\d+) degrees", text)
                if m:
                    out[key] = float(m.group(1))
        m = re.search(r"trunk leans (\d+) degrees to the (left|right)", text)
        out["trunk_lean"] = (float(m.group(1)) * (1 if m.group(2) == "left" else -1)) if m else 0.0
    elif domain == "hand":
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            m = re.search(f + r" flexion (\d+) degrees", text)
            if m:
                out[f"{f}_flex"] = float(m.group(1))
        m = re.search(r"finger spread (\d+) degrees", text)
        if m:
            out["spread"] = float(m.group(1))
    else:
        m = re.search(r"wrist (flexion|extension) (\d+) degrees", text)
        if m:
            out["wrist_flex"] = float(m.group(2)) * (1 if m.group(1) == "flexion" else -1)
        m = re.search(r"(radial|ulnar) deviation (\d+) degrees", text)
        if m:
            out["wrist_dev"] = float(m.group(2)) * (1 if m.group(1) == "radial" else -1)
    return out
