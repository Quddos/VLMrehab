"""Registry of assessment domains: data generator, measurements, Q&A templates, vocab."""
from types import SimpleNamespace

from rehabvlm import hand, hand_text, pose, text, wrist, wrist_text


def _body_targets(m):
    return [m[k] / 90 for k in pose.ANGLE_KEYS]


def _hand_targets(m):
    t = [m[f"{f}_flex"] / 150 for f in hand.FINGERS]
    t.append(m["spread"] / 45 if m["spread"] >= 0 else -1.0)
    t.append(m["pinch_gap"] / 100)
    return t


def _wrist_targets(m):
    return [m["wrist_flex"] / 60, m["wrist_dev"] / 20]


DOMAINS = {
    "body": SimpleNamespace(sample=pose.sample, questions=text.QUESTIONS, answer=text.answer,
                            build_vocab=text.build_vocab, targets=_body_targets, n_targets=7),
    "hand": SimpleNamespace(sample=hand.sample, questions=hand_text.QUESTIONS, answer=hand_text.answer,
                            build_vocab=hand_text.build_vocab, targets=_hand_targets, n_targets=7),
    "wrist": SimpleNamespace(sample=wrist.sample, questions=wrist_text.QUESTIONS, answer=wrist_text.answer,
                             build_vocab=wrist_text.build_vocab, targets=_wrist_targets, n_targets=2),
}
