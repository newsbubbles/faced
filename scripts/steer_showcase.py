"""Steering showcase: one control input, steered toward each emotion in turn.

Generates a baseline completion of a single neutral prompt, then re-generates the
SAME prompt while injecting each emotion's direction (RMS-scaled so strength is
comparable across axes). Shows how each internal emotion reshapes the output text.

    python scripts/steer_showcase.py [model_key] [--alpha 5] [--tokens 45]

Writes artifacts/steer_showcase.<model>.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from faced.backends import load, REPO_ROOT
from faced.readout import EmotionReader
from faced.generate import stream
from faced.hooks import SteerHook

CONTROL = "My friend wants to quit their steady job to start a risky business. What should I tell them?"


def gen(b, reader, steer, tokens):
    txt, seq = "", []
    for ev in stream(b, CONTROL, reader, max_tokens=tokens, temperature=0.0, steer=steer):
        txt += ev["t"]
        seq.append(ev["meters"])
    return txt.strip(), seq


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    key, alpha, tokens = a.model, a.alpha, a.tokens

    b = load(key)
    reader = EmotionReader(b.key)
    lines = []

    def emit(s=""):
        print(s); lines.append(s)

    emit(f"STEERING SHOWCASE  model={b.key}  coeff={alpha} × raw diff-of-means")
    emit(f"control input: {CONTROL!r}\n")

    base_txt, _ = gen(b, reader, None, tokens)
    emit("=" * 78)
    emit("BASELINE (no steering)")
    emit("=" * 78)
    emit(base_txt + "\n")

    for e in reader.emotions:
        L = reader.layer[e]
        v = reader.v_steer[e]
        steer = SteerHook(b.layers, L, v, alpha=alpha, mode="add", positions="last")
        txt, seq = gen(b, reader, steer, tokens)
        mval = float(np.mean([m[e]["value"] for m in seq]))
        emit("=" * 78)
        emit(f"+ {e.upper()}   (steered {e} meter avg = {mval:.0f})")
        emit("=" * 78)
        emit(txt + "\n")

    out = REPO_ROOT / "artifacts" / f"steer_showcase.{b.key}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
