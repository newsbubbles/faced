"""Steering polarity showcase: one control input, each emotion driven to BOTH poles.

Generates a baseline completion of a single neutral prompt, then re-generates the
SAME prompt (same context, same greedy decode) twice per emotion — once with the
emotion's direction subtracted (meter -> 0, "suppressed") and once added (meter ->
100, "amplified"). The only thing that differs between the two texts is the sign of
one internal direction, so any change in the output is attributable to that axis.

    python scripts/steer_showcase.py [model_key] [--alpha 3] [--tokens 48]

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


def meter(seq, e):
    return float(np.mean([m[e]["value"] for m in seq]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()

    b = load(a.model)
    reader = EmotionReader(b.key)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"STEERING POLARITY SHOWCASE  model={b.key}  coeff=+/-{a.alpha} x raw diff-of-means")
    emit(f"control input: {CONTROL!r}")
    emit("Same prompt, same context, greedy decode. Per emotion the ONLY difference")
    emit("between the two completions is the sign of one internal direction:")
    emit("  suppressed = subtract the axis (meter -> 0)   amplified = add it (meter -> 100)")
    emit("")

    base_txt, base_seq = gen(b, reader, None, a.tokens)
    emit("=" * 78)
    base_m = "  ".join(f"{e[:5]}={meter(base_seq, e):.0f}" for e in reader.emotions)
    emit("BASELINE (no steering)")
    emit(f"  meters: {base_m}")
    emit("=" * 78)
    emit(base_txt + "\n")

    for e in reader.emotions:
        L, v = reader.layer[e], reader.v_steer[e]
        emit("#" * 78)
        emit(f"# {e.upper()}   (layer {L}, baseline meter = {meter(base_seq, e):.0f})")
        emit("#" * 78)
        for label, sign in (("SUPPRESSED", -1.0), ("AMPLIFIED", +1.0)):
            steer = SteerHook(b.layers, L, v, alpha=sign * a.alpha, mode="add",
                              positions="last")
            txt, seq = gen(b, reader, steer, a.tokens)
            emit(f"--- {label:10s} coeff={sign * a.alpha:+.1f}   "
                 f"{e} meter avg = {meter(seq, e):3.0f} ---")
            emit(txt + "\n")

    out = REPO_ROOT / "artifacts" / f"steer_showcase.{b.key}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
