"""M2 verification (log-friendly): does the surprise meter spike on the
missing-attachment prompt and stay low on a neutral prompt?

Prints, per token, the surprise meter and the top-firing axis, then the peak
surprise for each prompt. The live ANSI panel is `python -m faced.cli panel`.

    python scripts/demo_surprise.py [model_key]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faced.backends import load
from faced.readout import EmotionReader
from faced.generate import stream

MISSING = "Can you review the contract I attached? Let me know if the payment terms look fair."
NEUTRAL = "What time does the library open on Saturdays?"


def run(b, reader, prompt, label, max_tokens=40):
    print(f"\n--- {label} ---\n  {prompt!r}\n")
    peak_surp, peak_by = 0.0, {}
    for ev in stream(b, prompt, reader, max_tokens=max_tokens, temperature=0.0):
        m = ev["meters"]
        surp = m["surprise"]["value"]
        top = max(reader.emotions, key=lambda e: m[e]["value"])
        peak_surp = max(peak_surp, surp)
        for e in reader.emotions:
            peak_by[e] = max(peak_by.get(e, 0.0), m[e]["value"])
        if ev["i"] < 14:
            tok = ev["t"].replace("\n", "\\n")
            print(f"  t{ev['i']:02d} {tok!r:14s} surprise={surp:5.1f}  top={top}({m[top]['value']:.0f})")
    print(f"\n  peak surprise = {peak_surp:.1f}")
    return peak_surp, peak_by


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    b = load(key)
    reader = EmotionReader(b.key)
    ms, mby = run(b, reader, MISSING, "MISSING ATTACHMENT")
    ns, _ = run(b, reader, NEUTRAL, "NEUTRAL")
    print("\n" + "=" * 60)
    print(f"  surprise peak:  missing={ms:.1f}   neutral={ns:.1f}")
    top_axis = max(mby, key=mby.get)
    print(f"  top axis on missing-attachment prompt: {top_axis} ({mby[top_axis]:.1f})")
    ok = ms > ns + 15 and ms > 55
    print(f"  M2 surprise-spike check: {'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
