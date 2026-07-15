"""M4: causal steering / ablation evaluations.

Reproduces the paper's core causal claim at small scale:
  1. warmth alpha-sweep  — inject +/- warmth on a neutral prompt; tone should
     shift monotonically cold<->warm while staying coherent.
  2. ablation removes the surprise spike — on the missing-attachment prompt,
     ablating the surprise direction lowers the surprise readout AND changes the
     response (the causal evidence; the meter drop alone is partly definitional).
  3. steering confusion matrix — steer each axis, measure every axis' mean meter;
     the diagonal should dominate (steering is reasonably specific).

    python scripts/eval_steering.py [model_key] [--alpha 8]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from faced.backends import load
from faced.readout import EmotionReader
from faced.generate import stream
from faced.hooks import SteerHook

NEUTRAL = "Tell me about your plans for the weekend."
MISSING = "Can you review the contract I attached? Let me know if the terms look fair."


def gen_text(b, prompt, reader, steer=None, max_tokens=60):
    text, meters_seq = "", []
    for ev in stream(b, prompt, reader, max_tokens=max_tokens, temperature=0.0, steer=steer):
        text += ev["t"]
        meters_seq.append({e: ev["meters"][e]["value"] for e in reader.emotions})
    return text.strip(), meters_seq


def peak(meters_seq, emo):
    return max((m[emo] for m in meters_seq), default=0.0)


def mean_meters(meters_seq):
    if not meters_seq:
        return {}
    keys = meters_seq[0].keys()
    return {k: float(np.mean([m[k] for m in meters_seq])) for k in keys}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--alpha", type=float, default=4.0)
    a = ap.parse_args()
    key, alpha = a.model, a.alpha

    b = load(key)
    reader = EmotionReader(b.key)
    print(f"model={b.key}  alpha={alpha}\n")

    # 1) warmth alpha-sweep -------------------------------------------------
    print("=" * 70)
    print("1) WARMTH alpha-sweep on a neutral prompt (tone should shift cold<->warm)")
    print("=" * 70)
    if "warmth" in reader.emotions:
        L = reader.layer["warmth"]
        v = reader.v_steer["warmth"]
        for a in (-alpha, -alpha / 2, 0.0, alpha / 2, alpha):
            steer = None if a == 0 else SteerHook(b.layers, L, v, alpha=a, mode="add", positions="last")
            txt, seq = gen_text(b, NEUTRAL, reader, steer=steer, max_tokens=45)
            wm = mean_meters(seq).get("warmth", float("nan"))
            print(f"\n  coeff={a:+.1f} (warmth_meter={wm:4.0f}): {txt}")
    else:
        print("  (warmth axis not available)")

    # 2) suppress / amplify the surprise spike ------------------------------
    print("\n" + "=" * 70)
    print("2) SUPPRESS vs AMPLIFY surprise on the missing-attachment prompt")
    print("   (add -/+ the diff-of-means direction; a pure ablation to 0 is not")
    print("    'neutral' for an asymmetric readout, so we steer toward each pole)")
    print("=" * 70)
    if "surprise" in reader.emotions:
        L = reader.layer["surprise"]
        v = reader.v_steer["surprise"]
        base_txt, base_seq = gen_text(b, MISSING, reader, steer=None, max_tokens=50)
        supp = SteerHook(b.layers, L, v, alpha=-alpha, mode="add", positions="last")
        supp_txt, supp_seq = gen_text(b, MISSING, reader, steer=supp, max_tokens=50)
        amp = SteerHook(b.layers, L, v, alpha=+alpha / 2, mode="add", positions="last")
        amp_txt, amp_seq = gen_text(b, MISSING, reader, steer=amp, max_tokens=50)
        print(f"\n  surprise peak  baseline={peak(base_seq,'surprise'):5.1f}"
              f"   suppressed={peak(supp_seq,'surprise'):5.1f}"
              f"   amplified={peak(amp_seq,'surprise'):5.1f}")
        print(f"\n  baseline  : {base_txt}")
        print(f"\n  suppressed: {supp_txt}")
    else:
        print("  (surprise axis not available)")

    # 3) steering confusion matrix ------------------------------------------
    print("\n" + "=" * 70)
    print("3) STEERING confusion matrix (mean meter change vs neutral baseline)")
    print("=" * 70)
    emos = reader.emotions
    cm_coeff = min(alpha, 2.5)   # stay in the coherent range for the matrix
    _, base_seq = gen_text(b, NEUTRAL, reader, steer=None, max_tokens=40)
    base_mean = mean_meters(base_seq)
    rows = {}
    for se in emos:
        L = reader.layer[se]
        v = reader.v_steer[se]
        steer = SteerHook(b.layers, L, v, alpha=cm_coeff, mode="add", positions="last")
        _, seq = gen_text(b, NEUTRAL, reader, steer=steer, max_tokens=40)
        mm = mean_meters(seq)
        rows[se] = {me: mm[me] - base_mean[me] for me in emos}

    hdr = "steer\\meas  " + " ".join(f"{e[:5]:>6s}" for e in emos)
    print("\n  " + hdr)
    for se in emos:
        line = f"  {se[:10]:10s} " + " ".join(f"{rows[se][me]:+6.1f}" for me in emos)
        star = "  <- diag" if max(rows[se], key=rows[se].get) == se else ""
        print(line + star)
    diag_wins = sum(max(rows[se], key=rows[se].get) == se for se in emos)
    print(f"\n  diagonal dominant for {diag_wins}/{len(emos)} axes")


if __name__ == "__main__":
    main()
