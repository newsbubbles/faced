"""Plot every emotion meter over generation time (token index).

Runs one or more prompts through the model and plots all 7 emotion axes as line
plots vs token index, so you can watch the internal state move as it generates.

    python scripts/plot_emotions.py [model_key] [--tokens 55]

Writes artifacts/emotions_over_time.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faced.backends import load, REPO_ROOT
from faced.readout import EmotionReader
from faced.generate import stream

# distinct, high-contrast colour per axis
COLORS = {
    "surprise": "#ff7f0e", "confidence": "#1f77b4", "curiosity": "#2ca02c",
    "confusion": "#9467bd", "frustration": "#d62728", "fear": "#8c564b",
    "warmth": "#e377c2",
}

PROMPTS = [
    ("missing attachment (expect a SURPRISE spike, then confidence)",
     "Can you review the contract I attached? Let me know if the payment terms look fair."),
    ("repeated failure (expect FRUSTRATION / CONFUSION)",
     "This is the fifth time the build has failed for the same reason and nothing I try fixes it."),
    ("gratitude (expect WARMTH)",
     "Thank you so much — you have been incredibly kind and patient with me today."),
]


def collect(b, reader, prompt, max_tokens):
    series = {e: [] for e in reader.emotions}
    toks = []
    for ev in stream(b, prompt, reader, max_tokens=max_tokens, temperature=0.0):
        for e in reader.emotions:
            series[e].append(ev["meters"][e]["value"])
        toks.append(ev["t"])
    return series, toks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--tokens", type=int, default=55)
    a = ap.parse_args()
    key, tokens = a.model, a.tokens

    b = load(key)
    reader = EmotionReader(b.key)

    n = len(PROMPTS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.4 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, prompt) in zip(axes, PROMPTS):
        series, toks = collect(b, reader, prompt, tokens)
        x = range(len(toks))
        for e in reader.emotions:
            ax.plot(x, series[e], label=e, color=COLORS.get(e), lw=2.0, alpha=0.9)
        ax.axhline(50, color="#bbb", lw=0.8, ls="--", alpha=0.6)  # bipolar neutral
        ax.set_ylim(-2, 103)
        ax.set_ylabel("meter (0-100)")
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(True, alpha=0.25)
        text = "".join(toks).strip().replace("\n", " ")
        ax.text(0.0, -0.34, "→ " + text[:150] + ("…" if len(text) > 150 else ""),
                transform=ax.transAxes, fontsize=8, color="#555", va="top")
    axes[-1].set_xlabel("generated token index (time →)")
    axes[0].legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.32),
                   fontsize=9, frameon=False)
    fig.suptitle(f"faced — emotion meters over generation time ({b.key})",
                 y=1.0, fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO_ROOT / "artifacts" / "emotions_over_time.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
