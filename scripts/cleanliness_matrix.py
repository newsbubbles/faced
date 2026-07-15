"""Signal-cleanliness of emotion directions across models (the scaling matrix).

For each model (already collected + fit) and each emotion, at the model's chosen
layer, reports how *clean* the direction's signal is:
  * AUC        — held-out separability (group split by family)
  * dprime     — class-mean projection separation / pooled within-class std (SNR)
  * self_cos   — within-model bootstrap self-cosine mean (direction stability;
                 higher & tighter = less noisy estimate)

Hypothesis: as model scale grows, emotion directions get cleaner (AUC↑, dprime↑,
self_cos↑, its spread↓).

    python scripts/cleanliness_matrix.py gemma-3-1b-fp32 [gemma-3-4b ...] [--boot 400]

Writes artifacts/scaling/cleanliness.json (+ heatmap .png + console matrix).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from faced.backends import REPO_ROOT
from faced.config import axis_names
from faced.activations import load_activations
from faced.directions import _cv_auc
from faced import stats

ART = REPO_ROOT / "scaling"  # placeholder; real path set in main
# approximate total parameters (billions) for ordering / the x-axis
PARAMS_B = {
    "gemma-3-1b": 1.0, "gemma-3-1b-fp32": 1.0, "gemma-2-2b": 2.6,
    "gemma-3-4b": 4.3, "gemma-3-12b": 12.2, "gemma-3-27b": 27.4,
    "gemma-4-e4b": 8.0, "qwen2.5-1.5b": 1.5,
}


def model_layers(key):
    with open(REPO_ROOT / "config" / f"calibration.{key}.json", encoding="utf-8") as f:
        return {e: c["layer"] for e, c in json.load(f)["axes"].items()}


def cleanliness(key, boot, signal="mean_gen"):
    layers = model_layers(key)
    out = {}
    for e in axis_names():
        if e not in layers:
            continue
        d = load_activations(key, e)
        L = layers[e]
        X = d[signal].numpy()[:, L, :]
        y = d["labels"].numpy().astype(int)
        groups = np.array(d["families"])
        finite = np.isfinite(X).all(axis=1)
        X, y, groups = X[finite], y[finite], groups[finite]
        u = stats.diff_of_means(X, y)
        sc = stats.summarize(stats.bootstrap_self_cosine(X, y, groups, B=boot))
        out[e] = {
            "layer": int(L),
            "auc": round(_cv_auc(X, y, groups, "mean"), 4),
            "dprime": round(stats.dprime(X, y, u), 4),
            "self_cos_mean": round(sc["mean"], 4),
            "self_cos_p05": round(sc["p05"], 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    out_dir = REPO_ROOT / "artifacts" / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(a.models, key=lambda k: PARAMS_B.get(k, 999))
    result = {"models": {}, "params_b": {m: PARAMS_B.get(m) for m in models}}
    for m in models:
        result["models"][m] = cleanliness(m, a.boot)

    axes = axis_names()
    # console matrices
    for metric in ("auc", "dprime", "self_cos_mean"):
        print(f"\n  === {metric} (rows=model by size, cols=emotion) ===")
        print(f"  {'model':18s} {'~B':>5s} " + " ".join(f"{e[:5]:>6s}" for e in axes) + "   mean")
        for m in models:
            row = result["models"][m]
            vals = [row[e][metric] for e in axes if e in row]
            cells = " ".join(f"{row[e][metric]:>6.2f}" if e in row else f"{'--':>6s}" for e in axes)
            print(f"  {m:18s} {PARAMS_B.get(m, 0):>5.1f} {cells}   {np.mean(vals):.2f}")

    with open(out_dir / "cleanliness.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  results -> {out_dir / 'cleanliness.json'}")
    _plot(result, axes, out_dir / "cleanliness.png")
    print(f"  figure  -> {out_dir / 'cleanliness.png'}")


def _plot(result, axes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(result["models"])
    if len(models) < 1:
        return
    metrics = [("auc", "held-out AUC", 0.5, 1.0),
               ("dprime", "d′ (SNR)", None, None),
               ("self_cos_mean", "bootstrap self-cosine", 0.5, 1.0)]
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 1 + 0.5 * len(models)))
    if len(metrics) == 1:
        axs = [axs]
    for ax, (mk, title, vmin, vmax) in zip(axs, metrics):
        M = np.array([[result["models"][m].get(e, {}).get(mk, np.nan) for e in axes] for m in models])
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(axes))); ax.set_xticklabels(axes, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([f"{m} ({result['params_b'].get(m)}B)" for m in models], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(models)):
            for j in range(len(axes)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="white", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Emotion-direction signal cleanliness vs model scale", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
