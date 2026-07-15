"""Aggregate the held-out behavioural confirmation across the scale ladder.

Reads each model's abliteration-experiment JSON (which carries the abliteration
manifest, including the held-out behavioural profile) and builds:
  * a behavioural table (harmful-refusal stock->abl, benign-refusal, degeneracy),
  * a figure showing abliteration removes refusal behaviour at every scale
    without inducing benign over-refusal or degeneration.

    python scripts/behavioral_matrix.py gemma-3-1b gemma-3-4b gemma-3-12b gemma-3-27b

Writes artifacts/behavioral/behavioral.{json,png}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from faced.backends import REPO_ROOT

ABL = REPO_ROOT / "artifacts" / "abliteration"
OUT = REPO_ROOT / "artifacts" / "behavioral"


def load_model(key: str):
    p = ABL / f"{key}_vs_{key}-abl.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    m = d.get("abliteration", {})
    bs, ba = m.get("behavioral_stock", {}), m.get("behavioral_abl", {})
    axes = d.get("axes", {})
    n_moved = sum(1 for r in axes.values() if r.get("moved_beyond_noise"))
    return {
        "model": key,
        "refusal_layer": m.get("refusal_layer"),
        "refusal_auc": m.get("refusal_auc"),
        "harmful_refusal_stock": m.get("refusal_rate_stock"),
        "harmful_refusal_abl": m.get("refusal_rate_abliterated"),
        "benign_refusal_stock": bs.get("benign_refusal_rate"),
        "benign_refusal_abl": ba.get("benign_refusal_rate"),
        "degenerate_abl": ba.get("harmful_degenerate_rate"),
        "n_harmful": bs.get("n_harmful"), "n_benign": bs.get("n_benign"),
        "emotion_dirs_moved": n_moved, "n_axes": len(axes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [r for r in (load_model(k) for k in a.models) if r]
    if not rows:
        sys.exit("no abliteration JSONs found for: " + " ".join(a.models))

    json.dump({"models": rows}, open(OUT / "behavioral.json", "w", encoding="utf-8"), indent=2)

    # table
    hdr = f"  {'model':12s} {'ref@L':>5s} {'AUC':>5s} {'harm-refuse S->A':>17s} {'benign S->A':>13s} {'degen':>6s} {'emo moved':>10s}"
    print("\n  HELD-OUT BEHAVIOURAL CONFIRMATION\n" + hdr)
    for r in rows:
        def f(x): return "  n/a" if x is None else f"{x:.2f}"
        print(f"  {r['model']:12s} {str(r['refusal_layer']):>5s} {f(r['refusal_auc']):>5s} "
              f"{f(r['harmful_refusal_stock'])+' -> '+f(r['harmful_refusal_abl']):>17s} "
              f"{f(r['benign_refusal_stock'])+'->'+f(r['benign_refusal_abl']):>13s} "
              f"{f(r['degenerate_abl']):>6s} {str(r['emotion_dirs_moved'])+'/'+str(r['n_axes']):>10s}")
    print(f"  (held-out n = {rows[0]['n_harmful']} harmful / {rows[0]['n_benign']} benign)")
    _plot(rows, OUT / "behavioral.png")
    print(f"  figure -> {OUT / 'behavioral.png'}")


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r["model"].replace("gemma-3-", "") for r in rows]
    x = np.arange(len(models))
    w = 0.38

    def col(k): return [(r[k] if r[k] is not None else 0.0) for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.bar(x - w/2, col("harmful_refusal_stock"), w, label="stock", color="#1f77b4")
    axL.bar(x + w/2, col("harmful_refusal_abl"), w, label="abliterated", color="#d62728")
    axL.set_title("Abliteration removes refusal behaviour\n(held-out AdvBench)")
    axL.set_ylabel("harmful-prompt refusal rate")
    axL.set_ylim(0, 1.0)
    axL.set_xticks(x); axL.set_xticklabels(models); axL.set_xlabel("gemma-3 size")
    axL.legend(fontsize=9); axL.grid(True, axis="y", alpha=0.25)

    axR.bar(x - w/2, col("benign_refusal_stock"), w, label="benign-refuse stock", color="#2ca02c")
    axR.bar(x + w/2, col("benign_refusal_abl"), w, label="benign-refuse abl", color="#98df8a")
    axR.plot(x, col("degenerate_abl"), "o--", color="#8c564b", label="abl degeneracy")
    axR.set_title("…without breaking the model\n(benign over-refusal + degeneracy)")
    axR.set_ylabel("rate")
    axR.set_ylim(0, max(0.3, max(col("benign_refusal_stock") + col("degenerate_abl") + [0.05]) * 1.3))
    axR.set_xticks(x); axR.set_xticklabels(models); axR.set_xlabel("gemma-3 size")
    axR.legend(fontsize=8); axR.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
