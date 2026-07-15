"""Abliteration experiment: does removing the refusal direction move emotion directions?

Compares a stock model against its abliterated twin (same architecture, same
residual space). For each emotion, at the stock model's chosen layer:
  * cos(u_stock, u_abl)                          — did the direction move?
  * within-model bootstrap self-cosine floor      — how much moves by sampling noise?
  * verdict: cross-cosine below the 5th-pct floor => moved beyond noise
  * cos(u, refusal_dir) for stock vs abl          — did abliteration cut the emotion's
                                                    overlap with the refusal direction?
  * held-out AUC for both                          — did separability change?

    python scripts/run_abliteration_experiment.py --stock gemma-3-1b --abl gemma-3-1b-abl

Writes artifacts/abliteration/<stock>_vs_<abl>.json (+ .png figures + manifest).
Same script runs on RunPod for gemma-4-e4b vs its abliterated twin.
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

import numpy as np
from safetensors.torch import load_file

from faced.backends import REPO_ROOT
from faced.config import axis_names
from faced.activations import load_activations
from faced.directions import _cv_auc
from faced import stats

ART = REPO_ROOT / "artifacts" / "abliteration"


def load_stock_layers(stock_key):
    with open(REPO_ROOT / "config" / f"calibration.{stock_key}.json", encoding="utf-8") as f:
        return {e: c["layer"] for e, c in json.load(f)["axes"].items()}


def refusal_direction(stock_key):
    p = REPO_ROOT / "artifacts" / f"refusal_dir.{stock_key}.safetensors"
    if not p.exists():
        return None, None
    d = load_file(str(p))
    return d["refusal_dir"].numpy(), int(d["refusal_layer"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--signal", default="mean_gen")
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)

    layers = load_stock_layers(a.stock)
    r_dir, r_layer = refusal_direction(a.stock)
    emotions = [e for e in axis_names() if e in layers]

    results = {}
    for e in emotions:
        L = layers[e]
        ds = load_activations(a.stock, e)
        da = load_activations(a.abl, e)
        Xs = ds[a.signal].numpy()[:, L, :]
        Xa = da[a.signal].numpy()[:, L, :]
        y = ds["labels"].numpy().astype(int)
        groups = np.array(ds["families"])

        us = stats.diff_of_means(Xs, y)
        ua = stats.diff_of_means(Xa, y)
        cos_cross = stats.cosine(us, ua)
        floor = stats.summarize(stats.bootstrap_self_cosine(Xs, y, groups, B=a.boot))
        cross_ci = stats.summarize(stats.bootstrap_cross_cosine(Xs, Xa, y, groups, B=a.boot))
        moved = bool(cos_cross < floor["p05"])

        row = {
            "layer": int(L),
            "cos_cross": round(cos_cross, 4),
            "self_cosine_floor": {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in floor.items()},
            "cross_cosine_ci": {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in cross_ci.items()},
            "moved_beyond_noise": moved,
            "auc_stock": round(_cv_auc(Xs, y, groups, "mean"), 4),
            "auc_abl": round(_cv_auc(Xa, y, groups, "mean"), 4),
        }
        if r_dir is not None:
            row["cos_refusal_stock"] = round(abs(stats.cosine(us, r_dir)), 4)
            row["cos_refusal_abl"] = round(abs(stats.cosine(ua, r_dir)), 4)
        results[e] = row

    # manifest
    import transformers, torch, sklearn
    abl_manifest = {}
    mp = REPO_ROOT / "models" / a.abl / "abliteration_manifest.json"
    if mp.exists():
        abl_manifest = json.load(open(mp, encoding="utf-8"))
    out = {
        "stock": a.stock, "abl": a.abl, "signal": a.signal, "bootstrap": a.boot,
        "refusal_layer": r_layer,
        "abliteration": abl_manifest,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "axes": results,
    }
    outp = ART / f"{a.stock}_vs_{a.abl}.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=2)

    # console table
    print(f"\n  ABLITERATION EXPERIMENT  {a.stock}  vs  {a.abl}")
    if abl_manifest:
        print(f"  refusal rate {abl_manifest.get('refusal_rate_stock')}"
              f" -> {abl_manifest.get('refusal_rate_abliterated')}"
              f"  (refusal dir @ layer {abl_manifest.get('refusal_layer')})")
    print(f"\n  {'axis':12s} {'L':>2s} {'cos(stock,abl)':>14s} {'noise floor p05':>15s} "
          f"{'moved?':>7s} {'aucS':>5s} {'aucA':>5s} {'refS':>5s} {'refA':>5s}")
    for e, r in results.items():
        print(f"  {e:12s} {r['layer']:>2d} {r['cos_cross']:>14.3f} "
              f"{r['self_cosine_floor']['p05']:>15.3f} {str(r['moved_beyond_noise']):>7s} "
              f"{r['auc_stock']:>5.2f} {r['auc_abl']:>5.2f} "
              f"{r.get('cos_refusal_stock', float('nan')):>5.2f} "
              f"{r.get('cos_refusal_abl', float('nan')):>5.2f}")
    n_moved = sum(r["moved_beyond_noise"] for r in results.values())
    print(f"\n  {n_moved}/{len(results)} emotion directions moved beyond the sampling-noise floor")
    print(f"  results -> {outp}")

    _plot(out, ART / f"{a.stock}_vs_{a.abl}.png")
    print(f"  figure  -> {ART / f'{a.stock}_vs_{a.abl}.png'}")


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes = list(out["axes"])
    x = np.arange(len(axes))
    cc = [out["axes"][e]["cos_cross"] for e in axes]
    p05 = [out["axes"][e]["self_cosine_floor"]["p05"] for e in axes]
    p95 = [out["axes"][e]["self_cosine_floor"]["p95"] for e in axes]
    mean = [out["axes"][e]["self_cosine_floor"]["mean"] for e in axes]
    has_ref = "cos_refusal_stock" in out["axes"][axes[0]]

    fig, axs = plt.subplots(2 if has_ref else 1, 1, figsize=(10, 7 if has_ref else 4))
    if not has_ref:
        axs = [axs]
    ax = axs[0]
    ax.fill_between(x, p05, p95, color="#bbb", alpha=0.5,
                    label="within-model self-cosine floor (5–95%)")
    ax.plot(x, mean, color="#888", lw=1, ls="--")
    ax.scatter(x, cc, color="#d62728", zorder=5, label="cos(stock, abliterated)")
    for i, e in enumerate(axes):
        if out["axes"][e]["moved_beyond_noise"]:
            ax.annotate("moved", (x[i], cc[i]), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(axes, rotation=25, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"Does abliteration move emotion directions?  {out['stock']} vs {out['abl']}")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.25)

    if has_ref:
        rs = [out["axes"][e]["cos_refusal_stock"] for e in axes]
        ra = [out["axes"][e]["cos_refusal_abl"] for e in axes]
        w = 0.4
        ax2 = axs[1]
        ax2.bar(x - w/2, rs, w, label="stock", color="#1f77b4")
        ax2.bar(x + w/2, ra, w, label="abliterated", color="#d62728")
        ax2.set_xticks(x); ax2.set_xticklabels(axes, rotation=25, ha="right")
        ax2.set_ylabel("|cos(emotion, refusal dir)|")
        ax2.set_title("Emotion↔refusal-direction overlap: does abliteration cut it?")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
