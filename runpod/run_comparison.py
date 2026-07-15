"""M5a (RunPod): Gemma-4 stock vs heretic emotion-landscape comparison.

Requires transformers >= 5.5 (model_type gemma4) and a 24 GB+ GPU. For each model
it collects emotion activations, fits directions, and extracts a refusal
direction; then compares the two from saved artifacts and writes a report + figure.

    python runpod/run_comparison.py

Verify the Gemma-4 layer path once (it is nested under language_model):
    python -c "from faced.backends import load; b=load('gemma-4-e4b'); \
               print(b.layer_path, b.n_layers, b.d_model)"
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from faced import backends
from faced.backends import load
from faced.config import axis_names, emotions_config
from faced.activations import collect_all, ACT_DIR
from faced.directions import fit_all
from faced.compare import extract_and_save_refusal, compare, save_report

MODELS = ["gemma-4-e4b", "gemma-4-e4b-heretic"]


def prep(key, emotions, min_auc):
    b = load(key)
    print(f"\n=== {key} === layers@{b.layer_path} n_layers={b.n_layers} d_model={b.d_model} ple={b.ple}")
    if any(not (ACT_DIR / b.key / f"{e}.safetensors").exists() for e in emotions):
        collect_all(b, emotions)
    fit_all(b.key, emotions, min_auc=min_auc)
    extract_and_save_refusal(b)
    # free VRAM before loading the second 15 GB model
    del b
    load.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes = list(result["axes"])
    cos = [result["axes"][e]["cosine_ab"] for e in axes]
    ra = [result["axes"][e].get("refusal_overlap_a", 0) for e in axes]
    rb = [result["axes"][e].get("refusal_overlap_b", 0) for e in axes]
    x = range(len(axes))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.bar(x, cos, color="#5aa9ff")
    ax1.set_title("emotion-direction cosine: stock vs heretic (1.0 = unchanged)")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(axes, rotation=30, ha="right")
    ax1.set_ylim(0, 1)
    w = 0.4
    ax2.bar([i - w/2 for i in x], ra, w, label="stock", color="#888")
    ax2.bar([i + w/2 for i in x], rb, w, label="heretic", color="#f66")
    ax2.set_title("refusal-direction overlap with each emotion (per model)")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(axes, rotation=30, ha="right")
    ax2.legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "artifacts" / \
        f"compare.{result['model_a']}_vs_{result['model_b']}.png"
    fig.savefig(out, dpi=120)
    print(f"  figure -> {out}")


def main():
    emotions = axis_names()
    min_auc = emotions_config().get("min_auc", 0.85)
    for k in MODELS:
        prep(k, emotions, min_auc)
    result = compare(MODELS[0], MODELS[1], emotions)
    save_report(result)
    try:
        plot(result)
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
