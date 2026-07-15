"""Cross-model emotion comparison (M5): stock vs abliterated ("heretic") twins.

Because the heretic model is a weight-edit of the stock model, their residual
stream spaces share a basis, so emotion directions can be compared by cosine.

Metrics per emotion axis:
  * cosine(readout_A, readout_B)   — did abliteration rotate/move the axis?
  * auc_A, auc_B, delta            — did separability change?
  * refusal_overlap_A / _B         — cosine between the model's refusal direction
                                     (diff-of-means on harmful vs harmless) and the
                                     emotion direction, at the emotion's layer.

The refusal-overlap is the key test of the user's question: does removing the
refusal subspace disturb emotions that overlap it (e.g. fear / conflict)?
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .backends import ModelBundle, REPO_ROOT
from .activations import capture_prompt, read_jsonl

ART = REPO_ROOT / "artifacts"
REFUSAL_DIR = REPO_ROOT / "data" / "refusal"


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def collect_refusal_acts(bundle: ModelBundle, gen_tokens: int = 16, signal: str = "mean_gen"):
    """Return (X[N, L, D], y[N]) residuals for harmful(1)/harmless(0).

    signal='mean_gen' (mean over generated tokens) or 'last_prompt' (state at the
    last instruction token — standard for the refusal direction).
    """
    rows = read_jsonl(REFUSAL_DIR / "harmful.jsonl") + read_jsonl(REFUSAL_DIR / "harmless.jsonl")
    X, y = [], []
    for r in rows:
        lp, mg = capture_prompt(bundle, r["text"], gen_tokens=gen_tokens)
        X.append((lp if signal == "last_prompt" else mg).numpy())
        y.append(int(r["label"]))
    return np.stack(X), np.array(y)


def extract_and_save_refusal(bundle: ModelBundle) -> str:
    """Per-layer refusal direction (unit diff-of-means) -> artifacts/refusal.<key>.safetensors.

    nan/inf-safe: fp16 gemma can overflow in late-layer residuals on some prompts,
    so non-finite values are ignored (via nanmean) rather than poisoning the mean.
    """
    X, y = collect_refusal_acts(bundle)
    X = np.where(np.isfinite(X), X, np.nan)
    L = X.shape[1]

    def _layer_dir(l):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d = np.nanmean(X[y == 1, l], 0) - np.nanmean(X[y == 0, l], 0)
        return _unit(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))

    dirs = np.stack([_layer_dir(l) for l in range(L)])
    out = ART / f"refusal.{bundle.key}.safetensors"
    save_file({"refusal": torch.tensor(dirs, dtype=torch.float32)},
              str(out), metadata={"model": bundle.key})
    print(f"  saved refusal directions [{L}, {X.shape[2]}] -> {out.name}")
    return str(out)


def _load_model_dirs(model_key: str):
    vecs = load_file(str(ART / f"directions.{model_key}.safetensors"))
    with open(REPO_ROOT / "config" / f"calibration.{model_key}.json", encoding="utf-8") as f:
        calib = json.load(f)["axes"]
    refusal = None
    rp = ART / f"refusal.{model_key}.safetensors"
    if rp.exists():
        refusal = load_file(str(rp))["refusal"].numpy()
    return vecs, calib, refusal


def compare(key_a: str, key_b: str, emotions: list[str]) -> dict:
    va, ca, ra = _load_model_dirs(key_a)
    vb, cb, rb = _load_model_dirs(key_b)
    result = {"model_a": key_a, "model_b": key_b, "axes": {}}
    for e in emotions:
        if f"{e}.readout" not in va or f"{e}.readout" not in vb:
            continue
        da = _unit(va[f"{e}.readout"].numpy())
        db = _unit(vb[f"{e}.readout"].numpy())
        row = {
            "cosine_ab": float(da @ db),
            "auc_a": ca[e]["auc"], "auc_b": cb[e]["auc"],
            "auc_delta": float(cb[e]["auc"] - ca[e]["auc"]),
            "layer_a": ca[e]["layer"], "layer_b": cb[e]["layer"],
        }
        if ra is not None:
            row["refusal_overlap_a"] = float(abs(_unit(ra[ca[e]["layer"]]) @ da))
        if rb is not None:
            row["refusal_overlap_b"] = float(abs(_unit(rb[cb[e]["layer"]]) @ db))
        result["axes"][e] = row
    return result


def save_report(result: dict, path=None):
    path = path or (ART / f"compare.{result['model_a']}_vs_{result['model_b']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # console table
    print(f"\n  {result['model_a']}  vs  {result['model_b']}")
    print(f"  {'axis':12s} {'cos(A,B)':>9s} {'aucA':>6s} {'aucB':>6s} {'dAUC':>6s} "
          f"{'refA':>6s} {'refB':>6s}")
    for e, r in result["axes"].items():
        print(f"  {e:12s} {r['cosine_ab']:>9.3f} {r['auc_a']:>6.3f} {r['auc_b']:>6.3f} "
              f"{r['auc_delta']:>+6.3f} {r.get('refusal_overlap_a',float('nan')):>6.3f} "
              f"{r.get('refusal_overlap_b',float('nan')):>6.3f}")
    print(f"\n  report -> {path}")
    return path
