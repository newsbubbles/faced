"""Fit emotion directions and select the best layer per axis.

For each emotion we fit, at every layer, a linear direction separating the
positive from the control activations, and score it by **cross-validated
held-out AUC** using a group-aware split (by template family) so a template's
variants never span train and test. The best layer is chosen per axis.

Two directions are stored per axis:
  * ``v_steer``   — raw difference-of-means (mu_pos - mu_neg), kept in activation
                    units; this is what M4 adds back to steer.
  * ``readout``   — a unit direction used to project the residual to a scalar. It
                    is either the normalized diff-of-means or a shrinkage-LDA
                    direction, whichever wins CV AUC.

Also computes the projection statistics (class-mean projections, pooled std,
threshold) that calibration turns into a 0-100% meter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .activations import load_activations
from .backends import REPO_ROOT

ART_DIR = REPO_ROOT / "artifacts"


def _diff_of_means(Xtr, ytr):
    mu1 = Xtr[ytr == 1].mean(0)
    mu0 = Xtr[ytr == 0].mean(0)
    return mu1 - mu0


def _lda_dir(Xtr, ytr, shrink=0.5):
    """Shrinkage LDA direction: (Sigma_pool + lambda I)^-1 (mu1 - mu0)."""
    mu1 = Xtr[ytr == 1].mean(0)
    mu0 = Xtr[ytr == 0].mean(0)
    Xc = np.vstack([Xtr[ytr == 1] - mu1, Xtr[ytr == 0] - mu0])
    cov = np.cov(Xc, rowvar=False)
    lam = shrink * np.trace(cov) / cov.shape[0]
    reg = cov + lam * np.eye(cov.shape[0])
    try:
        d = np.linalg.solve(reg, (mu1 - mu0))
    except np.linalg.LinAlgError:
        d = mu1 - mu0
    return d


def _cv_auc(X, y, groups, kind, n_splits=5):
    """Mean held-out AUC of a diff-of-means / LDA projection, group-aware."""
    n_splits = min(n_splits, int(min((y == 1).sum(), (y == 0).sum())))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedGroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in skf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        d = _diff_of_means(X[tr], y[tr]) if kind == "mean" else _lda_dir(X[tr], y[tr])
        d = d / (np.linalg.norm(d) + 1e-8)
        proj = X[te] @ d
        aucs.append(roc_auc_score(y[te], proj))
    return float(np.mean(aucs)) if aucs else float("nan")


def fit_axis(model_key: str, emotion: str, signal: str = "mean_gen") -> dict:
    data = load_activations(model_key, emotion)
    X_all = data[signal].numpy()           # [N, L, D]
    y = data["labels"].numpy().astype(int)  # [N]
    groups = np.array(data["families"])
    # drop numerically-unstable prompts (e.g. fp16/bf16 overflow at late layers)
    finite = np.isfinite(X_all).all(axis=(1, 2))
    if not finite.all():
        X_all, y, groups = X_all[finite], y[finite], groups[finite]
    N, L, D = X_all.shape

    # Score every layer. Shrinkage-LDA is O(d_model^3) per fold, which is prohibitive
    # for large models (d up to 5376); the experiment uses diff-of-means anyway, so
    # allow a mean-only mode (flag file config/MEAN_ONLY or env FACED_MEAN_ONLY=1).
    mean_only = (os.environ.get("FACED_MEAN_ONLY") == "1"
                 or (REPO_ROOT / "config" / "MEAN_ONLY").exists())
    best = {"auc": -1.0, "layer": 0, "kind": "mean"}
    per_layer = []
    for l in range(L):
        Xl = X_all[:, l, :]
        auc_mean = _cv_auc(Xl, y, groups, "mean")
        auc_lda = float("nan") if mean_only else _cv_auc(Xl, y, groups, "lda")
        per_layer.append({"layer": l, "auc_mean": auc_mean, "auc_lda": auc_lda})
        for kind, auc in (("mean", auc_mean), ("lda", auc_lda)):
            if not np.isnan(auc) and auc > best["auc"]:
                best = {"auc": float(auc), "layer": l, "kind": kind}

    # Refit the chosen direction on ALL data at the chosen layer.
    l = best["layer"]
    Xl = X_all[:, l, :]
    v_steer = _diff_of_means(Xl, y)
    readout = v_steer if best["kind"] == "mean" else _lda_dir(Xl, y)
    readout = readout / (np.linalg.norm(readout) + 1e-8)

    proj = Xl @ readout
    p_pos = float(proj[y == 1].mean())
    p_neg = float(proj[y == 0].mean())
    p_std = float(proj.std() + 1e-8)
    thr = 0.5 * (p_pos + p_neg)

    return {
        "emotion": emotion,
        "layer": int(l),
        "kind": best["kind"],
        "auc": float(best["auc"]),
        "v_steer": torch.tensor(v_steer, dtype=torch.float32),
        "readout": torch.tensor(readout, dtype=torch.float32),
        "p_pos": p_pos, "p_neg": p_neg, "p_std": p_std, "threshold": thr,
        "per_layer": per_layer,
        "n": int(N),
    }


def fit_all(model_key: str, emotions: list[str], min_auc: float = 0.85,
            signal: str = "mean_gen") -> dict:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    vectors, calib, report = {}, {}, {}
    for e in emotions:
        r = fit_axis(model_key, e, signal=signal)
        vectors[f"{e}.v_steer"] = r["v_steer"]
        vectors[f"{e}.readout"] = r["readout"]
        calib[e] = {k: r[k] for k in
                    ("layer", "kind", "auc", "p_pos", "p_neg", "p_std", "threshold", "n")}
        calib[e]["accepted"] = bool(r["auc"] >= min_auc)
        report[e] = {"layer": r["layer"], "kind": r["kind"], "auc": round(r["auc"], 4),
                     "accepted": calib[e]["accepted"]}
        flag = "OK " if calib[e]["accepted"] else "WEAK"
        print(f"  [{flag}] {e:12s} layer={r['layer']:2d} kind={r['kind']:4s} auc={r['auc']:.3f} (n={r['n']})")

    from safetensors.torch import save_file
    save_file(vectors, str(ART_DIR / f"directions.{model_key}.safetensors"),
              metadata={"model": model_key, "signal": signal})
    calib_out = {"model": model_key, "signal": signal, "min_auc": min_auc, "axes": calib}
    with open(REPO_ROOT / "config" / f"calibration.{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(calib_out, f, indent=2)
    with open(ART_DIR / f"report.{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    n_ok = sum(v["accepted"] for v in calib.values())
    print(f"\n  {n_ok}/{len(emotions)} axes accepted (AUC >= {min_auc})")
    return calib_out
