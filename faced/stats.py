"""Statistics for the abliteration experiment.

The central question: does removing the refusal direction move an emotion direction
*more than finite-prompt sampling noise would*? We answer it with a within-model
**bootstrap noise floor**:

  * point estimate: u = diff-of-means on the full contrastive set.
  * noise floor:    resample the prompts (by template family, with replacement) B
                    times, refit u_b, and record cos(u_b, u_full). This distribution
                    is how stable the *same model's* direction is under resampling.
  * cross-model:    cos(u_stock, u_abl).

If the cross-model cosine falls below (e.g.) the 5th percentile of the within-model
self-cosine distribution, abliteration moved the direction beyond sampling noise.

All randomness is seeded (np.random.default_rng(b)) so the whole test recreates.
"""
from __future__ import annotations

import numpy as np


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine(a, b):
    return float(unit(a) @ unit(b))


def diff_of_means(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def dprime(X, y, direction):
    """Signal-to-noise of the projection onto `direction`: class-mean separation in
    units of pooled within-class std. Larger = cleaner, more separable signal."""
    p = X @ unit(direction)
    mu1, mu0 = p[y == 1].mean(), p[y == 0].mean()
    pooled = np.sqrt(0.5 * (p[y == 1].var(ddof=1) + p[y == 0].var(ddof=1))) + 1e-8
    return float((mu1 - mu0) / pooled)


def _resample_indices(groups, seed):
    """Bootstrap by group (template family): sample groups w/ replacement, take their rows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    picks = rng.choice(uniq, size=len(uniq), replace=True)
    idx = []
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    for g in picks:
        idx.append(by_group[g])
    return np.concatenate(idx)


def bootstrap_self_cosine(X, y, groups, B=300):
    """cos(u_bootstrap, u_full) distribution for one model at one layer."""
    full = diff_of_means(X, y)
    out = []
    for b in range(B):
        idx = _resample_indices(groups, b)
        yb = y[idx]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        out.append(cosine(diff_of_means(X[idx], yb), full))
    return np.array(out)


def bootstrap_cross_cosine(Xa, Xb, y, groups, B=300):
    """Paired bootstrap of cos(u_a, u_b): resample once, refit both models on it.

    Xa, Xb are activations of the SAME prompts (same order) from two models.
    """
    out = []
    for b in range(B):
        idx = _resample_indices(groups, b)
        yb = y[idx]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        out.append(cosine(diff_of_means(Xa[idx], yb), diff_of_means(Xb[idx], yb)))
    return np.array(out)


def summarize(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }
