"""Lightweight tests that don't need a model loaded.

Run directly (python tests/test_faced.py) or via pytest.
Covers: layer-output unpack/repack, diff-of-means separability + CV AUC,
calibration meter monotonicity/clamping, and face-param blending/clamp.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from faced.hooks import unpack_layer_output, repack_layer_output
from faced.directions import _diff_of_means, _cv_auc


def test_unpack_repack():
    t = torch.randn(1, 4, 8)
    assert unpack_layer_output(t) is t
    tup = (t, "kv")
    assert unpack_layer_output(tup) is t
    new = torch.zeros_like(t)
    assert repack_layer_output(t, new) is new
    r = repack_layer_output(tup, new)
    assert isinstance(r, tuple) and r[0] is new and r[1] == "kv"


def test_diff_of_means_separates():
    rng = np.random.default_rng(0)
    d = 32
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    n = 40
    Xpos = rng.normal(size=(n, d)) + 3.0 * direction
    Xneg = rng.normal(size=(n, d)) - 3.0 * direction
    X = np.vstack([Xpos, Xneg])
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    v = _diff_of_means(X, y)
    # recovered direction aligns with the planted one
    cos = float(v @ direction / (np.linalg.norm(v) + 1e-8))
    assert cos > 0.8, cos
    # group-aware CV AUC is high on a clearly separable set
    groups = np.arange(2 * n) % 10
    auc = _cv_auc(X, y, groups, "mean")
    assert auc > 0.95, auc


def test_meter_monotonic_and_clamped():
    # replicate the anchored linear map used by Calibrator.meter
    p_pos, p_neg = 5.0, -1.0
    denom = p_pos - p_neg
    def value(proj):
        return max(0.0, min(100.0, 100.0 * (proj - p_neg) / denom))
    vals = [value(p) for p in (-3, -1, 0, 2, 5, 9)]
    assert vals[0] == 0.0 and vals[-1] == 100.0
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert abs(value((p_pos + p_neg) / 2) - 50.0) < 1e-6


def test_face_blend_clamp():
    from faced.faceparams import FaceMapper
    fm = FaceMapper()
    meters = {e: {"signed": 1.0, "accepted": True} for e in fm.weights}
    p = fm.to_params(meters)
    assert set(p) == set(fm.params)
    assert all(-1.0 <= v <= 1.0 for v in p.values())
    # a weak axis contributes nothing
    meters2 = {e: {"signed": 1.0, "accepted": False} for e in fm.weights}
    p2 = fm.to_params(meters2)
    assert all(abs(v) < 1e-9 for v in p2.values())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
