"""Turn a raw residual-stream projection into a 0-100% meter.

Anchored linear map: the control-class mean projection -> 0, the positive-class
mean projection -> 100, clamped. For a bipolar axis (e.g. confidence), 0 is the
negative pole (uncertainty), 100 the positive pole, 50 the neutral midpoint.

Optionally recentres to a neutral reference corpus (baseline drift) and applies a
noise floor (suppress movement within +/-1 MAD of neutral) so idle text doesn't
make the face twitch.
"""
from __future__ import annotations

import json

from .backends import REPO_ROOT
from .config import axis_by_name


def load_calibration(model_key: str) -> dict:
    with open(REPO_ROOT / "config" / f"calibration.{model_key}.json", "r", encoding="utf-8") as f:
        return json.load(f)


class Calibrator:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.calib = load_calibration(model_key)
        self.axes = self.calib["axes"]
        self._meta = axis_by_name()

    def is_bipolar(self, emotion: str) -> bool:
        return bool(self._meta.get(emotion, {}).get("bipolar", False))

    def accepted(self, emotion: str) -> bool:
        return bool(self.axes[emotion].get("accepted", True))

    def meter(self, emotion: str, proj: float) -> dict:
        """Map a projection to {value: 0..100, signed: -1..1}.

        Meter neutral point is 0% for a unipolar axis (its control class already
        sits at p_neg->0) and 50% for a bipolar axis. Reference-corpus recentring
        only applies to bipolar axes, where "typical generation" should read as
        neutral rather than pinned to one pole.
        """
        a = self.axes[emotion]
        bip = self.is_bipolar(emotion)
        p_pos, p_neg = a["p_pos"], a["p_neg"]
        denom = (p_pos - p_neg) or 1e-8
        c = a.get("ref_center")
        if c is not None and bip:
            proj = proj - (c - a["threshold"])
        value = 100.0 * (proj - p_neg) / denom
        neutral = 50.0 if bip else 0.0
        floor = a.get("noise_floor_pct")
        if floor and abs(value - neutral) < floor:
            value = neutral
        value = max(0.0, min(100.0, value))
        signed = (value - 50.0) / 50.0 if bip else value / 100.0
        return {"value": value, "signed": signed}
