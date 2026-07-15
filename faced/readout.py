"""Project residual-stream activations onto emotion directions -> live meters.

``EmotionReader`` loads the fitted readout directions + calibration and, given a
per-step all-layer residual stack ``[n_layers, d_model]``, returns a calibrated,
EMA-smoothed meter per axis. EMA speed depends on the axis' dynamics (phasic axes
react fast, tonic axes drift).
"""
from __future__ import annotations

import torch
from safetensors.torch import load_file

from .backends import REPO_ROOT
from .calibrate import Calibrator
from .config import emotions_config, axis_by_name


class EmotionReader:
    def __init__(self, model_key: str):
        self.model_key = model_key
        vecs = load_file(str(REPO_ROOT / "artifacts" / f"directions.{model_key}.safetensors"))
        self.calib = Calibrator(model_key)
        self.meta = axis_by_name()
        cfg = emotions_config()
        ema = cfg.get("ema", {"phasic": 0.5, "tonic": 0.25})

        self.emotions = [e for e in self.meta if f"{e}.readout" in vecs]
        self.readout = {e: vecs[f"{e}.readout"].float() for e in self.emotions}
        self.v_steer = {e: vecs[f"{e}.v_steer"].float() for e in self.emotions}
        self.layer = {e: self.calib.axes[e]["layer"] for e in self.emotions}
        self.alpha = {e: ema.get(self.meta[e].get("dynamics", "tonic"), 0.3)
                      for e in self.emotions}
        self._ema: dict[str, float] = {}

    def reset(self):
        self._ema.clear()

    @property
    def layers_used(self) -> list[int]:
        return sorted(set(self.layer[e] for e in self.emotions))

    def _read(self, get_resid) -> dict:
        """get_resid(layer_idx) -> a [d_model] residual tensor."""
        out = {}
        for e in self.emotions:
            h = get_resid(self.layer[e]).float().cpu()
            proj = float(h @ self.readout[e])
            m = self.calib.meter(e, proj)
            prev = self._ema.get(e, m["value"])
            a = self.alpha[e]
            val = a * m["value"] + (1 - a) * prev
            self._ema[e] = val
            bip = self.calib.is_bipolar(e)
            signed = (val - 50.0) / 50.0 if bip else val / 100.0
            out[e] = {
                "value": round(val, 1),
                "signed": round(signed, 3),
                "raw": round(proj, 3),
                "accepted": self.calib.accepted(e),
                "bipolar": bip,
            }
        return out

    def read(self, stack: torch.Tensor) -> dict:
        """stack: [n_layers, d_model] post-layer residuals for one token position."""
        return self._read(lambda L: stack[L])

    def read_captured(self, captured: dict) -> dict:
        """captured: {layer_idx: [batch, seq, d_model]} from a CaptureHook."""
        return self._read(lambda L: captured[L][0, -1])
