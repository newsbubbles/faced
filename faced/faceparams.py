"""Map calibrated emotion meters onto a shared FACS-lite face parameter vector.

A single linear map (weight rows in config/emotions.yaml) routes every axis into
the same set of facial control channels, so simultaneous emotions *blend* into a
coherent microexpression instead of fighting. Each param is clamped to [-1, 1].

Uses each axis' ``signed`` meter: unipolar in [0,1], bipolar in [-1,1] (so an
axis' negative pole flips its facial contribution).
"""
from __future__ import annotations

from .config import emotions_config


class FaceMapper:
    def __init__(self):
        face = emotions_config()["face"]
        self.params = list(face["params"])
        self.weights = face["weights"]

    def to_params(self, meters: dict) -> dict:
        p = {k: 0.0 for k in self.params}
        for emo, row in self.weights.items():
            if emo not in meters:
                continue
            s = meters[emo]["signed"]
            if not meters[emo].get("accepted", True):
                continue  # weak axes don't drive the face
            for param, w in row.items():
                p[param] = p.get(param, 0.0) + float(w) * s
        return {k: max(-1.0, min(1.0, v)) for k, v in p.items()}
