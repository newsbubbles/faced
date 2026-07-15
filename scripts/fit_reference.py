"""Calibrate meters against a neutral reference corpus.

Runs the reference corpus through the live readout path and records each axis'
median raw projection (baseline) and MAD (noise scale). Writes into
config/calibration.<model>.json:
  * ref_center     — median projection over neutral generation (bipolar recentring)
  * noise_floor_pct — dead-zone width (~1 MAD) so idle text doesn't jitter the face

    python scripts/fit_reference.py [model_key]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from faced.backends import load, REPO_ROOT
from faced.readout import EmotionReader
from faced.generate import stream
from faced.activations import read_jsonl
from faced.config import emotions_config


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    b = load(key)
    reader = EmotionReader(b.key)
    rows = read_jsonl(REPO_ROOT / "data" / "reference_corpus.jsonl")
    noise_mad = emotions_config().get("noise_floor_mad", 1.0)

    raw = {e: [] for e in reader.emotions}
    for i, r in enumerate(rows):
        for ev in stream(b, r["text"], reader, max_tokens=12, temperature=0.0):
            for e in reader.emotions:
                raw[e].append(ev["meters"][e]["raw"])
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(rows)}")

    calib_path = REPO_ROOT / "config" / f"calibration.{b.key}.json"
    calib = json.load(open(calib_path, encoding="utf-8"))
    for e in reader.emotions:
        arr = np.array(raw[e])
        center = float(np.median(arr))
        mad = float(np.median(np.abs(arr - center)) * 1.4826 + 1e-8)
        denom = abs(calib["axes"][e]["p_pos"] - calib["axes"][e]["p_neg"]) or 1e-8
        floor_pct = float(min(12.0, noise_mad * 100.0 * mad / denom))
        calib["axes"][e]["ref_center"] = center
        calib["axes"][e]["noise_floor_pct"] = round(floor_pct, 2)
        print(f"  {e:12s} center={center:8.2f} mad={mad:7.2f} floor={floor_pct:4.1f}%")
    json.dump(calib, open(calib_path, "w", encoding="utf-8"), indent=2)
    print(f"\n  updated {calib_path.name}")


if __name__ == "__main__":
    main()
