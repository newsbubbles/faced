"""M1 driver: collect activations then fit + validate emotion directions.

    python scripts/fit_all.py [model_key] [--recollect] [--signal mean_gen|last_prompt]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faced.backends import load, REPO_ROOT
from faced.config import emotions_config, axis_names
from faced.activations import collect_all, ACT_DIR
from faced.directions import fit_all


def main():
    args = sys.argv[1:]
    recollect = "--recollect" in args
    signal = "mean_gen"
    if "--signal" in args:
        signal = args[args.index("--signal") + 1]
    key = next((a for a in args if not a.startswith("--") and a not in ("mean_gen", "last_prompt")), None)

    cfg = emotions_config()
    emotions = axis_names()
    min_auc = cfg.get("min_auc", 0.85)

    b = load(key)
    print(f"model '{b.key}' ({b.hf_id}) | axes: {emotions}")

    need = recollect or any(
        not (ACT_DIR / b.key / f"{e}.safetensors").exists() for e in emotions)
    if need:
        print("collecting activations ...")
        t0 = time.time()
        collect_all(b, emotions)
        print(f"  collected in {time.time()-t0:.1f}s")
    else:
        print("activations present (use --recollect to redo)")

    print(f"\nfitting directions (signal={signal}) ...")
    fit_all(b.key, emotions, min_auc=min_auc, signal=signal)
    print(f"\nartifacts:\n  config/calibration.{b.key}.json\n  "
          f"artifacts/directions.{b.key}.safetensors\n  artifacts/report.{b.key}.json")


if __name__ == "__main__":
    main()
