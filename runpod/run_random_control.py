"""On-pod random-ablation control (§7.5 fix).

Per model, fits the stock once, then builds BOTH a refusal-abliterated twin and a
random-direction-abliterated twin (matched procedure) and runs the direction-
comparison against the shared stock. If random ablation moves/degrades a model as
much as refusal ablation does, the "movement" is generic weight-surgery
destabilization, not a refusal-specific effect.

    python runpod/run_random_control.py --models gemma-3-1b gemma-3-4b gemma-3-12b \
        --dtype bfloat16 --results-repo <user>/faced-runpod
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from faced.backends import REPO_ROOT

LADDER = {
    "gemma-3-1b":  {"hf": "unsloth/gemma-3-1b-it",  "arch": "causal_lm",          "lp": "model.layers"},
    "gemma-3-4b":  {"hf": "unsloth/gemma-3-4b-it",  "arch": "image_text_to_text", "lp": "model.language_model.layers"},
    "gemma-3-12b": {"hf": "unsloth/gemma-3-12b-it", "arch": "image_text_to_text", "lp": "model.language_model.layers"},
    "gemma-3-27b": {"hf": "unsloth/gemma-3-27b-it", "arch": "image_text_to_text", "lp": "model.language_model.layers"},
}
LOG = REPO_ROOT / "artifacts" / "randctl_run.log"


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def register(entries):
    lp = REPO_ROOT / "config" / "models.local.yaml"
    cur = (yaml.safe_load(open(lp, encoding="utf-8")) if lp.exists() else {}) or {}
    cur.setdefault("models", {}).update(entries)
    yaml.safe_dump(cur, open(lp, "w", encoding="utf-8"), sort_keys=False)


def run(cmd):
    log("$ " + " ".join(cmd))
    p = subprocess.run([sys.executable] + cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    log(((p.stdout or "")[-1200:]) + ((p.stderr or "")[-500:]))
    return p.returncode == 0


def upload(hf, repo):
    if hf:
        try:
            hf.upload_folder(folder_path=str(REPO_ROOT / "artifacts"), path_in_repo="artifacts",
                             repo_id=repo, repo_type="dataset", commit_message="randctl")
        except Exception as e:
            log(f"(upload skip: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-1b", "gemma-3-4b", "gemma-3-12b"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--results-repo", default=None)
    a = ap.parse_args()

    register({k: {"hf_id": LADDER[k]["hf"], "arch": LADDER[k]["arch"],
                  "layer_path": LADDER[k]["lp"], "dtype": a.dtype, "ple": False, "chat": True}
              for k in a.models})
    (REPO_ROOT / "config" / "MEAN_ONLY").touch()  # fast fits at scale

    hf = None
    if a.results_repo:
        from huggingface_hub import HfApi
        hf = HfApi()

    status = {"done": [], "failed": []}
    g = str(a.gen)
    for m in a.models:
        log(f"===== {m} =====")
        ok = (run(["scripts/collect_fast.py", m, "--gen", g]) and
              run(["-m", "faced.cli", "fit", "--model", m]) and
              # refusal-ablated twin
              run(["scripts/make_abliterated.py", m, "--out", f"{m}-abl", "--register"]) and
              run(["scripts/collect_fast.py", f"{m}-abl", "--gen", g]) and
              run(["-m", "faced.cli", "fit", "--model", f"{m}-abl"]) and
              run(["scripts/run_abliteration_experiment.py", "--stock", m, "--abl", f"{m}-abl"]) and
              # random-direction control twin
              run(["scripts/make_abliterated.py", m, "--out", f"{m}-rand", "--random", "--seed", "0", "--register"]) and
              run(["scripts/collect_fast.py", f"{m}-rand", "--gen", g]) and
              run(["-m", "faced.cli", "fit", "--model", f"{m}-rand"]) and
              run(["scripts/run_abliteration_experiment.py", "--stock", m, "--abl", f"{m}-rand"]))
        (status["done"] if ok else status["failed"]).append(m)
        json.dump(status, open(REPO_ROOT / "artifacts" / "randctl_status.json", "w"), indent=2)
        upload(hf, a.results_repo)

    (REPO_ROOT / "artifacts" / "RANDCTL_DONE.marker").write_text(json.dumps(status))
    upload(hf, a.results_repo)
    log("RANDOM CONTROL COMPLETE " + json.dumps(status))


if __name__ == "__main__":
    main()
