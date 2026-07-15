"""On-pod runner: the gemma-3 scaling ladder, both experiments.

For each model (small->large): collect (gen=4) -> fit -> cleanliness; then
abliterate -> collect -> fit -> abliteration experiment. Results (+ a live log and
per-model status) are uploaded to a private HF dataset after every model, so
partial progress survives a crash and the local orchestrator can poll it. A
DONE.marker is written at the very end.

    python runpod/run_ladder.py --models gemma-3-1b gemma-3-4b gemma-3-12b gemma-3-27b \
        --dtype bfloat16 --results-repo <user>/faced-runpod

Runs the tested standalone scripts as subprocesses so each frees the GPU on exit.
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

# gemma-3-1b is text-only; 4b/12b/27b are multimodal (Gemma3ForConditionalGeneration)
# with decoder layers nested under model.language_model.layers.
LADDER = {
    "gemma-3-1b":  {"hf": "unsloth/gemma-3-1b-it",  "arch": "causal_lm",
                    "layer_path": "model.layers"},
    "gemma-3-4b":  {"hf": "unsloth/gemma-3-4b-it",  "arch": "image_text_to_text",
                    "layer_path": "model.language_model.layers"},
    "gemma-3-12b": {"hf": "unsloth/gemma-3-12b-it", "arch": "image_text_to_text",
                    "layer_path": "model.language_model.layers"},
    "gemma-3-27b": {"hf": "unsloth/gemma-3-27b-it", "arch": "image_text_to_text",
                    "layer_path": "model.language_model.layers"},
}
LOG = REPO_ROOT / "artifacts" / "ladder_run.log"
STATUS = REPO_ROOT / "artifacts" / "ladder_status.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def register(entries: dict):
    lp = REPO_ROOT / "config" / "models.local.yaml"
    cur = yaml.safe_load(open(lp, encoding="utf-8")) if lp.exists() else {}
    cur = cur or {}
    cur.setdefault("models", {}).update(entries)
    yaml.safe_dump(cur, open(lp, "w", encoding="utf-8"), sort_keys=False)


def run(cmd, results_repo=None, hf=None):
    log("$ " + " ".join(cmd))
    p = subprocess.run([sys.executable] + cmd, cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-800:]
    log(tail)
    if p.returncode != 0:
        log(f"!! exit {p.returncode}")
    if hf and results_repo:
        _upload(hf, results_repo)
    return p.returncode == 0


def _upload(hf, repo):
    try:
        hf.upload_folder(folder_path=str(REPO_ROOT / "artifacts"), path_in_repo="artifacts",
                         repo_id=repo, repo_type="dataset", commit_message="progress")
    except Exception as e:
        log(f"(upload skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(LADDER))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--results-repo", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    # LDA layer-selection is O(d_model^3) — intractable at d=5376 (27b). Mean-only
    # picks the layer by diff-of-means AUC instead (same AUC, ~100x faster) and keeps
    # the whole ladder consistent. The abliteration cosines use diff-of-means anyway.
    (REPO_ROOT / "config" / "MEAN_ONLY").touch()
    log("mean-only fits enabled (config/MEAN_ONLY)")

    # register stock models in the local registry (per-model arch / layer path)
    register({k: {"hf_id": LADDER[k]["hf"], "arch": LADDER[k]["arch"],
                  "layer_path": LADDER[k]["layer_path"], "dtype": a.dtype,
                  "ple": False, "chat": True} for k in a.models})

    hf = None
    if a.results_repo and not a.dry:
        from huggingface_hub import HfApi
        hf = HfApi()

    status = {"models": a.models, "dtype": a.dtype, "done": [], "failed": []}
    STATUS.parent.mkdir(parents=True, exist_ok=True)

    for key in a.models:
        log(f"===== {key} ({LADDER[key]['hf']}) =====")
        steps = [
            ["scripts/collect_fast.py", key, "--gen", str(a.gen)],
            ["-m", "faced.cli", "fit", "--model", key],
            ["scripts/make_abliterated.py", key, "--out", f"{key}-abl", "--register"],
            ["scripts/collect_fast.py", f"{key}-abl", "--gen", str(a.gen)],
            ["-m", "faced.cli", "fit", "--model", f"{key}-abl"],
            ["scripts/run_abliteration_experiment.py", "--stock", key, "--abl", f"{key}-abl"],
        ]
        if a.dry:
            for s in steps:
                log("DRY $ " + " ".join(s))
            status["done"].append(key)
            continue
        ok = all(run(s, a.results_repo, hf) for s in steps)
        (status["done"] if ok else status["failed"]).append(key)
        json.dump(status, open(STATUS, "w"), indent=2)
        if hf:
            _upload(hf, a.results_repo)

    # final aggregate matrices across all models: cleanliness + behavioural confirmation
    if not a.dry:
        run(["scripts/cleanliness_matrix.py"] + a.models, a.results_repo, hf)
        run(["scripts/behavioral_matrix.py"] + a.models, a.results_repo, hf)
    (REPO_ROOT / "artifacts" / "DONE.marker").write_text(json.dumps(status), encoding="utf-8")
    if hf:
        _upload(hf, a.results_repo)
    log("LADDER COMPLETE " + json.dumps(status))


if __name__ == "__main__":
    main()
