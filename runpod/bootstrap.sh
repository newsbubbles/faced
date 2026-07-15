#!/usr/bin/env bash
# Runs on the pod. Installs deps, pulls the code, runs the ladder, uploads results.
set -x
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace
mkdir -p /workspace/artifacts
echo "bootstrap start $(date)" > /workspace/artifacts/startup.log

pip install -q -U "transformers>=5.5" safetensors "huggingface_hub>=0.24" scikit-learn \
    numpy einops pyyaml matplotlib accelerate >> /workspace/artifacts/startup.log 2>&1

huggingface-cli download fractalnature/faced-runpod code.tar.gz --repo-type dataset \
    --local-dir /workspace >> /workspace/artifacts/startup.log 2>&1
tar xzf /workspace/code.tar.gz -C /workspace

python /workspace/runpod/run_ladder.py \
    --models gemma-3-1b gemma-3-4b gemma-3-12b gemma-3-27b \
    --dtype bfloat16 --results-repo fractalnature/faced-runpod \
    >> /workspace/artifacts/startup.log 2>&1

# safety-net final upload of the full artifacts dir
huggingface-cli upload fractalnature/faced-runpod /workspace/artifacts artifacts \
    --repo-type dataset >> /workspace/artifacts/startup.log 2>&1 || true
echo "bootstrap done $(date)" >> /workspace/artifacts/startup.log
huggingface-cli upload fractalnature/faced-runpod /workspace/artifacts/startup.log \
    artifacts/startup.log --repo-type dataset || true
