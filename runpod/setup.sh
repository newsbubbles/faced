#!/usr/bin/env bash
# RunPod environment for M5 (Gemma-4 stock-vs-heretic comparison + SAE).
# Needs a 24 GB+ GPU (A6000 / 4090 / A100). Gemma-4-E4B is ~15 GB in bf16 and
# requires transformers >= 5.5 (model_type gemma4).
set -e

# torch is preinstalled on standard RunPod PyTorch images; install the rest.
pip install --upgrade "transformers>=5.5" accelerate safetensors "huggingface_hub>=0.24" \
    scikit-learn numpy einops pyyaml matplotlib fastapi uvicorn

# Optional: sae_lens for production SAEs. train_sae.py has a dependency-free
# top-k SAE fallback, so this is not required.
pip install sae_lens || echo "sae_lens not installed (optional; train_sae.py works without it)"

cat <<'NOTE'

Next steps on the pod:
  1) huggingface-cli login              # and accept the Gemma-4 licenses on:
        https://huggingface.co/google/gemma-4-E4B-it
     (igorls/gemma-4-E4B-it-heretic is ungated)
  2) python -c "from faced.backends import load; b=load('gemma-4-e4b'); \
                print(b.layer_path, b.n_layers, b.d_model)"   # verify layer path
  3) python runpod/run_comparison.py    # M5a: emotion comparison + refusal overlap
  4) python runpod/harvest_activations.py gemma-4-e4b --layer 21 --max-tokens 2000000
     python runpod/train_sae.py         gemma-4-e4b --layer 21 --features 16384 --k 32 --steps 20000
NOTE
