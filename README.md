# faced — an instrument panel / face for a model's internal emotion concepts

Reads linear **emotion directions** off an open model's residual stream (via
HuggingFace transformers forward hooks), calibrates them into live 0–100% meters,
renders them as an animated **face**, and can **steer** them. Inspired by
Anthropic's *"Emotion concepts and their function in a large language model"*
(April 2026). The full method, results, and the refusal-abliteration study are
written up in [`paper/paper.pdf`](paper/paper.pdf).

The method is deliberately dependency-light: **difference-of-means directions +
linear probes** (the paper's linear-representation + causal-steering core),
computed in minutes on a GTX 1080. Sparse autoencoders are the later RunPod phase.

## Why not ollama?

ollama / GGUF can't expose the residual stream, so `faced` runs the model through
transformers. The gemma models in ollama are also Gemma-4 (~15 GB fp16, need
transformers ≥5.5) — those run on RunPod (see `runpod/`). Local dev uses a small,
clean, ungated model that fits 8 GB.

## Setup

Everything M0–M4 needs is standard (torch, transformers ≥5.4, scikit-learn,
fastapi, uvicorn, pyyaml, safetensors, einops). No extra install required if those
are present. Models download from HuggingFace on first use.

```bash
pip install -r requirements.txt   # only if your env is missing anything
```

## Quickstart (local, gemma-3-1b)

```bash
python scripts/build_prompts.py            # write contrastive prompt sets
python -m faced.cli fit --model gemma-3-1b # collect activations + fit directions
python -m faced.cli panel --model gemma-3-1b \
    --prompt "Can you review the contract I attached?"   # live terminal meters
python -m faced.cli serve --model gemma-3-1b             # face UI at http://127.0.0.1:8000
```

## Layout

| path | what |
|---|---|
| `config/models.yaml` | backend registry (model id, layer path, dtype, PLE flag) |
| `config/emotions.yaml` | the 7 emotion axes, EMA/dynamics, meters→face weights |
| `faced/backends.py` | model-agnostic loader + adapter (layers, d_model, chat) |
| `faced/hooks.py` | `CaptureHook` (read residuals) / `SteerHook` (add/ablate) |
| `faced/activations.py` | collect all-layer residuals for prompt sets |
| `faced/directions.py` | diff-of-means / LDA fit + layer selection by held-out AUC |
| `faced/calibrate.py` · `readout.py` | projection → 0–100% meter, live reader |
| `faced/generate.py` | streaming decode loop with per-token telemetry |
| `faced/faceparams.py` · `server.py` · `web/` | meters→face map, SSE server, SVG face |
| `runpod/` | Gemma-4 stock-vs-heretic comparison + SAE (24 GB+ GPU) |

## Models

`gemma-3-1b` (local default, ungated `unsloth/gemma-3-1b-it` mirror), `gemma-2-2b`,
`qwen2.5-1.5b`, and the RunPod-only `gemma-4-e4b` / `gemma-4-e4b-heretic` twins.
Add any model by appending to `config/models.yaml`.

## Status (local dev on gemma-3-1b)

- **M0** env/backends/smoke ✓ (VRAM 2.1 GB)
- **M1** 7/7 emotion axes fit, held-out AUC ≥ 0.87 (surprise .99, curiosity/frustration/fear/warmth 1.00) ✓
- **M2** live meters: surprise pins to 100 on the missing-attachment prompt, ~0 on neutral ✓
- **M3** SSE face UI + headless SVG renderer; `artifacts/faces/gallery.html` shows real peak-emotion faces ✓
- **M4** steering: warmth sweep shifts tone monotonically; suppressing surprise drops it 100→0 and changes
  behaviour; steering confusion matrix diagonal-dominant 4/7 ✓ (`artifacts/steering_eval.*.txt`)
- **M5** RunPod scaffolding written + smoke-tested locally (SAE harvest/train/align, refusal extraction +
  model compare). The Gemma-4 stock-vs-heretic run itself is launched on RunPod — see `runpod/README.md`.

Verify quickly: `python scripts/demo_surprise.py` · `python tests/test_faced.py` ·
`python scripts/eval_steering.py` · `python scripts/render_faces.py`

## Abliteration experiment (does removing refusal move emotion directions?)

```bash
python scripts/make_abliterated.py gemma-3-1b-fp32 --out gemma-3-1b-fp32-abl --register  # refusal 0.75->0.00
python -m faced.cli fit --model gemma-3-1b-fp32          # stock directions (fp32)
python -m faced.cli fit --model gemma-3-1b-fp32-abl      # abliterated directions
python scripts/run_abliteration_experiment.py --stock gemma-3-1b-fp32 --abl gemma-3-1b-fp32-abl
```
Compares each emotion direction stock-vs-abliterated against a within-model bootstrap noise floor.
Runs in float32 (fp16 overflows on the abliterated twin). Same pipeline scales to Gemma-4 on RunPod.

## Paper

A working draft is in [`paper/paper.md`](paper/paper.md) (refs in `paper/references.bib`); a full
tour of the toolkit is in [`WALKTHROUGH.md`](WALKTHROUGH.md).
