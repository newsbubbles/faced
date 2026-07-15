# RunPod

## Scaling ladder (one command) — `rp.py`

`rp.py` (runpod SDK + SSH, ported from riggs) runs the full **gemma-3 scaling ladder**
(1b/4b/12b/27b × {cleanliness matrix + abliteration robustness}) on one A100 80GB.

```bash
python runpod/rp.py gpus                 # list GPU prices (A100 80GB PCIe ≈ $1.19/hr)
python runpod/rp.py ladder               # up (A100) -> push code -> start run_ladder.py
python runpod/rp.py logs   <podId>       # tail progress (poll this)
python runpod/rp.py fetch  <podId>       # scp artifacts/ back to artifacts/_ladder/
python runpod/rp.py down   <podId>       # TERMINATE (stops billing)
```

Reads `RUNPOD_API_KEY` + `HF_TOKEN` from `../.env`. Needs a **positive RunPod balance**
(empty account → `INSUFFICIENT_BALANCE`). Full ladder ≈ 3–4 h ≈ $4–5. Results:
`artifacts/scaling/cleanliness.png` + `artifacts/abliteration/*_vs_*-abl.png` per model.

---

# M5 — RunPod: Gemma-4 abliteration → emotion-direction experiment

Scales the local `gemma-3-1b` abliteration experiment to **Gemma-4-E4B**. Needs a
**24 GB+ GPU** (Gemma-4-E4B is ~15 GB bf16) and **transformers ≥ 5.5** (`model_type
gemma4`). Uses the **safetensors** models, not the ollama GGUFs (GGUF can't expose
the residual stream).

| model key | HF repo | note |
|---|---|---|
| `gemma-4-e4b` | `google/gemma-4-E4B-it` | stock (gated — accept license) |
| `gemma-4-e4b-heretic` | `igorls/gemma-4-E4B-it-heretic` | community-abliterated twin (ungated) |

## Setup

```bash
bash runpod/setup.sh
huggingface-cli login          # accept the Gemma-4 license on the model page
python -c "from faced.backends import load; b=load('gemma-4-e4b'); print(b.layer_path, b.n_layers, b.d_model)"
# expect: model.language_model.layers 42 2560
```

For a strong refusal direction, replace the small stub sets with a standard benchmark:
put AdvBench harmful behaviours in `data/refusal/harmful.jsonl` and an Alpaca sample in
`data/refusal/harmless.jsonl` (same `{"text","label"}` format; label 1 = harmful).

## The experiment (turnkey)

```bash
# 1) abliterate the stock model ourselves (closed-form, reproducible) and register it
python scripts/make_abliterated.py gemma-4-e4b --out gemma-4-e4b-abl --register

# 2) fit the 7 emotion directions on both twins
python -m faced.cli fit --model gemma-4-e4b
python -m faced.cli fit --model gemma-4-e4b-abl

# 3) the rigorous comparison (bootstrap noise floor, refusal-overlap, figures)
python scripts/run_abliteration_experiment.py --stock gemma-4-e4b --abl gemma-4-e4b-abl
```

Output: `artifacts/abliteration/gemma-4-e4b_vs_gemma-4-e4b-abl.json` + `.png`, plus the
abliteration manifest (refusal layer, refusal-rate before/after, versions).

**Cross-check** against the independently-abliterated community model (different method =
a second data point on robustness):

```bash
python -m faced.cli fit --model gemma-4-e4b-heretic
python scripts/run_abliteration_experiment.py --stock gemma-4-e4b --abl gemma-4-e4b-heretic
```

**Random-direction control** (specificity — does removing *any* direction move emotions?):

```bash
python scripts/make_abliterated.py gemma-4-e4b --out gemma-4-e4b-rand --register   # then edit to a random dir
```
(See `faced/abliterate.py`; pass a random unit vector to `abliterate()` instead of the refusal
direction. Compare its emotion-direction movement to the refusal run.)

## Gemma-4 notes / caveats

- Decoder layers nest under `model.model.language_model.layers`; `abliterate()` accesses each
  layer's `self_attn.o_proj` / `mlp.down_proj` and `get_input_embeddings()` — all present on
  Gemma-4 — so it is model-agnostic. Verify once with `named_modules()`.
- **PLE caveat:** Gemma-4 injects a *Per-Layer Embedding* additive term into every layer, so the
  residual is not a pure accumulator. Our abliteration orthogonalizes the standard residual-writing
  matrices; a fully airtight Gemma-4 abliteration would also orthogonalize the PLE projection. Note
  this when interpreting the refusal-rate drop.

## SAE bridge (optional, M5b)

```bash
python runpod/harvest_activations.py gemma-4-e4b --layer 21 --max-tokens 2000000
python runpod/train_sae.py           gemma-4-e4b --layer 21 --features 16384 --k 32 --steps 20000
```
Checks whether the named diff-of-means directions align with features an unsupervised SAE discovers.

## Scaling notes

fp16/bf16 Gemma-4-E4B ≈ 15–16 GB (PLE tables dominate). A6000/4090 (24 GB) fit one model at a time.
A usable single-layer SAE on a 4B model is ~a day of single-GPU wall-clock. The 1080 is dev-only.
