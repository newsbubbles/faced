# faced — complete walkthrough

An instrument panel / face for a language model's internal **emotion concepts**. It reads
linear emotion directions out of an open model's residual stream *live, per token*, shows
them as meters and an animated face, and can **steer** them. This document is the full tour:
the idea, the architecture, every module, the data flow, how to run each piece, the design
decisions that make it work, and where it goes next.

---

## 1. The idea

Anthropic's "emotion concepts" paper showed that inside an LLM, concepts like *surprise*,
*desperation*, and *warmth* exist as **linear directions in the residual stream** — and that
they're **causal**: ablate or amplify a direction and behaviour changes. Humans expose their
internal state through a face; a model has none. `faced` builds the missing instrument.

We reproduce the paper's two load-bearing claims at small scale:

- **Linearity** — a single direction per emotion separates "feeling it" from "not"
  (held-out AUC 0.87–1.00 on gemma-3-1b).
- **Causality** — adding/subtracting that direction changes the model's tone and behaviour.

The core deliberately avoids sparse autoencoders (the paper's *discovery* method — expensive,
unlabeled) and uses **difference-of-means directions**, which are named by construction and fit
in minutes on a single GTX 1080. SAEs return as the RunPod research phase.

Full write-up: [`paper/paper.pdf`](paper/paper.pdf).

---

## 2. Architecture at a glance

```
 CONTRASTIVE PROMPTS            OFFLINE FIT (once)                  LIVE (per token)
 ───────────────────           ──────────────────                 ────────────────
 data/prompts/*.jsonl          activations.py                     generate.py
 surprise: pos vs neutral  ──►  run each prompt through   ──┐      manual decode loop
 confidence: conf vs unc        the model, capture the      │      + KV cache
 ... 7 axes                     residual stream at ALL       │           │
                                layers (last-tok, mean-gen)  │      CaptureHook reads
                                        │                    │      residual @ each
                                        ▼                    │      emotion's layer
                                directions.py                │           │
                                per layer: μ⁺−μ⁻ (diff-of-   │      readout.py
                                means) + shrinkage-LDA;      │      project → calibrate
                                pick best layer by held-out  │      → 0-100% meter (EMA)
                                AUC (group split by family)  │           │
                                        │                    │      ┌────┴─────┐
                                        ▼                    │      ▼          ▼
                                directions.<model>.safeten.  │   faceparams  server.py
                                calibration.<model>.json ◄───┘   meters→FACS   SSE stream
                                (fit_reference.py adds            face params      │
                                 baseline + noise floor)               │          ▼
                                        ▲                              ▼      web/ SVG face
                                        │                         hooks.SteerHook  + meters
                                        └── steering (M4): add ±α·v_steer / ablate
```

Everything is **model-agnostic**: a backend registry (`config/models.yaml`) says how to load
any model and where its decoder layers live. Swap gemma-3-1b → Qwen → Gemma-4 by adding an entry.

---

## 3. Repo layout & module-by-module

### The package — `faced/`

| module | what it does | key symbols |
|---|---|---|
| `backends.py` | Loads any registered model via HF transformers; resolves the decoder-layer `ModuleList`; uniform chat/forward surface | `load()`, `ModelBundle`, `registry()` |
| `hooks.py` | The two primitives everything hinges on | `CaptureHook`, `SteerHook` (add / add_rms / ablate) |
| `activations.py` | Runs prompts through the model, captures all-layer residuals (last-prompt + mean-over-generated) | `capture_prompt()`, `collect_emotion()` |
| `directions.py` | Fits directions & selects the layer | `_diff_of_means()`, `_lda_dir()`, `_cv_auc()`, `fit_all()` |
| `calibrate.py` | Raw projection → 0–100% meter (anchored map; bipolar recentring; noise floor) | `Calibrator.meter()` |
| `readout.py` | Loads directions+calibration; projects a residual → calibrated, EMA-smoothed meters | `EmotionReader.read()`, `.read_captured()` |
| `generate.py` | The manual streaming decode loop unifying capture + steer + telemetry | `stream()` |
| `faceparams.py` | Linear map: emotion meters → shared FACS-lite face params | `FaceMapper.to_params()` |
| `faceviz.py` | Headless renderer: face params → standalone SVG (no browser) | `render_svg()` |
| `server.py` | FastAPI: `/api/meta`, `/api/stream` (SSE per-token meters+face) | `build_app()`, `run()` |
| `cli.py` | `panel` / `fit` / `serve`; the ANSI terminal panel | `TerminalPanel` |
| `compare.py` | M5: refusal-direction extraction + cross-model emotion comparison | `extract_and_save_refusal()`, `compare()` |
| `config.py` | Loads `emotions.yaml` | `emotions_config()`, `axis_names()` |

### Config — `config/`
- `models.yaml` — backend registry: `hf_id`, `arch`, `layer_path`, `dtype`, `ple`.
- `emotions.yaml` — the 7 axes, EMA speeds, `min_auc`, and the meters→face weight matrix.
- `calibration.<model>.json` — *generated*: per-axis layer, kind, AUC, projection anchors, baseline.

### Data — `data/`
- `prompts/*.jsonl` — 7 contrastive sets (positive vs neutral/opposite, 4 styles, tagged by template family).
- `reference_corpus.jsonl` — neutral prompts for baseline calibration.
- `refusal/{harmful,harmless}.jsonl` — for the abliteration experiment.
- `activations/<model>/*.safetensors` — *generated* captured residuals.

### Scripts — `scripts/`
`build_prompts.py` · `fit_all.py` · `fit_reference.py` · `demo_surprise.py` · `eval_steering.py` ·
`plot_emotions.py` · `steer_showcase.py` · `render_faces.py` · `serve.py` · `smoke_m0.py`

### Web — `web/`
`index.html` (layout + SVG skeleton), `face.js` (eases FACS params → SVG geometry), `meters.js`.

### RunPod — `runpod/`
`setup.sh` · `run_comparison.py` (M5a) · `harvest_activations.py` + `train_sae.py` (M5b) · `README.md`.

---

## 4. The data flow in detail

**a) Build contrastive prompts.** For each emotion, positives (evoking it) and negative controls
across four styles — *situation* ("review the contract I attached" with no attachment),
*first-person*, *third-person*, *matched-neutral* — each tagged with a **template family** so
variants can't leak across the train/test split.

**b) Collect activations.** `capture_prompt()` chat-templates each prompt, greedily generates
~16 tokens, and records the post-layer residual stream at **every layer** for the last prompt
token and the mean over generated tokens (`mean_gen` is primary — it matches how we read live).

**c) Fit directions & pick the layer.** Per emotion, per layer: `v_steer = μ_pos − μ_neg` (raw
diff-of-means — interpretable, steerable) and a shrinkage-LDA readout. Score each layer by
**cross-validated held-out AUC** with a *group split by template family*; keep the best
`(layer, kind)`. → `directions.<model>.safetensors` + `calibration.<model>.json`.

**d) Calibrate.** The meter is an anchored linear map: negative-class projection → 0%,
positive-class → 100%. `fit_reference.py` runs a neutral corpus to set each axis's baseline (so
bipolar *confidence* rests at 50 instead of pinning) plus a small noise floor.

**e) Read live.** `EmotionReader` + `generate.py`'s `stream()`: a KV-cached decode loop where a
`CaptureHook` reads the residual at each emotion's layer, projects onto its readout direction,
calibrates, EMA-smooths (fast for phasic *surprise*, slow for tonic moods), and yields per-token
`{text, meters}`.

**f) Show it.** `FaceMapper` maps the 7 signed meters through a weight matrix into ~13 shared
facial channels so simultaneous emotions **blend**. `server.py` streams `{text, meters, face}`
over SSE; `web/` renders the bars and the SVG face.

**g) Steer.** `SteerHook` adds `α·v_steer` (raw or RMS-scaled) or ablates the direction.

---

## 5. How to run it

```bash
# one-time fit (~20 min collection on the 1080, then seconds to fit)
python scripts/build_prompts.py
python -m faced.cli fit --model gemma-3-1b      # collect activations + fit directions
python scripts/fit_reference.py                 # baseline/noise-floor calibration

# see it work
python scripts/demo_surprise.py                 # meters spike on the missing attachment
python scripts/plot_emotions.py                 # all emotions over generation time -> PNG
python -m faced.cli panel --prompt "..."        # live ANSI meter panel in the terminal
python -m faced.cli serve                        # face UI at http://127.0.0.1:8000
python scripts/render_faces.py                   # artifacts/faces/gallery.html

# causal experiments
python scripts/steer_showcase.py                 # one control input, steered per emotion
python scripts/eval_steering.py                  # warmth sweep, suppress surprise, confusion matrix
python tests/test_faced.py                        # unit tests
```

### Illustrative outputs
- `artifacts/emotions_over_time.png` — all 7 meters as line plots vs token index, for three
  prompts (surprise / frustration / warmth). You can watch the internal state move as it generates.
- `artifacts/steer_showcase.<model>.txt` — the same neutral prompt, generated once per emotion
  with that emotion's direction injected, so you can read how each internal state colours the text.
- `artifacts/faces/gallery.html` — the real model's peak-emotion faces.
- `artifacts/steering_eval.<model>.txt` — warmth α-sweep, surprise suppression, steering matrix.

---

## 6. Design decisions & gotchas

- **ollama can't expose the residual stream** → load weights in HF transformers and hook decoder
  layers directly. ollama stays a possible *behavioural* side-channel only.
- **Two vectors per axis**: raw diff-of-means (interpretable, what you *steer*) and shrinkage-LDA
  (max separability, what you *read*). SAE features live in raw activation space → align with
  diff-of-means, not LDA.
- **Group-split-by-family AUC** stops a template's variants inflating the score across the split.
- **Asymmetric calibration**: "ablating to 0" isn't neutral for a unipolar axis (surprise's
  baseline projection is ≈ −260), so suppression = steering *toward the negative pole*, not zeroing.
- **transformers-5.x subtlety**: `output_hidden_states` is snapshotted *before* forward-hook edits,
  so a steered residual wouldn't show in the meters. Fix: read via a `CaptureHook` registered
  **after** the steer hook (done in `generate.py`).
- **rAF doesn't fire in the preview pane** → the SVG face animates via `setInterval` + an immediate
  update per token.
- **gemma fp16 overflow** in late layers on some prompts → nan-safe means in refusal extraction.
- **Model-agnostic**: adding a model is one YAML entry; the backend auto-detects the decoder
  `ModuleList` if the path is wrong.

---

## 7. Results so far (gemma-3-1b, 2.1 GB VRAM)

- **7/7 emotion axes** fit with held-out AUC 0.87–1.00.
- **Surprise** pins to 100 the moment the model registers the missing attachment; ~0 on neutral.
- **Warmth steering** shifts tone monotonically cold→warm (coherent to ~+2, breaks ~+5 — expected
  for a 1B model).
- **Suppressing surprise** drops the meter 100 → 0 and changes the response.
- **Steering confusion matrix** diagonal-dominant for 4/7 axes (steering is mostly specific).

---

## 8. RunPod phase (M5) — the abliteration experiment

The Gemma-4-E4B twins are too big for the 1080 (~15 GB bf16, transformers ≥5.5), so M5 runs on a
24 GB+ pod. See [`runpod/README.md`](runpod/README.md).

- **M5a — `run_comparison.py`**: fit the same 7 axes on `google/gemma-4-E4B-it` (stock) vs
  `igorls/gemma-4-E4B-it-heretic` (abliterated), then measure per-axis **cosine(A,B)** (did
  abliteration rotate the emotion?), **AUC delta**, and the **cosine overlap between the refusal
  direction and each emotion**. This quantifies how *removing the refusal subspace* perturbs the
  emotion landscape. (Local smoke test hints: refusal overlaps confusion/frustration most,
  warmth least.)
- **M5b — `harvest_activations.py` + `train_sae.py`**: train an SAE and check whether the cheap
  diff-of-means directions correspond to features an unsupervised SAE discovers, and surface axes
  we didn't name. Both are model-agnostic and were smoke-tested locally on gemma-3-1b.

---

## 9. Extending it

- **Add a model**: append to `config/models.yaml` (id, arch, layer_path, dtype). Re-run `fit`.
- **Add an emotion axis**: add seeds to `build_prompts.py`, an entry + face-weight row in
  `emotions.yaml`, re-fit.
- **New face**: `faceparams` weights and `web/face.js` / `faceviz.py` geometry are decoupled from
  the model.
