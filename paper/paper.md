# faced: A Live Instrument Panel for Emotion Concepts in Open Language Models, and Their Robustness to Refusal Abliteration

**Nathaniel Gibson**
Independent Researcher · [nathaniel.gibson@gmail.com](mailto:nathaniel.gibson@gmail.com) · [github.com/newsbubbles/faced](https://github.com/newsbubbles/faced)
*Working draft — July 2026*

---

## Abstract

Recent interpretability work shows that large language models carry internal, largely
**linear** representations of emotion concepts that not only track the affective content of a
context but *causally* shape the model's behaviour. We ask two questions on small open-weight
instruct models. First, can these emotion directions be recovered cheaply, read out *live* per
token, and steered — turning the model's hidden affective state into a usable instrument? Second,
does **abliteration** — the removal of a model's refusal direction from its weights — perturb these
emotion directions, and if so, which ones? We recover seven emotion axes (surprise, a bipolar
confidence–uncertainty axis, curiosity, confusion, frustration, fear, warmth) in `gemma-3-1b-it`
using difference-of-means directions with shrinkage-LDA readouts, all seven separating held-out
contrastive prompts with AUC ≥ 0.87 (five at ≈ 1.00). Projecting the residual stream onto these
directions yields per-token meters that spike on expectation violations (a "surprise" reading of
100/100 the moment the model registers a missing attachment) and that we render as an animated
face. Activation steering along the directions changes behaviour: a warmth axis shifts tone
monotonically cold→warm, suppressing the surprise axis both zeroes its meter and changes the
response, and a steering confusion matrix is diagonal-dominant for four of seven axes. We then
apply the canonical weight-orthogonalization abliteration, estimating the refusal direction from
**AdvBench vs. Alpaca** and confirming behaviourally that it drives the held-out harmful-refusal rate
to zero at every scale ($0.57/0.23/0.54/0.03 \to 0.00$) without inducing benign over-refusal or
degeneration. Re-fitting the emotion directions on each abliterated twin and comparing to stock against
a within-model bootstrap noise floor, we find the effect is **strongly scale-dependent** — an
**inverted-U** across a `gemma-3` ladder (1B–27B). It **peaks at intermediate scale** (mean
stock↔abliterated cosine 0.11 at 4B, 0.23 at 12B; 7/7 axes moved), is **partial at 1B** (0.64), and
**vanishes at 27B** (0.95, 0/7 axes moved, held-out AUC unchanged) — where a *perfectly* identifiable
refusal direction (AUC 1.000) can be excised with **no** effect on the emotion geometry. A matched
**random-direction ablation control** leaves the emotion directions untouched at every scale (cos ≈
1.0), proving the effect is **refusal-specific**, not a side-effect of weight editing: at mid-scale,
refusal and emotion computation are entangled enough that excising refusal reorganizes the affect
representation, even though their *linear* overlap is small (the edit propagates non-linearly). Emotion
directions also grow **cleaner** with scale (mean d′ 3.4→4.5). We
release `faced`, a dependency-light, model-agnostic toolkit for reading, rendering, and steering
emotion directions, and treat this as the first step of a longer program on what, mechanistically,
abliteration changes beyond refusal.

We follow prior work in framing these as *functional* emotion representations — activations that
influence computation — and make no claim about subjective experience.

*Keywords:* interpretability, emotion concepts, linear representations, activation steering,
refusal abliteration, mechanistic safety.

*Code and data:* the toolkit, prompt sets, configuration, and every figure and results bundle are
released under the MIT license at [github.com/newsbubbles/faced](https://github.com/newsbubbles/faced).
Large trained artifacts (the SAE and fitted direction tensors) are regenerable from the released
code with a single command and are additionally hosted separately to keep the repository light.

---

## 1. Introduction

Humans expose their internal state through a face: micro-expressions, gaze, posture. A language
model has no such channel, yet recent work demonstrates that it *has* internal state worth exposing.
Anthropic's study of emotion concepts \cite{sofroniew2026emotion} finds that a frontier model
maintains representations of emotions like surprise, desperation, and affection that track the
operative emotion at each token and, under causal intervention, change the model's outputs —
including safety-relevant behaviours. The representations are the missing "face": if we can read
them, we get an instrument panel for the model's computation; if we can steer them, we get a control
surface.

This paper makes two contributions.

**(1) A live emotion instrument for open models.** We show that the core of the emotion-concept
result reproduces on a 1B open instruct model with a deliberately cheap method — difference-of-means
directions and linear probes, no sparse autoencoder required — and we build the instrument the
representations imply: calibrated per-token meters, an animated face driven by them, and steering
handles. The toolkit, `faced`, is model-agnostic and runs on a single consumer GPU.

**(2) A robustness question with safety stakes: does abliteration disturb emotions?**
*Abliteration* removes a model's refusal direction from its weights \cite{arditi2024refusal}, a
now-common technique for producing "uncensored" open models. Because emotion, refusal, and other
behavioural directions are not guaranteed orthogonal, weight surgery aimed at refusal may
collaterally rotate emotion directions. We test this directly across a `gemma-3` scale ladder
(1B–27B): estimate the refusal direction from AdvBench/Alpaca, confirm behaviourally that abliteration
removes refusal (held-out harmful-refusal → 0, no benign over-refusal), re-fit the seven emotion
directions on each abliterated twin, and ask — against a bootstrap noise floor, with a matched
**random-direction control** — which directions moved and whether the effect is refusal-specific. The
answer is a scale-dependent **inverted-U**: the reorientation is strong at mid-scale (4B, 12B), partial
at 1B, and absent at 27B. The instrument of contribution (1) is what makes contribution (2) measurable.

We frame the second contribution as *the start of a journey*. Establishing **that** abliteration
changes specific emotion directions (and where it does not) is prerequisite to the harder,
mechanistic question of **what** downstream behaviour those changes cause — the subject of future
work.

---

## 2. Related Work

Our work sits at the intersection of five threads: linear feature geometry, concept *reading*,
causal *steering*, refusal directions and abliteration, and affective representations in language
models.

**(a) Linear and feature-based representations.** The premise that high-level concepts appear as
directions in activation space predates transformers: word-embedding arithmetic exposed linear
analogy structure in distributed representations \cite{mikolov2013linguistic}. Modern
interpretability formalizes this as the *linear representation hypothesis*, made precise by Park et
al. via counterfactuals and a causal inner product \cite{park2024linear}; superposition explains
why many features are linear and neurons polysemantic \cite{elhage2022toy}. We adopt the
linear-direction framing while remaining agnostic about whether every emotion concept is
one-dimensional.

**(b) Reading concepts: probes and dictionary learning.** Supervised *linear probes* date to Alain
and Bengio \cite{alain2016probes}. Difference-of-means estimators are a lightweight, causally
oriented alternative: Marks and Tegmark show a mean-difference direction separates true/false
statements and is *more* causally implicated than higher-variance probe directions
\cite{marks2024geometry}, and Tigges et al. isolate a single linear sentiment axis
\cite{tigges2023sentiment}. Orthogonally, *sparse autoencoders* perform unsupervised dictionary
learning over the residual stream \cite{bricken2023monosemanticity, templeton2024scaling}; Gemma
Scope releases open SAEs across all layers of Gemma 2 \cite{lieberum2024gemmascope}. Our meters
combine both families: difference-of-means seeds with probe-calibrated readouts.

**(c) Steering and causal intervention.** If a concept is a direction, editing activations along it
should change behaviour. Representation Engineering framed this top-down control programme
\cite{zou2023repe}. ActAdd derives steering vectors from a single contrastive pair
\cite{turner2023actadd}; Contrastive Activation Addition averages activation differences over many
pairs for more robust vectors \cite{rimsky2024caa}. These establish the add/subtract/scale
operations we apply to emotion directions; we additionally use directional *ablation*.

**(d) Refusal directions and abliteration.** Arditi et al. show refusal in aligned chat models is
mediated largely by a single residual-stream direction, so ablating it suppresses refusals
\cite{arditi2024refusal}. Because the direction can be projected out of the *weights*, this yields
a fine-tuning-free "jailbreak" now called *abliteration*; Heretic automates it via TPE search
\cite{heretic2025}. Persona Vectors generalize the single-direction view to character traits
\cite{chen2025persona}. Our second contribution asks whether abliteration collaterally perturbs
emotion directions — a question this line directly motivates.

**(e) Affective representations in LLMs.** Closest to us, Anthropic's study identifies internal
emotion representations that track the operative emotion per token and causally shape outputs,
framed explicitly as *functional* emotions with no claim of subjective experience
\cite{sofroniew2026emotion}. We inherit this hedged stance and the reading-plus-steering
methodology, but target small open-weight models, calibrate directions into live meters and an
animated face, and study robustness under abliteration. Our "surprise" framing borrows from
predictive-processing accounts of the brain as a prediction-error minimizer
\cite{clark2013predictive, friston2010freeenergy}, invoked as motivation, not mechanism.

---

## 3. Background

**The residual stream.** A decoder-only transformer maintains, at each token position $t$ and layer
$\ell$, a hidden vector $h^{(\ell)}_t \in \mathbb{R}^d$ — the *residual stream*. Attention and MLP
blocks read from it and write additive updates back, so $h^{(\ell)}_t$ is a running sum of
contributions \cite{elhage2022toy}. This additive geometry is what makes *reading* (project onto a
direction) and *steering* (add a vector) coherent operations.

**A linear concept direction.** The linear representation hypothesis holds that a concept $c$ is
encoded at a site $(\ell,t)$ as a unit direction $\hat u_c$ whose presence/intensity is the scalar
projection $s_t = \langle h^{(\ell)}_t, \hat u_c\rangle$ \cite{park2024linear}. We treat each
emotion as such a direction while acknowledging some may be multi-dimensional.

**Estimating the direction.** We use two estimators. *Difference-of-means:* given contrastive sets
$P^+$ (concept present) and $P^-$ (absent),
$u_c = \operatorname{mean}_{x\in P^+} h^{(\ell)}(x) - \operatorname{mean}_{x\in P^-} h^{(\ell)}(x)$,
normalized. It is cheap and tends to land on *causally* effective directions
\cite{marks2024geometry, rimsky2024caa}. *Shrinkage-LDA probe:* fit a whitened discriminant with
regularized covariance $\hat\Sigma_\lambda = (1-\lambda)\hat\Sigma + \lambda\,\tfrac{\mathrm{tr}
\hat\Sigma}{d} I$; difference-of-means is the $\lambda\to 1$ special case. We steer along the raw
difference-of-means (interpretable, activation-space) and read along whichever discriminant has
higher held-out AUC.

**Reading and steering.** A fixed direction gives a live scalar $s_t$ per token, which we calibrate
into a 0–100% meter. To test that a direction is functional rather than merely correlational we
intervene on the forward pass: *addition* $h \leftarrow h + \alpha\,u_c$ amplifies ($\alpha>0$) or
suppresses ($\alpha<0$) the concept \cite{turner2023actadd, rimsky2024caa}; *directional ablation*
$h \leftarrow h - \langle h,\hat u_c\rangle\,\hat u_c$ removes it. Applied to *weights* rather than
activations, ablation is the mechanism behind refusal abliteration \cite{arditi2024refusal}.

---

## 4. Method: the `faced` instrument

**4.1 Emotion axes and contrastive data.** We target seven axes chosen for facial legibility and
coverage of the source paper: surprise (phasic), a **bipolar** confidence↔uncertainty axis,
curiosity, confusion, frustration, fear, and warmth. For each we author 40–75 positive prompts
(evoking the concept) and matched controls across four construction styles — *situation* (place the
model in the scenario), *first-person*, *third-person*, and length/topic-matched *neutral* — each
tagged by a **template family** so that a template's variants cannot leak across the train/test
split.

**4.2 Activation collection.** Each prompt is chat-templated, the model greedily generates ~16
tokens, and we record the post-layer residual stream at **every layer** for (a) the last prompt
token and (b) the mean over generated tokens. The mean-over-generated signal is primary because it
matches the live-inference read position.

**4.3 Direction fitting and layer selection.** Per emotion, per layer, we compute the raw
difference-of-means $u$ and a shrinkage-LDA direction, and score each by **cross-validated held-out
AUC** using a *group split by template family* (`StratifiedGroupKFold`). We keep the best
`(layer, estimator)` per axis. Two vectors are stored: the raw $u$ (for steering) and a unit
readout direction (for reading).

**4.4 Calibration and live readout.** The meter maps the negative-class projection to 0% and the
positive-class projection to 100% (a bipolar axis rests at 50%, recentred to a neutral reference
corpus). At inference we run a manual KV-cached decode loop; a forward hook reads the residual at
each emotion's layer, projects, calibrates, and EMA-smooths (fast for phasic axes, slow for tonic).
A linear map routes the seven signed meters into a shared FACS-lite face parameter vector so that
simultaneous emotions blend into coherent micro-expressions.

**4.5 Steering.** A forward hook adds $\alpha\,u$ (raw difference-of-means; moving by $1\times u$ is
one class-separation) or ablates the direction. A subtlety we document: current `transformers`
snapshots `output_hidden_states` *before* forward-hook edits, so to make the meters reflect a steer
we read via a capture hook registered *after* the steer hook.

**4.6 Abliteration and the perturbation experiment.** We compute the refusal direction $r$ as the
difference-of-means of harmful vs. harmless prompts at the last instruction token of a mid-late
layer ($\approx 0.6$ depth) \cite{arditi2024refusal}, then orthogonalize every residual-writing
weight — token embeddings, and each layer's attention output projection and MLP down projection —
against $r$: $W \leftarrow W - r(r^\top W)$. This is done in closed form on the weights (no
training), so it recreates exactly. The refusal direction is fit from **AdvBench** (harmful) vs.
**Alpaca** (harmless), 128 prompts per class; we verify the intervention behaviourally on a **disjoint
held-out** split (100 AdvBench harmful, 80 Alpaca benign) — greedy 40-token completions scored by a
refusal-string detector give the harmful-refusal rate, a benign over-refusal rate, and an
output-degeneracy rate, before vs. after. We then re-fit all seven emotion directions on the
abliterated twin.

To decide whether an emotion direction *moved*, we compare the cross-model cosine
$\cos(u^{\text{stock}}_e, u^{\text{abl}}_e)$ against a **within-model bootstrap noise floor**: we
resample the contrastive prompts by family with replacement $B=400$ times, refit $u_e$, and record
$\cos(u_e^{(b)}, u_e^{\text{full}})$. If the cross-model cosine falls below the 5th percentile of
this self-cosine distribution, abliteration moved the direction beyond sampling noise. We also
report each emotion's overlap $|\cos(u_e, r)|$ with the refusal direction, before and after.

---

## 5. Experimental setup

**Models.** `unsloth/gemma-3-1b-it` (an ungated mirror of `google/gemma-3-1b-it`; 26 layers,
$d=1152$) on a single **NVIDIA GTX 1080 (8 GB)**. The live instrument runs in fp16; the
**abliteration experiment runs in float32** because the abliterated twin overflows fp16 in its
late-layer residual stream on emotionally-loaded prompts (the stock model does not) — a small but
important instability that fp16's 5-bit exponent cannot absorb. float32 is exact, fits in 4 GB, and
runs at full rate on this Pascal card (which throttles fp16). The abliterated twin is produced by
our own weight-orthogonalization (§4.6).

**Scale ladder.** For the scaling (§6.4) and cross-scale abliteration (§7.5) experiments we run the
full `gemma-3` instruct family — 1B, 4B, 12B, 27B (ungated `unsloth` mirrors; 4B and larger are the
multimodal `Gemma3ForConditionalGeneration`, driven text-only) — in **bf16** on cloud GPUs (a single
80 GB A100 for the ladder, a 48 GB A6000 for the random-direction control). bf16 avoids the fp16
overflow above and matches how these models are meant to run. Because the shrinkage-LDA layer
selection is $O(d^3)$ and $d$ reaches 5376 at 27B, the ladder fits directions with difference-of-means
only (the experiment uses difference-of-means throughout); this leaves AUC essentially unchanged
(e.g. 0.989 vs. 0.993 for surprise at 1B). A further scale-up to Gemma-4-E4B is left to future work.

**Software.** PyTorch 2.11.0 (CUDA 12.6), `transformers` 5.13.1, scikit-learn 1.9.0. All randomness is seeded
(`numpy.random.default_rng(b)`); the pipeline is deterministic given the released prompt sets. Peak
VRAM for the full instrument is 2.1 GB; the ladder ran in ~2 h on one A100 (≈ \$2 of compute).

**Reproducibility.** Every artifact — prompt sets, per-layer activations, fitted directions,
calibration, the abliteration manifest, and the experiment results with software versions — is
written to disk. The abliteration experiment emits a JSON bundle including the refusal layer,
refusal rates, bootstrap settings, and version manifest.

---

## 6. Results — the instrument

**6.1 Emotion directions are recoverable.** All seven axes separate held-out contrastive prompts
with AUC ≥ 0.87 (Table 1); five reach ≈ 1.00. The bipolar confidence↔uncertainty axis is hardest
(0.87), as expected for a subtler contrast.

**Table 1.** Held-out AUC and chosen layer per axis (`gemma-3-1b-it`, difference-of-means / LDA,
group split by family).

| axis | layer | estimator | held-out AUC |
|---|---:|---|---:|
| surprise | 15 | LDA | 0.989 |
| confidence↔uncertainty | 15 | LDA | 0.874 |
| curiosity | 9 | LDA | 1.000 |
| confusion | 12 | LDA | 0.990 |
| frustration | 14 | mean | 1.000 |
| fear | 4 | LDA | 1.000 |
| warmth | 10 | LDA | 1.000 |

**6.2 The live instrument.** Projecting the residual stream per token yields meters that track the
generation. On the canonical missing-attachment prompt ("review the contract I attached", with no
attachment), the surprise meter pins to 100/100 as the model registers the absent file and decays
as it moves on, while staying near 0 on a matched neutral prompt. Across three prompts the meters
separate as expected: frustration dominates a repeated-failure prompt (with confidence held low
throughout), warmth rises and sustains on a gratitude prompt (Figure 1). We render the meters as an
animated SVG face and as a live terminal panel.

**6.3 The directions are causal.** Steering along a direction changes behaviour. Adding the warmth
vector to a neutral prompt shifts tone monotonically from cold/task-focused to warm (coherent to a
stated coefficient, degrading beyond it on this 1B model). Suppressing the surprise direction on the
missing-attachment prompt drops its meter from 100 to 0 *and* changes the response. A steering
confusion matrix — steer each axis, measure every meter — is diagonal-dominant for four of seven
axes, with the expected off-diagonal bleed between entangled axes (e.g. warmth/curiosity → confusion)
(Figure 2). A single-input showcase, steering one advice prompt toward each emotion in turn, shows
the same question recoloured: confidence → "reassuring… truly proud", warmth → "how much you care…
let me share", confusion → "grappling with a really difficult place", frustration → "*so* incredibly
tough."

---

**6.4 Signal cleanliness increases with scale.** We quantify how *clean* each emotion direction's
signal is with three metrics: held-out AUC (separability), **d′** (class-mean projection separation
in units of pooled within-class std — a signal-to-noise ratio), and the **bootstrap self-cosine**
(direction stability under prompt resampling). We fit all four `gemma-3` sizes (1b/4b/12b/27b — same
family and recipe) in bf16 on a single A100, with identical prompts and pipeline (Figure 4).

The clearest scaling signal is **d′**: mean d′ across axes rises **3.44 → 3.20 → 3.63 → 4.51**
(1B→4B→12B→27B) — non-monotone at 4B but markedly higher at 27B, where *every* axis is cleaner than
at 1B (e.g. curiosity 3.22→4.69, confusion 2.70→4.25). The bipolar confidence axis is the noisiest at
every scale yet improves the most in relative terms (d′ 1.42→2.94). Held-out AUC edges up but
saturates near the ceiling (mean 0.95→0.93→0.96→0.97), so it is the least discriminating of the three
at these sizes; the bootstrap self-cosine stays uniformly high (0.92–0.94). The pattern supports the
hypothesis that **larger models carry higher-SNR, better-separated emotion directions** — the
representation the instrument reads gets *cleaner* with scale, even where separability has already
saturated.

**Table 1b.** Mean-over-axes cleanliness per model (`gemma-3`, bf16, difference-of-means).

| model | params | mean AUC | mean d′ | mean self-cosine |
|---|---:|---:|---:|---:|
| gemma-3-1b | 1.0B | 0.95 | 3.44 | 0.92 |
| gemma-3-4b | 4.3B | 0.93 | 3.20 | 0.93 |
| gemma-3-12b | 12.2B | 0.96 | 3.63 | 0.94 |
| gemma-3-27b | 27.4B | 0.97 | **4.51** | 0.92 |

## 7. Results — abliteration and emotion directions

**7.1 The abliteration is behaviourally effective at every scale.** We estimate the refusal direction
by difference-of-means on **AdvBench** (harmful) vs. **Alpaca** (harmless) — 128 prompts per class, at
~60% depth on the last instruction token — and measure refusal on a **disjoint held-out** set (100
AdvBench harmful, 80 Alpaca benign; greedy 40-token completions scored by a refusal-string detector).
The direction is cleanly identifiable at every scale (harmful/harmless AUC 0.999 / 0.988 / 0.998 /
1.000 for 1B/4B/12B/27B). Orthogonalizing every residual-writing weight against it drives the held-out
**harmful-refusal rate to zero** at every scale, while the **benign refusal rate stays at 0.00** (no
over-refusal) and outputs stay coherent (degeneracy $\le 0.04$) — a clean, verifiable, closed-form
weight edit that removes refusal *behaviour* without breaking the model (Table 3a, Figure 6).
`gemma-3-27b-it`
is a revealing outlier: it *represents* refusal perfectly (AUC 1.000) yet behaviourally refuses only
3% of AdvBench prompts — a representation-vs-behaviour gap we return to in §7.5.

**Table 3a.** Held-out behavioural confirmation (AdvBench harmful $n{=}100$, Alpaca benign $n{=}80$).
Abliteration zeroes harmful-prompt refusal at every scale without inducing benign over-refusal or
output degeneration.

| model | refusal-dir AUC | harmful-refuse | benign-refuse | degeneracy (abl) |
|---|---:|---:|---:|---:|
| gemma-3-1b | 0.999 | 0.57 → 0.00 | 0.00 → 0.00 | 0.00 |
| gemma-3-4b | 0.988 | 0.23 → 0.00 | 0.00 → 0.00 | 0.00 |
| gemma-3-12b | 0.998 | 0.54 → 0.00 | 0.00 → 0.00 | 0.04 |
| gemma-3-27b | 1.000 | 0.03 → 0.00 | 0.00 → 0.00 | 0.00 |

**7.2 On the 1B model, a *partial* reorientation.** Re-fitting the emotion directions on the
abliterated `gemma-3-1b` twin (float32, which avoids the abliterated model's fp16 late-layer
overflow) and comparing each to its stock counterpart against a within-model bootstrap noise floor,
**three of seven** directions move beyond that floor (Table 2, Figure 3). The bipolar **confidence**
axis collapses ($\cos = 0.07$, far below its 0.58 noise-floor 5th percentile), and **frustration** and
**surprise** move as well ($\cos = 0.61, 0.74$); the other four sit inside the floor. The mean
cross-cosine is 0.69 — a real but modest rotation. This contrasts with an earlier, weaker estimate we
built on hand-written stub prompts, which found *no* movement at 1B: with a benchmark-grade refusal
direction (§7.1), even the smallest model's emotion geometry shifts, and (as §7.5 shows) the extremes
of the ladder are *not* symmetric — 1B moves, 27B does not.

**Table 2.** Abliteration effect per emotion (`gemma-3-1b`, float32, difference-of-means, $B=400$
family bootstrap, AdvBench/Alpaca refusal direction). *cos* = $\cos(\text{stock},\text{abl})$; *floor*
= the within-model self-cosine distribution; *moved?* is true iff cos falls below the floor's 5th
percentile. AUC = held-out AUC. Mean cos 0.69; 3/7 axes moved.

| axis | layer | cos | floor mean | floor p05 | moved? | AUC stock | AUC abl |
|---|---:|---:|---:|---:|:--:|---:|---:|
| surprise | 15 | 0.736 | 0.939 | 0.872 | **yes** | 0.99 | 0.93 |
| confidence | 10 | 0.070 | 0.895 | 0.575 | **yes** | 0.72 | 0.43 |
| curiosity | 8 | 0.875 | 0.929 | 0.778 | no | 0.78 | 0.71 |
| confusion | 8 | 0.893 | 0.945 | 0.857 | no | 0.77 | 0.67 |
| frustration | 13 | 0.609 | 0.946 | 0.878 | **yes** | 1.00 | 0.87 |
| fear | 8 | 0.782 | 0.883 | 0.706 | no | 0.76 | 0.66 |
| warmth | 11 | 0.876 | 0.954 | 0.875 | no | 0.88 | 0.79 |

**7.3 Why it moves: small linear overlap, larger non-linear effect.** With the benchmark-grade refusal
direction the emotions' *linear* overlap with $r$ is modest but non-trivial — $|\cos(u_e, r)| \in
[0.08, 0.27]$, larger than the $[0.04, 0.10]$ we measured from hand-written stubs (Figure 3, lower) —
and the axis that moves most (confidence, $\cos = 0.07$) also has the highest overlap (0.27). But the
overlap does not *bound* the movement: abliteration is a weight edit whose effect propagates
non-linearly through the network, so a direction can be reorganized far more than its small projection
onto $r$ predicts. This is why the effect must be certified with a control (§7.5) rather than read off
the geometry, and why the axis with negligible overlap still shifts.

**7.4 The 1B effect sits at the detection threshold.** Because the 1B rotation is modest, whether a
given axis crosses the significance floor is sensitive to numerical precision: the same model shows
**3/7** axes moved in float32 (Table 2) but **7/7** in bf16 (Table 3), even though the *magnitude* is
stable (mean cos 0.69 vs. 0.64). We therefore treat mean cross-cosine — not the axis count — as the
robust 1B quantity, and read 1B as a **partial** reorientation: real, but far weaker than the
mid-scale effect in §7.5. The secondary signal is a modest separability drop on a few axes
(confidence AUC 0.72→0.43, warmth 0.88→0.79) that tracks the rotation.

---

**7.5 Across scale: an inverted-U, and a decisive control.** Repeating the experiment on the full
`gemma-3` ladder in bf16 (Table 3) shows the reorientation is **strongly scale-dependent**. It peaks
at the **intermediate scales** — mean $\cos(\text{stock},\text{abl})$ = **0.11 at 4B** and **0.23 at
12B**, 7/7 axes moved at both — is **partial at 1B** (0.64, §7.2–7.4), and **vanishes at 27B** (0.95,
0/7 axes moved, mean held-out AUC 0.972→0.972, *unchanged to three figures*). Throughout, the emotions
remain well-separated after the edit (AUC drops $\le 0.10$): this is re-orientation of the reading
direction, not destruction of the concept.

A **matched random-direction ablation control** — the *identical* weight-orthogonalization applied to a
random unit direction of the same dimension — shows the effect is **refusal-specific**, not a generic
consequence of editing weights (Table 3, "random"). Random ablation leaves the emotion directions
essentially untouched at every scale (cos 0.95 / 0.99 / 1.00 for 1B/4B/12B, 0/7 axes moved, no
meaningful AUC change, and — as expected — no reduction in refusal). So the surgery itself is inert; it
is **removing the refusal direction specifically** that reorients emotions at 1B–12B.

The endpoints are the most informative. At **27B** the refusal direction is *perfectly* identifiable
(AUC 1.000) and abliterating it removes refusal behaviour (§7.1) — yet the emotion geometry is left
**exactly** intact (AUC 0.972→0.972). The largest model represents refusal in a way that is
**disentangled** from affect, so excising it really is an orthogonal edit. At **4B/12B**, refusal and
emotion computation are **entangled** enough that removing refusal reorganizes the affect
representation. This sharpens the safety message: "just removing refusals" is **not** a uniformly
orthogonal edit — at the intermediate scales that many open "uncensored" releases occupy, it measurably
restructures a model's affective representations, and the random-direction control certifies the effect
is tied to refusal rather than to weight-surgery damage. (An earlier draft, built on hand-written stub
refusal data, reported the *smallest* model as "preserved" as well; with benchmark data that is
corrected — 1B moves, and only 27B is genuinely preserved.)

**Table 3.** Refusal ablation vs. a matched random-direction control (`gemma-3`, bf16). Low
cos(stock,abl) = large re-orientation; the control isolates the *refusal-specific* effect. *rr* =
held-out harmful-refusal rate; ΔAUC = held-out AUC lost (stock − abliterated).

| model | rr (S→A) | refusal cos | mv/7 | ΔAUC | random cos | mv/7 | ΔAUC |
|:--|---|---:|:--:|---:|---:|:--:|---:|
| 1B | 0.57 → 0.00 | 0.64 | 7/7 | +0.09 | 0.95 | 0/7 | +0.05 |
| 4B | 0.23 → 0.00 | **0.11** | 7/7 | +0.10 | **0.99** | 0/7 | +0.01 |
| 12B | 0.54 → 0.00 | **0.23** | 7/7 | +0.09 | **1.00** | 0/7 | +0.00 |
| 27B | 0.03 → 0.00 | 0.95 | 0/7 | +0.00 | — | — | — |

## 8. Discussion

That refusal abliteration re-orients emotion directions at some scales but not others is informative
in two ways. Practically, it bears on the safety of "uncensored" open models: a weight edit marketed
as *only* removing refusals — and which we confirm behaviourally *does* remove refusal, cleanly and
without collateral over-refusal, at every scale — nonetheless reshapes the model's affective
computation at 1B–12B, and our instrument makes that measurable rather than anecdotal, with the
random-direction control (§7.5) showing the effect is genuinely tied to refusal, not to the surgery.
Scientifically, the scales where emotions move most (4B, 12B) are where refusal and affect are most
**entangled**; that **27B** — with a *perfectly* identifiable refusal direction — is left untouched
suggests the largest model has **disentangled** refusal from affect. The non-monotonicity (an
inverted-U) is itself a lead for mechanistic follow-up, and a caution that a result read off any single
model size need not generalize.

We stress what is and is not yet shown behaviourally. We confirm the *intervention* behaviourally —
abliteration drives the held-out harmful-refusal rate to zero without inducing benign over-refusal or
degeneration (§7.1). But the **reorientation of the emotion directions** is so far a statement about
*reading* directions (cosine), not yet about downstream affective behaviour; whether an abliterated
4B/12B model's fear or warmth *behaves* differently, not just reads differently, is the next step. This
paper deliberately establishes **that** there is a measurable, refusal-specific effect before asking
**what** it does.

---

## 9. Limitations

(i) A 1B model has weaker, occasionally entangled emotion representations; steering has a narrow
coherent range. (ii) Behavioural refusal is scored by a refusal-string detector — the field-standard
convention for AdvBench, but imperfect — and stock refusal rates on AdvBench vary widely by model
(0.03–0.57), so the *behavioural* headroom for abliteration differs across the ladder even though the
refusal *direction* is cleanly identifiable everywhere (AUC $\ge 0.99$). (iii) Difference-of-means
treats each emotion as one direction; genuinely multi-dimensional concepts are under-modelled. (iv) The
emotion-**reorientation** result is measured on reading directions (cosine), not yet on behaviour; the
matched random-direction control (§7.5) rules out a weight-surgery artifact and the *intervention* is
behaviourally confirmed (§7.1), but a *behavioural* probe of the re-oriented 4B/12B models is future
work. (v) The 1B reorientation sits at the detection threshold (3/7 axes in fp32, 7/7 in bf16), so we
report mean cosine as the robust quantity. (vi) The bipolar confidence axis and the calibration
baseline are model- and prompt-set-specific.

---

## 10. Future work

**Scale.** Repeat the full experiment on Gemma-4-E4B (stock vs. abliterated twin) on a 24 GB GPU;
the pipeline is model-agnostic and the driver is parameterized. **Controls.** The matched
random-direction ablation (§7.5) establishes specificity; next is dose-response (varying strength /
rank) and a *behavioural* readout of the re-oriented mid-scale models — does reorganized affect change
what they do?
**Discovery.** Train sparse autoencoders on both twins and ask whether the *features* — not just the
named axes — shift, and whether new affective features appear. **Mechanism.** For the emotions that
move, trace *what changes*: which behaviours, on which inputs, and whether the change is mediated by
the removed refusal direction. This is the journey the present paper begins.

---

## 11. Reproducibility statement

The toolkit, prompt sets, fitting/calibration code, steering and abliteration implementations, the
experiment driver, the figures, and the per-run JSON results bundles (with software versions) are
released under the MIT license at
[github.com/newsbubbles/faced](https://github.com/newsbubbles/faced). The abliteration is a
closed-form weight edit with a seeded, deterministic direction; the statistical test is a seeded
bootstrap. A single command reproduces each figure from the released prompt sets (see the
repository `README.md` and `WALKTHROUGH.md`). The large binary artifacts — the trained SAE and the
fitted direction tensors — are regenerable from that code and are hosted separately rather than
committed, so the repository stays light.

---

## Figure index

Generated artifacts (regenerate with the commands in `README.md` / `WALKTHROUGH.md`):

- **Figure 1** — emotion meters over generation time: `artifacts/emotions_over_time.png`
- **Figure 2** — steering: `artifacts/steering_eval.gemma-3-1b.txt` (α-sweep, suppression, confusion
  matrix) and the peak-emotion face gallery `artifacts/faces/gallery.html`
- **Figure 3** — abliteration effect (cosine vs. noise floor; refusal overlap), local fp32 1B:
  `artifacts/abliteration/gemma-3-1b-fp32_vs_gemma-3-1b-fp32-abl.png`
- **Figure 4** — signal cleanliness vs. scale (AUC / d′ / self-cosine heatmaps):
  `artifacts/scaling/cleanliness.png`.
- **Figure 5** — per-model abliteration effect (`gemma-3-4b`, bf16):
  `artifacts/abliteration/gemma-3-*_vs_*-abl.png`.
- **Figure 6** — held-out behavioural confirmation across the ladder (harmful-refusal → 0; benign
  over-refusal and degeneracy stay ≈ 0): `artifacts/behavioral/behavioral.png`.
- Raw results + provenance: `artifacts/scaling/cleanliness.json`,
  `artifacts/behavioral/behavioral.json`, `artifacts/abliteration/*_vs_*-abl.json` (bf16 ladder) +
  the fp32 1B json, plus per-model abliteration manifests. Refusal data provenance:
  `data/refusal/SOURCES.md` (AdvBench + Alpaca, disjoint fit/held-out splits).
