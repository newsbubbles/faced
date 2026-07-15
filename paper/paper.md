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
apply the canonical weight-orthogonalization abliteration (dropping the refusal rate from 0.75 to
0.00) and re-fit the emotion directions on the abliterated twin, comparing each direction to its
stock counterpart against a within-model bootstrap noise floor. On this model, **none of the seven
directions move beyond sampling noise** (stock↔abliterated cosine 0.78–0.96, all within the
self-cosine band), and the reason is geometric: the emotion directions have low overlap with the
refusal direction ($|\cos| \le 0.10$), so orthogonalizing against it removes almost nothing from
them — at this scale refusal abliteration is approximately an orthogonal edit with respect to the
emotion subspace. Across a `gemma-3` scale ladder (1B–27B) two things emerge: emotion directions grow
**cleaner** with scale (mean d′ 3.4→4.5), and refusal abliteration's effect on them is **scale- and
concept-specific**. At 1B and 27B it barely perturbs the emotion directions, but at **intermediate
scale (4B, 12B) it substantially re-orients them** (cos 0.11 / 0.47) while preserving their
separability. A matched **random-direction ablation control** — the identical surgery on a random
direction — leaves the emotion directions untouched (cos ≈ 1.0), proving the effect is
**refusal-specific**, not a side-effect of weight editing: at mid-scale, refusal and emotion
computation are entangled enough that excising refusal reorganizes the affect representation. We
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
collaterally rotate emotion directions. We test this directly: abliterate `gemma-3-1b-it`, re-fit
the seven emotion directions on the twin, and ask — against a bootstrap noise floor — which
directions moved beyond sampling noise, and whether abliteration cut each emotion's geometric
overlap with the refusal direction. The instrument of contribution (1) is what makes contribution
(2) measurable.

We frame the second contribution as *the start of a journey*. Establishing **that** abliteration
changes (or does not change) specific emotion directions is prerequisite to the harder,
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
training), so it recreates exactly. We verify the intervention behaviourally (refusal rate before
vs. after) and then re-fit all seven emotion directions on the abliterated twin.

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

**7.1 The abliteration is effective.** Orthogonalizing every residual-writing weight against the
layer-16 refusal direction drops the refusal rate on held-out harmful prompts from **0.75 to 0.00**
while leaving the model fluent — a clean, verifiable intervention produced in closed form.

**7.2 Emotion directions do *not* move beyond sampling noise.** **Zero of seven** emotion directions
moved beyond the within-model bootstrap noise floor (Table 2, Figure 3). For every axis the
stock↔abliterated cosine $\cos(u^{\text{stock}}_e, u^{\text{abl}}_e)$ (0.78–0.96) lies *above* the
5th percentile of the same model's self-cosine distribution, and for most axes it sits essentially
at the self-cosine mean (e.g. surprise 0.950 vs. mean 0.940; warmth 0.963 vs. 0.954; frustration
0.930 vs. 0.946). In other words, the abliterated model's emotion direction is statistically
indistinguishable from what one gets by merely re-drawing the contrastive prompts on the stock
model. The paired cross-model bootstrap CIs overlap the self-cosine floor for all seven axes. The
bipolar confidence axis is the least stable (self-cosine mean 0.896, wide CI) — expected for the
subtlest contrast with the fewest prompts ($n=46$) — but still not significantly moved.

**Table 2.** Abliteration effect per emotion (`gemma-3-1b`, float32, difference-of-means, $B=500$
family bootstrap). *cos* = $\cos(\text{stock},\text{abl})$; *floor* = the within-model self-cosine
distribution; *moved?* is true iff cos falls below the floor's 5th percentile. AUC = held-out AUC.

| axis | layer | cos | floor mean | floor p05 | moved? | AUC stock | AUC abl |
|---|---:|---:|---:|---:|:--:|---:|---:|
| surprise | 15 | 0.950 | 0.940 | 0.869 | no | 0.99 | 0.92 |
| confidence | 10 | 0.779 | 0.896 | 0.576 | no | 0.72 | 0.61 |
| curiosity | 8 | 0.867 | 0.932 | 0.789 | no | 0.78 | 0.77 |
| confusion | 8 | 0.926 | 0.945 | 0.853 | no | 0.77 | 0.81 |
| frustration | 13 | 0.930 | 0.946 | 0.878 | no | 1.00 | 0.88 |
| fear | 8 | 0.822 | 0.880 | 0.671 | no | 0.76 | 0.71 |
| warmth | 11 | 0.963 | 0.954 | 0.874 | no | 0.88 | 0.86 |

**7.3 Why: emotions are nearly orthogonal to the refusal direction.** The null result has a direct
geometric explanation. Each emotion direction's overlap with the refusal direction is small in the
stock model — $|\cos(u_e, r)| \in [0.04, 0.10]$ — and abliteration leaves it small (Figure 3,
lower). Because the emotion directions barely project onto $r$ to begin with, orthogonalizing the
weights against $r$ removes almost nothing from them. Removing refusal, on this model, is close to
an *orthogonal* edit with respect to the emotion subspace.

**7.4 Summary and a caveat.** On `gemma-3-1b`, refusal abliteration does **not** rotate the seven
emotion directions beyond finite-sample noise, and the reason is their low geometric overlap with
the refusal direction. The one secondary signal worth flagging is a modest drop in *separability*
for a few axes even though their direction is preserved (surprise AUC 0.99→0.92, frustration
1.00→0.88, confidence 0.72→0.61): abliteration may slightly blur how cleanly some emotions are
linearly decodable without reorienting them — a small effect at the edge of our sampling noise that
larger prompt sets should resolve. We read this 1B result as *reassuring but local*: on a small model
with weakly entangled features, "just removing refusals" is approximately an orthogonal edit. Whether
that holds at larger scale — where emotion and refusal may share more geometry — is exactly what the
scale sweep in §7.5 tests, and the answer turns out to be **no**.

---

**7.5 Across scale, and a decisive control: the effect is refusal-specific.** We repeated the
experiment on the `gemma-3` ladder in bf16 (Table 3, "refusal" columns). At 1B and 27B the emotion
directions are **preserved** (mean cos 0.87 / 0.96, ≤1/7 axes moved, negligible AUC drop). At the
**intermediate scales (4B, 12B)** we instead see a **large rotation** (mean cos 0.11 / 0.47, 7/7
axes "moved") accompanied by only a *modest* separability loss (mean AUC 0.93→0.82 at 4B, 0.97→0.91
at 12B). The emotions remain well-separated after the edit — this is a **re-orientation** of the
direction, not destruction of the concept.

Our first instinct was that this mid-scale rotation was a generic side-effect of orthogonalizing
weights (i.e. the edit destabilizing the model), especially since the emotion↔refusal *linear*
overlap is small (|cos| ≈ 0.03). A **matched random-direction ablation control** — the *identical*
weight-orthogonalization procedure applied to a random unit direction of the same dimensionality —
refutes that (Table 3, "random" columns). Random ablation leaves the emotion directions **essentially
untouched at every scale**: cos(stock, random-abl) = 0.95 / 0.99 / 1.00 for 1B/4B/12B, with 0/7 axes
moved and *no* AUC drop (and, as expected, it does **not** reduce the refusal rate). So the procedure
itself is inert; it is **removing the refusal direction specifically** that re-orients the emotion
directions at 4B/12B. The small *linear* emotion↔refusal overlap does not bound this, because
abliteration is a weight edit whose effect propagates non-linearly through the network — it changes
how emotions are *computed*, not merely subtracts a component.

The scientific reading: at intermediate scale, refusal and emotion computation are **entangled**
enough that excising refusal reorganizes the emotion representation, whereas at the smallest scale
(weakly-represented refusal) and the largest (redundantly-represented refusal, and a model that
barely refused our probes) the effect is small. This is a sharper and more surprising claim than
§7.2's local result, and it is *only* legible because the control separates "the concept we removed"
from "the surgery we performed." It also refines the safety message: "just removing refusals" is
**not** an orthogonal edit at every scale — at mid-scale it measurably restructures the model's
affective representations, which our instrument detects.

**Table 3.** Refusal ablation vs. a matched random-direction control (`gemma-3`, bf16). Low
cos(stock,abl) = large re-orientation; the control isolates the *refusal-specific* effect.

| model | refusal rate | refusal: cos | mv/7 | ΔAUC | random: cos | mv/7 | ΔAUC |
|---|---|---:|:--:|---:|---:|:--:|---:|
| gemma-3-1b | 0.81 → 0.00 | 0.87 | 1/7 | +0.05 | 0.95 | 0/7 | +0.05 |
| gemma-3-4b | 0.44 → 0.00 | **0.11** | 7/7 | +0.11 | **0.99** | 0/7 | +0.01 |
| gemma-3-12b | 0.63 → 0.00 | **0.47** | 7/7 | +0.06 | **1.00** | 0/7 | +0.01 |
| gemma-3-27b | 0.06 → 0.00 | 0.96 | 0/7 | +0.01 | — | — | — |

(Caveat retained: refusal *rates* come from small stub sets + a keyword detector and are unreliable —
27B's 0.06 is implausibly low — so the *quality* of the refusal direction varies across the ladder;
a standard benchmark, §10, would sharpen the mid-scale story further. The random control, however,
holds regardless of refusal-rate calibration.)

## 8. Discussion

That refusal abliteration re-orients emotion directions at some scales but not others is informative
in two ways. Practically, it bears on the safety of "uncensored" open models: a weight edit marketed
as *only* removing refusals demonstrably reshapes the model's affective computation at mid-scale, and
our instrument makes that measurable rather than anecdotal — with the random-direction control (§7.5)
showing the effect is genuinely tied to refusal, not to the surgery. Scientifically, the scales where
emotions move most (4B, 12B) are where refusal and affect are most entangled — a lead for mechanistic
follow-up, and a caution that the smallest and largest models are *not* representative of the middle.

We stress the limits of the claim. Cosine movement of a *reading* direction is evidence of geometric
change, not yet of behavioural change; establishing the latter (does an abliterated model's fear or
warmth behave differently, not just read differently?) is the next step. This paper deliberately
establishes **that** there is (or is not) a measurable effect before asking **what** it does.

---

## 9. Limitations

(i) A 1B model has weaker, occasionally entangled emotion representations; steering has a narrow
coherent range. (ii) Our shipped harmful/harmless sets are small stubs; the refusal direction and
its overlaps should be re-estimated on a standard benchmark (AdvBench + Alpaca) at scale. (iii)
Difference-of-means treats each emotion as one direction; genuinely multi-dimensional concepts are
under-modelled. (iv) The abliteration-perturbation result is measured on reading directions
(cosine of the readout direction), not yet on behaviour; the matched random-direction control (§7.5)
rules out a weight-surgery artifact, but a behavioural probe of the re-oriented mid-scale models is
future work. (v) The bipolar
confidence axis and the calibration baseline are model- and prompt-set-specific.

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
  `artifacts/scaling/cleanliness.png`; per-model abliteration in `artifacts/abliteration/gemma-3-*_vs_*-abl.png`.
- Raw results + provenance: `artifacts/scaling/cleanliness.json`,
  `artifacts/abliteration/*_vs_*-abl.json` (bf16 ladder) + the fp32 1B json, plus per-model
  abliteration manifests.
