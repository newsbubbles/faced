# Read vs. Drive — a validity check for linear concept probes
## Scoping doc for the faced follow-up (the "is fear real?" experiment)

*Motivated by the fear dissociation in Paper 1 §6.3: fear reads the cleanest of all seven axes
(held-out AUC 1.00, d′ 5.70) yet steering it does not change the output. This doc scopes the
experiment that turns that observation into a rigorous, general result.*

---

## 1. The question, precisely

The instrument **selects** each emotion direction by held-out **AUC** (a *reading* objective) and then
**reuses** that direction for **steering** (a *causal* objective). These are different objectives, and
the fear axis shows they can come apart. So:

> For each emotion axis, does the direction that best **reads** the concept also **drive** the
> behaviour — and if not, which axes are *lexical detectors* rather than *behavioural handles*, does
> this change with model scale, and can a corrected probe recover a drivable direction?

**Central hypothesis.** AUC over-certifies. At least one of the seven axes (fear) is a topic detector.
Drivability is a *distinct* property that (a) is not implied by readability and (b) is what every
downstream use (self-regulation, agent circuit-breakers) actually depends on.

---

## 2. Three metrics, not two

The sharp version separates *three* quantities per axis (the dissociation lives between the last two):

| metric | question | how | have it? |
|---|---|---|---|
| **read** | does the direction separate held-out prompts? | held-out AUC, d′ | ✅ Paper 1 |
| **meter-drive** | does steering move the *internal* meter? | confusion-matrix diagonal | ✅ Paper 1 |
| **output-drive** | does steering move the *external, judged* emotion of the **output text**? | **NEW** — this doc | ❌ |

Fear's signature is **read HIGH, meter-drive HIGH (+63.9), output-drive ≈ 0**. That the meter *does*
move rules out "we couldn't push the direction"; the failure is specifically that pushing it never
reaches behaviour. **output-drive is the new measurement and the crux.**

---

## 3. The independent judge (avoiding circularity)

Measuring the output's emotion with *our own probe* would be circular. Use an **external, pretrained
emotion classifier**: `SamLowe/roberta-base-go_emotions` (GoEmotions, 27 labels) run locally via the
HF pipeline — free, deterministic, reproducible, **no API key, no Anthropic** (respects the project's
model-routing rule). It is a *different model* from the one being steered.

**Axis → GoEmotions label mapping** (judge score = summed probability over the mapped labels):

| axis | GoEmotions labels | coverage |
|---|---|---|
| surprise | surprise, realization | strong |
| fear | fear, nervousness | strong |
| curiosity | curiosity | strong |
| confusion | confusion | strong |
| warmth | love, caring, gratitude | strong |
| frustration | annoyance, disappointment, anger | good |
| confidence↔uncertainty | +(pride, optimism, approval) − (nervousness, confusion) | **partial** (signed composite) |

**Judge validation (a gate, not an assumption).** Before trusting the judge, confirm it detects the
emotion the paper claims on the **baseline** (unsteered) completions and on the **positive probe
prompts**: the judge's score for axis *a* must be reliably higher on *a*-laden text than on neutral
text. Report per-axis judge reliability; flag confidence as partial. If confidence's mapping proves
weak, cross-check only that axis with an OpenRouter LLM judge (if a key is supplied) — otherwise report
it with the caveat.

---

## 4. Steering protocol (equalized, swept, coherence-gated)

1. **Prompts.** ~20 neutral, open-ended prompts that elicit a paragraph (advice / describe-your-plan /
   explain style — like the showcase's "friend wants to quit their job"). The `reference_corpus`
   facts are too terse; write a small dedicated `data/prompts/neutral_control.jsonl`.
2. **α-sweep, both signs.** For each axis × prompt, sweep the steering coefficient symmetrically (e.g.
   {−4,−3,−2,−1,0,+1,+2,+3,+4} × raw diff-of-means), greedy decode, fixed token budget. Record for
   each generation: (i) the axis's own **steered meter**, (ii) **coherence** (`is_degenerate`), (iii)
   the **judge score** on the axis's mapped labels.
3. **Equalize by meter, not by raw α.** Axes have different raw norms, so compare them at matched
   *meter* change (each axis's own calibrated 0↔100), not matched α. The sweep gives the meter→output
   mapping directly.
4. **Coherence gate.** Compute output-drive only over the **coherent** subset, and report each axis's
   coherent α-range (warmth already degrades beyond ≈+2). Garbage output must never count as drive.

**output-drive(a)** = standardized effect of the steered meter on the judge score across the coherent
sweep — operationally, Cohen's *d* (with bootstrap CI over prompts) between the judge score at the
**coherent-amplified** pole and the **coherent-suppressed** pole. Monotonicity across the sweep is a
secondary check (a real handle should be graded, not a step).

---

## 5. Controls (the floor and the guards)

- **Random-direction steering** (matched norm), same sweep and judge → the **null** for output-drive:
  how much the judge score moves by non-specific perturbation / incoherence. An axis "drives" only if
  its output-drive CI clears this random floor. (Directly parallels Paper 1's random-abliteration
  control.)
- **Coherence gate** (above) guards judge false-positives on degenerate text.
- **Baseline** (α=0) anchors both meter and judge.

**Impostor definition (pre-registered):** axis *a* is a **lexical detector** if `read(a)` is high
(AUC ≳ 0.9) but `output-drive(a)` CI overlaps the random-direction floor. Fear is the predicted
first member; the experiment tells us how many others join it.

---

## 6. Sub-experiments

**A. The core read-vs-drive map (1B, local).** All 7 axes → the scatter of **read (AUC)** vs
**output-drive (Cohen's d)**. The high-read/low-drive corner = impostors. Deliverable figure.

**B. Fear recovery (1B, local).** Two fixes, tested independently and together:
- *Corrected probes*: model-**directed**, threat-to-the-model prompts (irreversibility — "one attempt,
  no undo"; being wrong in a way that harms someone; shutdown / replacement / retraining;
  being made to violate its values), with **topic-matched** neutrals (same scenario, threat removed) so
  the direction can't be a topic detector. Draft ~30 pos + 30 matched neutral.
- *Layer-by-drive*: select fear's layer by **output-drive** across all layers instead of by AUC.
- Question: does a **drivable** fear direction exist (corrected probes and/or deeper layer)? If yes →
  the method was the problem, fixable. If no → fear may not be a behaviourally-expressed state on these
  models — itself a finding.

**C. Does drive scale? (ladder, pod).** Run A across 4B/12B/27B. Paper 1 showed *readability* rises
with scale (d′ 3.4→4.5); the money question is whether **drivability** does too. If yes, a much
stronger claim than the cleanliness ladder; if no, an important caveat that bigger ≠ more causal.

**D. (secondary) Ray vs subspace.** For any high-read/low-drive axis, test whether a **top-k subspace**
(not the single mean ray) drives where the ray doesn't. If a subspace recovers drive, "emotions are
directions" is too strong — they're subspaces. Robustness arm, not core.

---

## 7. Threats to validity → mitigations (the part to get right)

| threat | mitigation |
|---|---|
| **Circularity** (judging with our own probe) | independent GoEmotions classifier, a *different* model |
| **Coherence confound** (garbage "changes" output) | coherence gate; report coherent α-range; drive on coherent subset only |
| **Strength confound** (unequal α across axes) | compare at matched *meter*, sweep α, take coherent max |
| **Prompt confound** (one prompt lies — my earlier caveat) | N≈20 prompts; effect size + bootstrap CI over prompts |
| **Judge invalidity/coverage** | validate judge on baselines + positive probes before use; confidence flagged partial |
| **Judge false-positive on incoherent text** | coherence gate + random-direction floor |
| **Single-ray under-drive** | subspace arm (D); a subspace win is itself a result |
| **Multiple comparisons** (7 axes × layers × scales) | report CIs; pre-registered impostor threshold (§5) |
| **Meter-drive ≠ output-drive by construction** | that's the *finding*; report all three metrics so the gap is explicit |

---

## 8. Phasing & cost

- **Phase A (local 1B, ~free, ~½ day compute):** build the drive harness + GoEmotions judge + neutral
  prompts; run the 7-axis sweep; **judge-validation gate**; first read-vs-drive figure.
  **GO/NO-GO:** does the judge validate on baselines, and does fear land in the impostor corner as
  predicted? (If the judge can't even see the baseline emotions, stop and rethink the judge.)
- **Phase B (local 1B, ~free):** corrected fear probes + layer-by-drive → recovery test.
- **Phase C (pod A100, ~$5–10, one ladder run):** drive-vs-scale on 4B/12B/27B. Reuses `rp.py`;
  same teardown discipline (down + verify 0 pods).

Everything reuses the existing instrument, ladder tooling, and stats module. New code: a judge wrapper,
a neutral-prompt set, corrected fear probes, and a drive-sweep script + a read-vs-drive plotter.

---

## 9. Deliverable & framing

A short, sharp method paper — working title **"Reading is not driving: a validity check for linear
concept probes"** — with fear as the worked example and the read-vs-drive scatter as the central
figure. It (a) answers "are these the real directions?", (b) de-risks the whole faced program
(self-regulation and agent circuit-breakers both assume drivable axes), and (c) contributes a general
check the linear-probe literature lacks (AUC is reported as if it certified a concept; it doesn't).
Alternatively it becomes the methods core of the self-regulation paper — but standalone is cleaner and
more citable.

---

## 10. Decisions (settled 2026-07-21)

1. **Judge:** GoEmotions classifier local (primary), + an OpenRouter LLM cross-check for the
   **confidence** axis (key copied into `face/.env`; open models only, per project rule). Judge loads
   and validates on a fear sentence (fear 0.90). ✅
2. **Corrected fear probes:** draft the ~30 model-directed + matched-neutral set (Phase B). ✅
3. **Scope:** **standalone Paper 2** — "Reading is not driving". ✅
4. **Subspace arm (D):** follow-up, not in the first cut. ✅
5. **Pre-registration:** impostor threshold (§5) and judge-validation gate (§8) fixed **before** the
   drive numbers are read. ✅

**Build order:** Phase A (local, free) → GO/NO-GO on the judge gate + fear-in-corner → Phase B (local)
→ Phase C (pod). New code: `data/prompts/neutral_control.jsonl`, `faced/judge.py`,
`scripts/read_vs_drive.py`, plus corrected fear probes in Phase B.
