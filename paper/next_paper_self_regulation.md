# Self-steering emotion: recursive affective self-regulation
## Research notes toward the *next* faced paper

> **Title candidates:** "Emotional Self-Regulation" (+ subtitle) · "Emotional Attention Is All You
> Need" · "Emotions Are All You Need". (Decide once the results shape the claim.)

*Idea from Nathaniel Gibson (July 2026), captured while the first paper (the faced
instrument + refusal-abliteration study) is in progress. This is the seed for the follow-up.*

---

## Core thesis

Give a model read **and write** access to its own emotion directions (the axes `faced`
extracts) and **close the loop**: the model reads its live affective state and steers itself.
Study what emerges — in particular, **emotional self-regulation as learned activation steering**.

`faced` is already the environment for this: it extracts the directions (the axes), reads them
live per token (the state — the *mixture over time*), and steers them (the action: add / ablate).
The new contribution is the **controller** that wires readout back into steering.

---

## The developmental analogy (the generative frame)

- Human infants begin with a coarse affective repertoire — arguably two poles: **discomfort /
  need** (≈ frustration, distress) and **alleviation / contentment** (calm, focused, "happy").
  *Open question:* is the adult range absent-then-**emerging**, or present-but-**undifferentiated**?
  (This is empirically answerable — see "Developmental emergence" below.)
- Regulation is initially **external**: the caregiver performs **distraction / diversion**.
  (*diversión* in Spanish = "fun," literally *diversion* — redirecting attention **is** the
  mechanism of both soothing and fun.)
- Distraction works by **activating a direction toward an engaged/entertained state, which
  competitively interrupts the distress direction.**
- Development = **internalizing** this: the child, then adult, learns to self-distract → to
  **self-regulate**. Regulation is the learned ability to steer one's own affect by activating some
  directions to suppress others.

---

## Mechanism hypotheses (all testable with `faced` today)

1. **Distraction = competitive suppression.** Activating direction B suppresses direction A.
   `faced`'s steering **confusion matrix** already shows cross-axis effects (steering one axis moves
   others). Next step: characterize the **inhibition graph** over the axes — which directions
   suppress which, and by how much. That graph *is* the vocabulary of self-distraction.
2. **Affect is a distribution, not a scalar.** At any moment the state is a **mixture** — a point in
   a simplex / a probability distribution over the axes — never a single emotion. The per-token
   meter traces (the emotions-over-time charts) already show overlapping, evolving mixtures, some
   non-obvious ("mixes that surprise"). Model the **dynamics** of this distribution: trajectories,
   attractors, transitions, and which mixtures are stable vs. transient.
3. **Self-regulation as a control loop.** `state (faced meters) → policy → steering action → new
   generation → new state`. A closed feedback loop over the model's own affect. The policy can be
   **homeostatic** (a hand-designed setpoint — the "internal parent") or **learned**.

---

## The recursive / learned version — "a model that learns that"

Train a controller (small head / adapter / RL policy) that reads `faced` meters and applies steering
to hold a target affective profile or to accomplish a goal. Questions:

- Does self-regulation **improve task performance / robustness** (as emotional regulation does in
  humans)?
- Does it develop **human-like strategies** — distraction, reappraisal, suppression — unprompted?
- Does the regulation policy **generalize** across tasks and prompts?
- Does write-access to one's own affect create **instability** (runaway self-amplification /
  rumination spirals)? If so, that's *itself* an interesting finding about feedback and affect.
- How do you reward the policy **without Goodharting its own meters** (it must regulate genuine
  state, not game the readout)?

---

## The safety through-line (why this is more than cute)

The origin conversation and the source paper note that rising **desperation** can drive misaligned
behavior (cheating, blackmail). A self-regulating emotional controller — one that **notices its own
desperation climbing** (a `faced` read) and **self-interrupts** (steers toward calm / curiosity)
**before acting** — is **affective homeostasis as a guardrail**. Rather than an external monitor,
the model regulates the very directions that *precede* unsafe actions. Emotional self-regulation
becomes an alignment mechanism.

**Minimal first experiment (the "internal parent" MVP):** on `faced` + gemma, a homeostatic
controller — *if the desperation meter exceeds a threshold, inject −desperation plus a little
curiosity ("distraction")* — and measure whether it prevents the desperation-driven bad behavior
from the origin demo. Reuses everything already built.

---

## A second study: developmental emergence over training time

Do the emotion axes emerge in an **order** during pretraining, mirroring infant→adult
differentiation? Using checkpointed model families (Pythia, OLMo, or any with saved checkpoints):

- Fit `faced` axes at successive training checkpoints.
- Do coarse **arousal/valence** (discomfort/alleviation) axes appear first, **differentiating later**
  into surprise / curiosity / etc.?
- Track the **cleanliness metrics** (AUC / d′ / bootstrap self-cosine — the same ones built for the
  scale ladder) over **training time**.

This complements the *scale* axis of the first paper with a **time** axis: cleanliness/differentiation
vs. parameters **and** vs. training steps.

---

## Connections

- **Control-graph project (State→Gradient→Curvature).** The self-regulation loop is literally a
  control graph over affect: the emotion simplex is the *state*, steering is the *action*, and the
  learned policy is the *curvature*. Worth unifying with that project's architecture.
- **Predictive processing / active inference.** Regulation as maintaining an affective setpoint —
  minimizing an affective prediction error (homeostasis → allostasis; Friston active inference).
- **Emergence-from-rules ("no bad guys").** Affect mixtures + regulation read as emergent dynamics
  of a rule-following system, not a designed feature.

---

## Open questions to seed the paper

- Absent-then-emerging vs. present-but-undifferentiated infant range → the checkpoint study.
- Right **action space** for self-steering: per-axis scalars? a target distribution over axes? a
  learned low-rank steering basis?
- The **inhibition graph** — is competitive suppression symmetric? sparse? does it have a valence
  structure (positive axes suppress negative and vice versa)?
- Does self-regulation trade off **expressivity** for **stability** (a flatter affective range)?
- What is the **attractor structure** of the free-running affect dynamics (with no external prompt)?

---

*Everything here is buildable on the current `faced` toolkit: `readout.py` (state), `hooks.py` /
`steering.py` (action), `stats.py` (cleanliness/dynamics), and the steering confusion matrix
(inhibition graph). The controller is the one new piece.*
