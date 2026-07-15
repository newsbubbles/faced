# Future directions — the faced research program

`faced` (a live emotion instrument + causal steering) opens several distinct papers. This is the
running index. The lead follow-up (self-regulation) has its own file; the others are captured here
as they come up.

---

## 1. Recursive affective self-regulation  *(lead follow-up)*
→ [`next_paper_self_regulation.md`](next_paper_self_regulation.md). Give a model read+write access to
its own emotion directions and close the loop; distraction = competitive suppression; affect as a
distribution; safety = self-interrupt rising desperation before it drives misalignment.

---

## 2. Affective telemetry for tool-calling agents

**Core.** In an agent loop, a tool returns text the model must ingest before continuing. Read the
`faced` state **at tool-call boundaries**: (a) *over the tool-output tokens* as the model processes
them — surprise should spike exactly where a result violates expectation, the way our
missing-attachment demo already spikes (that demo **is** a tool-boundary signal); and (b) in the
*continuation* right after, where frustration/confusion/confidence about the result live.

**Signals & their triggers.**
- **surprise** — tool returns an unexpected value / schema / error.
- **frustration / desperation** — repeated tool failures on the same subgoal (this is *exactly* the
  origin paper's desperation, now localized to the agent loop).
- **confusion** — contradictory results across tools.
- **confidence↑** — result confirms the plan.

**The actionable part (the user's question).** Wire the detected state to a **deterministic
intervention**: if desperation crosses a threshold after N failed calls, *force* a strategy change —
re-plan, widen search, ask for help, or back off — instead of letting the model stochastically
persist (or, per the origin paper, start bending rules). This is a **control policy over the agent
loop driven by affective telemetry**, and it's the self-regulation idea (§1) applied to *behavior*,
not just tone.

**Instruction-tuned caveat, turned into the experiment.** Yes, instruct models are *implicitly*
trained to switch approach on frustration — so the contribution is to make that signal **explicit,
measurable, and reliably controllable** rather than emergent-and-stochastic. Measure: do instruct
models already self-switch when their frustration meter climbs (read `faced` during their agent
run)? Then add the deterministic controller and measure the delta — fewer wasted retries, fewer
rule-bending / reward-hacking moves, faster recovery.

**Safety framing.** The origin paper's desperation→cheating occurs precisely in agentic/benchmark
contexts. An **affect-triggered circuit breaker** — detect rising desperation on a hard task, force a
safe re-plan — is a concrete alignment mechanism for tool-using agents.

**Setup.** Instrument an agent harness at every tool boundary; inject controlled tool
failures/surprises; log per-boundary meters; correlate with the model's behavior change; then A/B the
deterministic controller.

---

## 3. Emotion representations across model types: base vs instruct vs thinking

**Core.** Fit `faced` axes on **base** (pretrained-only), **instruction-tuned**, and
**thinking/reasoning** variants — same family where possible (e.g. Qwen base vs instruct; a reasoning
variant) — and compare cleanliness (AUC / d′ / self-cosine) and geometry. This adds a
**training-stage / model-type** axis alongside this paper's **scale** axis.

**Questions.**
- Do **base** models carry emotion directions at all, or do they **emerge / sharpen** with
  instruction tuning and RLHF? (Alignment training as a *differentiation* stage — the developmental
  frame again: base = undifferentiated, instruct = socialized, thinking = deliberating.)
- Do **thinking** models show emotion **dynamics during the `<think>` phase** — frustration building
  through a hard chain, surprise at a contradiction, an "aha"/relief on solution? The reasoning trace
  is a long, rich substrate — potentially the most vivid emotions-over-time signal we can produce.
- Does the emotion→behavior **causal** link (steering effect size) strengthen with alignment
  training?

**Why it matters.** It separates *what the pretrained model represents* from *what alignment shapes*,
and tells us whether the safety-relevant affect (desperation) is a pretraining artifact or an
alignment-induced structure — which changes how you'd intervene on it.

---

## Cross-cutting structure

The program has a few orthogonal axes, all measured with the same instrument:

| axis | paper |
|---|---|
| **scale** (1B→27B) | this paper (cleanliness ladder) |
| **training stage / model type** (base / instruct / thinking) | §3 |
| **training time** (checkpoints) | dev-emergence study in §1's notes |
| **the loop** (read→steer→read) | §1 self-regulation |
| **the agent boundary** (tool calls) | §2 |
| **robustness** (abliteration) | this paper |

**The through-line is safety:** the origin observation was *desperation → misbehavior*. Every paper
here turns that into a measurable, and eventually controllable, handle — read it (this paper),
localize it to the agent loop (§2), regulate it (§1), and understand where in training it comes from
(§3).
