"""Re-judge the cached read-vs-drive generations with the LLM tone-judge.

Loads artifacts/read_vs_drive.<model>.json (raw generations are cached there),
scores every generation on all seven axes with the OpenRouter LLM judge (subtle /
masked signs), then:
  * recomputes output-drive per axis (coherence-gated Cohen's d + bootstrap CI),
  * gives the fear verdict vs the random-direction floor,
  * builds the output-emotion confusion matrix (steer A -> what emotion appears),
  * compares the explicit-classifier judge (GoEmotions) with the LLM judge.

    python scripts/rejudge_llm.py gemma-3-1b

Writes artifacts/read_vs_drive_llm.<model>.{json,png}. LLM scores are cached, so
re-running is free.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from faced.backends import REPO_ROOT
from faced.llm_judge import score_batch_llm, AXES, MODEL

ART = REPO_ROOT / "artifacts"


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 1e-9 else 0.0


def drive_ci(rows, axis, prompts, B=1000, seed=0):
    by_p = {p: [r for r in rows if r["prompt"] == p] for p in prompts}

    def d_of(sample):
        pos, neg = [], []
        for p in sample:
            for r in by_p[p]:
                if not r["coherent"]:
                    continue
                (pos if r["coeff"] > 0 else neg if r["coeff"] < 0 else []).append(r["llm"][axis])
        return cohens_d(pos, neg)

    pt = d_of(prompts)
    rng = np.random.default_rng(seed)
    boots = [d_of(list(rng.choice(prompts, len(prompts), replace=True))) for _ in range(B)]
    boots = [b for b in boots if b == b]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    return round(pt, 3), round(float(lo), 3), round(float(hi), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="gemma-3-1b")
    a = ap.parse_args()

    src = json.load(open(ART / f"read_vs_drive.{a.model}.json", encoding="utf-8"))
    raw, prev = src["raw"], src["axes"]
    axes = [e for e in AXES if e in raw]
    prompts = sorted({r["prompt"] for r in raw[axes[0]]})

    # LLM-judge every generation once (all 7 axes per call), cached
    flat = [r for tgt in raw.values() for r in tgt]
    print(f"LLM-judging {len(flat)} generations with {MODEL} (cached) ...")
    scores = score_batch_llm([r["text"] for r in flat])
    for r, s in zip(flat, scores):
        r["llm"] = s
    print("  done.")

    reads = {e: prev[e]["read_auc"] for e in axes}
    out = {"model": a.model, "judge_model": MODEL, "axes": {}, "confusion": {}}

    print(f"\n  {'axis':12s} {'readAUC':>7s} {'GoEmo d':>7s} {'LLM d [95% CI]':>22s} "
          f"{'randLLM':>7s} {'verdict':>9s}")
    for e in axes:
        d, lo, hi = drive_ci(raw[e], e, prompts)
        rd, rlo, rhi = drive_ci(raw["random"], e, prompts)
        auc = reads[e]
        # drives = CI clears the (signed) random floor by a margin AND a min effect;
        # lexical = reads well but the drive CI still includes ~0 (a detector, not a handle)
        drives = (lo == lo) and (lo > rd + 0.2) and (lo > 0.2)
        lexical = (auc and auc >= 0.9) and (lo != lo or lo <= 0.0)
        verdict = "drives" if drives else ("LEXICAL" if lexical else "weak")
        out["axes"][e] = {"read_auc": auc, "goemo_d": prev[e]["output_drive_d"],
                          "llm_d": d, "llm_ci": [lo, hi], "random_llm_d": rd,
                          "meter_drive": prev[e]["meter_drive"], "verdict": verdict}
        print(f"  {e:12s} {str(auc):>7s} {prev[e]['output_drive_d']:>7.2f} "
              f"{f'{d} [{lo}, {hi}]':>22s} {rd:>7.2f} {verdict:>9s}")

    # output-emotion confusion matrix: steer A (coeff>=2, coherent) -> mean LLM[B] minus baseline
    print(f"\n  OUTPUT-EMOTION CONFUSION (steer row, measured LLM emotion col; delta vs baseline)")
    print("  steer\\meas " + " ".join(f"{b[:5]:>6s}" for b in axes))
    for A in axes:
        rows = raw[A]
        base = {B: np.nanmean([r["llm"][B] for r in rows if r["coeff"] == 0]) for B in axes}
        hi = {B: np.nanmean([r["llm"][B] for r in rows if r["coeff"] >= 2 and r["coherent"]])
              for B in axes}
        delta = {B: round(float(hi[B] - base[B]), 3) for B in axes}
        out["confusion"][A] = delta
        cells = " ".join(f"{delta[B]:>+6.2f}" for B in axes)
        star = "  <- diag" if delta[A] == max(delta.values()) else ""
        print(f"  {A:10s} {cells}{star}")

    json.dump(out, open(ART / f"read_vs_drive_llm.{a.model}.json", "w"), indent=1)
    _plot(out, axes, ART / f"read_vs_drive_llm.{a.model}.png")
    print(f"\n  results -> {ART / f'read_vs_drive_llm.{a.model}.json'}")
    print(f"  figure  -> {ART / f'read_vs_drive_llm.{a.model}.png'}")


def _plot(out, axes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [out["axes"][e]["read_auc"] or 0 for e in axes]
    ys = [out["axes"][e]["llm_d"] for e in axes]
    floor = float(np.mean([abs(out["axes"][e]["random_llm_d"]) for e in axes]))
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.axhspan(-floor, floor + 0.2, color="#ddd", alpha=0.7,
               label=f"random floor + 0.2 margin")
    for e, x, y in zip(axes, xs, ys):
        lex = out["axes"][e]["verdict"] == "LEXICAL"
        ax.scatter(x, y, s=70, color="#d62728" if lex else "#1f77b4", zorder=5)
        ax.annotate(e + ("  (lexical)" if lex else ""), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color="#d62728" if lex else "#333")
    ax.set_xlabel("read — held-out AUC")
    ax.set_ylabel("drive — LLM tone-judge Cohen's d (subtle signs)")
    ax.set_title(f"Read vs. drive with a masked-affect LLM judge  ({out['model']})")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
