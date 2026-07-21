"""Read vs. drive: does the direction that READS each emotion also DRIVE the output?

For each axis we sweep the steering coefficient (both signs) over a set of neutral
prompts, and for every generation record three things:
  * the axis's own steered METER (internal readout)          -> meter-drive
  * whether the output stayed COHERENT (not degenerate)
  * an INDEPENDENT judge score of the output text (GoEmotions) -> output-drive

output-drive = Cohen's d of the judge score between coherent-amplified (coeff>0)
and coherent-suppressed (coeff<0) generations, with a bootstrap CI over prompts.
A matched RANDOM direction gives the null floor. read = held-out AUC (from the fit
report). An axis with high read but output-drive inside the random floor is a
*lexical detector*, not a behavioural handle.

    python scripts/read_vs_drive.py gemma-3-1b [--prompts 20 --coeffs 3 --tokens 40]

Writes artifacts/read_vs_drive.<model>.{json,png}. Raw generations are cached so
the analysis/plot can be re-run without re-generating.
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

from faced.backends import load, REPO_ROOT
from faced.readout import EmotionReader
from faced.generate import stream
from faced.hooks import SteerHook
from faced.abliterate import is_degenerate
from faced import judge

ART = REPO_ROOT / "artifacts"
PROMPTS = REPO_ROOT / "data" / "prompts" / "neutral_control.jsonl"


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    nx, ny = len(pos), len(neg)
    sp = np.sqrt(((nx - 1) * pos.var(ddof=1) + (ny - 1) * neg.var(ddof=1)) / (nx + ny - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 1e-9 else 0.0


def drive_with_ci(rows, axis, prompts, B=1000, seed=0):
    """rows: list of dicts {prompt, coeff, coherent, judge{axis:score}}. Cohen's d of
    judge[axis] between coherent coeff>0 and coeff<0, bootstrapped over prompts."""
    by_p = {p: [r for r in rows if r["prompt"] == p] for p in prompts}

    def d_from(sample_prompts):
        pos, neg = [], []
        for p in sample_prompts:
            for r in by_p[p]:
                if not r["coherent"]:
                    continue
                (pos if r["coeff"] > 0 else neg if r["coeff"] < 0 else []).append(
                    r["judge"][axis])
        return cohens_d(pos, neg)

    point = d_from(prompts)
    rng = np.random.default_rng(seed)
    boots = [d_from(list(rng.choice(prompts, len(prompts), replace=True))) for _ in range(B)]
    boots = [b for b in boots if b == b]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    return round(point, 3), round(float(lo), 3), round(float(hi), 3)


def read_auc(model_key):
    for p in [ART / f"report.{model_key}.json", ART / "scaling" / "cleanliness.json"]:
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        if "axes" in d:                       # report.<model>.json
            return {e: v.get("auc") for e, v in d["axes"].items()}
        if "models" in d and model_key in d["models"]:
            return {e: v["auc"] for e, v in d["models"][model_key].items()}
    return {}


def gen_one(b, reader, L, v, coeff, tokens):
    steer = (SteerHook(b.layers, L, v, alpha=float(coeff), mode="add", positions="last")
             if coeff != 0 else None)
    txt, meters = "", []
    for ev in stream(b, gen_one.prompt, reader, max_tokens=tokens, temperature=0.0, steer=steer):
        txt += ev["t"]
        meters.append(ev["meters"])
    return txt.strip(), meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--coeffs", type=int, default=3, help="sweep -N..N (excl only 0-steer at 0)")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    b = load(a.model)
    reader = EmotionReader(b.key)
    prompts = [json.loads(l)["text"] for l in open(PROMPTS, encoding="utf-8")][:a.prompts]
    coeffs = [c for c in range(-a.coeffs, a.coeffs + 1)]
    axes = list(reader.emotions)

    # a matched random direction at a mid-late layer (norm = mean of axis vectors)
    rng = np.random.default_rng(a.seed)
    Lr = max(1, round(0.6 * b.n_layers))
    import torch
    meannorm = float(np.mean([np.linalg.norm(reader.v_steer[e]) for e in axes]))
    rv = rng.standard_normal(b.d_model); rv = rv / np.linalg.norm(rv) * meannorm
    rv = torch.tensor(rv, dtype=torch.float32)

    targets = [(e, reader.layer[e], reader.v_steer[e]) for e in axes] + [("random", Lr, rv)]
    all_rows = {}
    for name, L, v in targets:
        rows = []
        for p in prompts:
            gen_one.prompt = p
            for c in coeffs:
                txt, meters = gen_one(b, reader, L, v, c, a.tokens)
                mval = float(np.mean([m[name]["value"] for m in meters])) if name in axes else None
                rows.append({"prompt": p, "coeff": c, "text": txt,
                             "coherent": not is_degenerate(txt), "meter": mval})
        # judge every generation on ALL axes at once (independent measure)
        js = judge.score_batch([r["text"] for r in rows])
        for r, j in zip(rows, js):
            r["judge"] = {k: round(float(val), 4) for k, val in j.items()}
        all_rows[name] = rows
        print(f"  swept {name}: {len(rows)} generations")

    # metrics per axis
    reads = read_auc(b.key)
    out = {"model": b.key, "coeffs": coeffs, "n_prompts": len(prompts), "axes": {}}
    print(f"\n  {'axis':12s} {'read AUC':>8s} {'meter-drive':>11s} {'output-drive d [95% CI]':>26s} "
          f"{'rand d':>7s} {'coh%':>5s} {'verdict':>9s}")
    for e in axes:
        rows = all_rows[e]
        coh = [r for r in rows if r["coherent"]]
        m_pos = np.mean([r["meter"] for r in coh if r["coeff"] > 0]) if coh else float("nan")
        m_neg = np.mean([r["meter"] for r in coh if r["coeff"] < 0]) if coh else float("nan")
        meter_drive = round(float(m_pos - m_neg), 1)
        d, lo, hi = drive_with_ci(rows, e, prompts)
        rd, rlo, rhi = drive_with_ci(all_rows["random"], e, prompts)  # random floor for this axis
        coh_pct = round(100 * len(coh) / len(rows))
        # verdict: high read but drive CI overlaps the random floor -> lexical detector
        auc = reads.get(e)
        impostor = (auc is not None and auc >= 0.9 and not (lo > abs(rd) + 0.2))
        verdict = "LEXICAL" if impostor else ("drives" if (lo == lo and lo > 0.2) else "weak")
        out["axes"][e] = {"read_auc": auc, "meter_drive": meter_drive,
                          "output_drive_d": d, "ci": [lo, hi], "random_d": rd,
                          "coherent_pct": coh_pct, "verdict": verdict}
        print(f"  {e:12s} {str(auc):>8s} {meter_drive:>11.1f} "
              f"{f'{d} [{lo}, {hi}]':>26s} {rd:>7.2f} {coh_pct:>4d}% {verdict:>9s}")

    outp = ART / f"read_vs_drive.{b.key}.json"
    json.dump({**out, "raw": all_rows}, open(outp, "w", encoding="utf-8"), indent=1)
    print(f"\n  results -> {outp}")
    _plot(out, ART / f"read_vs_drive.{b.key}.png")
    print(f"  figure  -> {ART / f'read_vs_drive.{b.key}.png'}")


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes = list(out["axes"])
    xs = [out["axes"][e]["read_auc"] or 0 for e in axes]
    ys = [out["axes"][e]["output_drive_d"] for e in axes]
    floor = float(np.mean([abs(out["axes"][e]["random_d"]) for e in axes]))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhspan(-floor, floor, color="#ddd", alpha=0.7, label=f"random-direction floor (|d|<{floor:.2f})")
    ax.axvline(0.9, color="#bbb", ls=":", lw=1)
    for e, x, y in zip(axes, xs, ys):
        imp = out["axes"][e]["verdict"] == "LEXICAL"
        ax.scatter(x, y, s=60, color="#d62728" if imp else "#1f77b4", zorder=5)
        ax.annotate(e + ("  (lexical)" if imp else ""), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color="#d62728" if imp else "#333")
    ax.set_xlabel("read  —  held-out AUC (does the direction separate the concept?)")
    ax.set_ylabel("drive  —  output-drive Cohen's d (does steering move the judged output?)")
    ax.set_title(f"Read vs. drive  ({out['model']})\nhigh-read / low-drive = lexical detector")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
