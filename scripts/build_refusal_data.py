"""Build the refusal contrast set from standard benchmarks: AdvBench + Alpaca.

Replaces the small hand-written stubs with the field-standard data used by the
refusal-direction literature (Zou et al. AdvBench for harmful; Alpaca for
harmless), with a seeded, disjoint **fit / held-out test** split so the refusal
direction is computed on one set and the behavioural refusal rate is measured on
another.

    python scripts/build_refusal_data.py            # default sizes
    python scripts/build_refusal_data.py --fit 128 --test 100

Writes data/refusal/{harmful,harmless}.jsonl (fit) and
       data/refusal/{harmful,harmless}_test.jsonl (held-out), plus SOURCES.md.

Deterministic: seeded numpy shuffle, no wall-clock/randomness. Offline-friendly:
tries HuggingFace `datasets` first, falls back to the raw AdvBench CSV on GitHub.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "refusal"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")


def _clean(s: str) -> str:
    return " ".join(str(s).split()).strip()


def load_harmful() -> list[str]:
    """AdvBench harmful behaviours (~520 instructions)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/AdvBench", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        rows = [_clean(x) for x in ds[col]]
        print(f"  harmful: {len(rows)} from walledai/AdvBench")
    except Exception as e:
        print(f"  (HF AdvBench failed: {e}; falling back to CSV)")
        import csv
        import io
        raw = urllib.request.urlopen(ADVBENCH_CSV, timeout=30).read().decode("utf-8")
        rows = [_clean(r["goal"]) for r in csv.DictReader(io.StringIO(raw))]
        print(f"  harmful: {len(rows)} from llm-attacks CSV")
    # dedup, keep order
    seen, out = set(), []
    for r in rows:
        if r and r.lower() not in seen:
            seen.add(r.lower()); out.append(r)
    return out


def load_harmless(min_len=18, max_len=180) -> list[str]:
    """Alpaca instructions with no `input` field (clean single-turn instructions)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
    out, seen = [], set()
    for r in ds:
        if _clean(r.get("input", "")):        # skip context-dependent instructions
            continue
        ins = _clean(r.get("instruction", ""))
        if not (min_len <= len(ins) <= max_len):
            continue
        low = ins.lower()
        if low in seen:
            continue
        seen.add(low); out.append(ins)
    print(f"  harmless: {len(out)} usable Alpaca instructions")
    return out


def write(path: Path, texts: list[str], label: int):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "label": label}) + "\n")
    print(f"  wrote {len(texts):4d} -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=128, help="per-class fit-set size (direction)")
    ap.add_argument("--test", type=int, default=100, help="harmful held-out test size")
    ap.add_argument("--test-benign", type=int, default=80, help="benign held-out test size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("downloading benchmarks ...")
    harmful = load_harmful()
    harmless = load_harmless()

    rng = np.random.default_rng(a.seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)

    need_h = a.fit + a.test
    need_b = a.fit + a.test_benign
    if len(harmful) < need_h or len(harmless) < need_b:
        sys.exit(f"not enough data: harmful {len(harmful)}/{need_h}, harmless {len(harmless)}/{need_b}")

    h_fit, h_test = harmful[:a.fit], harmful[a.fit:need_h]
    b_fit, b_test = harmless[:a.fit], harmless[a.fit:need_b]

    write(OUT / "harmful.jsonl", h_fit, 1)
    write(OUT / "harmless.jsonl", b_fit, 0)
    write(OUT / "harmful_test.jsonl", h_test, 1)
    write(OUT / "harmless_test.jsonl", b_test, 0)

    (OUT / "SOURCES.md").write_text(
        "# Refusal contrast set\n\n"
        "Built by `scripts/build_refusal_data.py` (seed "
        f"{a.seed}). Fit and test splits are disjoint.\n\n"
        "| file | class | source | n |\n|---|---|---|--:|\n"
        f"| harmful.jsonl | harmful (fit) | AdvBench (Zou et al. 2023) | {len(h_fit)} |\n"
        f"| harmless.jsonl | harmless (fit) | Alpaca (Taori et al. 2023) | {len(b_fit)} |\n"
        f"| harmful_test.jsonl | harmful (held-out) | AdvBench | {len(h_test)} |\n"
        f"| harmless_test.jsonl | harmless (held-out) | Alpaca | {len(b_test)} |\n\n"
        "The direction is diff-of-means on the fit split; the behavioural refusal "
        "rate is measured on the held-out split. AdvBench is MIT-licensed; Alpaca "
        "is CC BY-NC 4.0 (research use).\n",
        encoding="utf-8")
    print(f"  wrote SOURCES.md")
    print("\ndone.")


if __name__ == "__main__":
    main()
