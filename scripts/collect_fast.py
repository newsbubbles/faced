"""Faster semantic activation collection for the abliteration experiment.

Uses the generate-based collector but with a short response (mean over the first
`--gen` generated tokens), so directions are the *semantic* emotion concept (the
model's response state) rather than a lexical property of the prompt tokens — while
being much cheaper than the full 16-token collection.

    python scripts/collect_fast.py <model_key> [<model_key> ...] [--gen 4]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from safetensors.torch import save_file

from faced.backends import load
from faced.activations import read_jsonl, capture_prompt, PROMPT_DIR, ACT_DIR
from faced.config import axis_names

GEN = int(sys.argv[sys.argv.index("--gen") + 1]) if "--gen" in sys.argv else 4


def collect(model_key):
    b = load(model_key)
    out_dir = ACT_DIR / b.key
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in axis_names():
        rows = read_jsonl(PROMPT_DIR / f"{e}.jsonl")
        lasts, means = [], []
        for r in rows:
            lp, mg = capture_prompt(b, r["text"], gen_tokens=GEN)
            lasts.append(lp); means.append(mg)
        Tl, Tm = torch.stack(lasts), torch.stack(means)
        save_file(
            {"last_prompt": Tl, "mean_gen": Tm,
             "labels": torch.tensor([float(r["label"]) for r in rows])},
            str(out_dir / f"{e}.safetensors"),
            metadata={"families": json.dumps([r.get("family", str(i)) for i, r in enumerate(rows)]),
                      "styles": json.dumps([r.get("style", "") for r in rows]),
                      "emotion": e, "model": b.key, "signal": f"mean_gen{GEN}"},
        )
        print(f"  {b.key}/{e}: {Tm.shape[0]} prompts, finite={bool(torch.isfinite(Tm).all())}")
    # free VRAM before the next model
    from faced import backends
    del b
    backends.load.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = sys.argv[1:]
    if "--gen" in args:
        i = args.index("--gen"); del args[i:i + 2]
    for key in args:
        print(f"collecting (mean over {GEN} generated tokens) {key} ...")
        collect(key)


if __name__ == "__main__":
    main()
