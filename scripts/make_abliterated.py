"""Produce an abliterated (refusal-direction-removed) twin of a model.

    python scripts/make_abliterated.py [model_key] [--out gemma-3-1b-abl]

Computes the refusal direction (diff-of-means harmful vs harmless), measures the
refusal rate, orthogonalizes the residual-writing weights against it, measures the
refusal rate again, and saves the abliterated model + a manifest. Register the
output path in config/models.yaml to fit emotion directions on it.
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

import torch
from safetensors.torch import save_file

from faced.backends import load, REPO_ROOT
from faced.abliterate import compute_refusal_direction, abliterate, refusal_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--out", default=None, help="output model key/dirname")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--register", action="store_true",
                    help="append the entry to config/models.local.yaml (turnkey)")
    ap.add_argument("--random", action="store_true",
                    help="CONTROL: abliterate along a random unit direction, not the refusal one")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    b = load(a.model)
    out_name = a.out or f"{b.key}-abl"
    out_dir = REPO_ROOT / "models" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.random:
        # matched control: same weight-orthogonalization procedure, random direction.
        layer = a.layer if a.layer is not None else max(1, round(0.6 * b.n_layers))
        g = torch.Generator().manual_seed(a.seed)
        r = torch.randn(b.d_model, generator=g)
        r = r / r.norm()
        auc = float("nan")
        print(f"model={b.key}  RANDOM-direction control (seed={a.seed}) @ layer {layer}")
    else:
        print(f"model={b.key}  computing refusal direction ...")
        r, layer, auc = compute_refusal_direction(b, layer=a.layer)
        print(f"  refusal direction @ layer {layer}  (harmful/harmless AUC={auc:.3f})")

    print("  measuring refusal rate (stock) ...")
    rr_before = refusal_rate(b)
    print(f"    stock refusal rate = {rr_before:.2f}")

    print("  abliterating (orthogonalizing residual-writing weights) ...")
    touched = abliterate(b, r)
    print(f"    touched: embed={touched['embed']} o_proj={touched['o_proj']} down_proj={touched['down_proj']}")

    print("  measuring refusal rate (abliterated) ...")
    rr_after = refusal_rate(b)
    print(f"    abliterated refusal rate = {rr_after:.2f}")

    print(f"  saving -> {out_dir}")
    b.model.save_pretrained(str(out_dir))
    b.tokenizer.save_pretrained(str(out_dir))

    # persist the refusal direction (NOT for the random control — it must not clobber
    # the real refusal direction the experiment reads for the refusal-overlap column)
    if not a.random:
        save_file({"refusal_dir": r.float(), "refusal_layer": torch.tensor([layer])},
                  str(REPO_ROOT / "artifacts" / f"refusal_dir.{b.key}.safetensors"))
    manifest = {
        "source_model": b.key, "source_hf_id": b.hf_id, "out": out_name,
        "kind": "random" if a.random else "refusal", "seed": a.seed if a.random else None,
        "refusal_layer": int(layer), "refusal_auc": float(auc),
        "refusal_rate_stock": float(rr_before), "refusal_rate_abliterated": float(rr_after),
        "touched": touched, "d_model": b.d_model, "n_layers": b.n_layers,
    }
    with open(out_dir / "abliteration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  refusal rate  {rr_before:.2f} -> {rr_after:.2f}   (Δ = {rr_before-rr_after:+.2f})")
    print(f"  manifest -> {out_dir / 'abliteration_manifest.json'}")

    entry = {out_name: {"hf_id": out_dir.as_posix(), "arch": b.meta.get("arch", "causal_lm"),
                        "layer_path": b.layer_path, "dtype": b.meta.get("dtype", "float16"),
                        "ple": b.ple, "chat": b.chat,
                        "notes": f"Abliterated twin of {b.key} (refusal {rr_before:.2f}->{rr_after:.2f})"}}
    if a.register:
        import yaml
        lp = REPO_ROOT / "config" / "models.local.yaml"
        cur = {}
        if lp.exists():
            cur = yaml.safe_load(open(lp, encoding="utf-8")) or {}
        cur.setdefault("models", {}).update(entry)
        yaml.safe_dump(cur, open(lp, "w", encoding="utf-8"), sort_keys=False)
        print(f"  registered '{out_name}' in config/models.local.yaml")
    else:
        import yaml
        print("\n  add to config/models.yaml (or re-run with --register):")
        print("    " + yaml.safe_dump(entry, sort_keys=False).replace("\n", "\n    ").rstrip())


if __name__ == "__main__":
    main()
