"""Harvest residual-stream activations at one layer over a corpus (for SAE training).

Model-agnostic: defaults to the local dev model so the SAE pipeline can be
smoke-tested on the GTX 1080 before scaling to a Gemma-4 on RunPod.

    python runpod/harvest_activations.py [model_key] --layer 12 --max-tokens 200000

Writes artifacts/sae/<model>_L<layer>_acts.safetensors  ({acts: [N, d_model]}).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from safetensors.torch import save_file

from faced.backends import load, REPO_ROOT
from faced.activations import read_jsonl

PROMPT_DIR = REPO_ROOT / "data" / "prompts"


def corpus_texts():
    texts = []
    for p in sorted(PROMPT_DIR.glob("*.jsonl")):
        texts += [r["text"] for r in read_jsonl(p)]
    ref = REPO_ROOT / "data" / "reference_corpus.jsonl"
    if ref.exists():
        texts += [r["text"] for r in read_jsonl(ref)]
    return texts


@torch.inference_mode()
def harvest(model_key, layer, max_tokens):
    b = load(model_key)
    layer = layer if layer is not None else b.n_layers // 2
    buf, total = [], 0
    for text in corpus_texts():
        ids = b.encode_chat(b.user(text))
        out = b.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer + 1][0].float().cpu()  # [seq, d_model]
        buf.append(h)
        total += h.shape[0]
        if total >= max_tokens:
            break
    acts = torch.cat(buf, 0)[:max_tokens]
    out_dir = REPO_ROOT / "artifacts" / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{b.key}_L{layer}_acts.safetensors"
    save_file({"acts": acts}, str(out), metadata={"model": b.key, "layer": str(layer)})
    print(f"  harvested {acts.shape[0]} token-activations d={acts.shape[1]} at layer {layer}")
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=200000)
    a = ap.parse_args()
    harvest(a.model, a.layer, a.max_tokens)


if __name__ == "__main__":
    main()
