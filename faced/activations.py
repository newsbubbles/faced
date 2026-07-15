"""Collect residual-stream activations for contrastive prompt sets.

For each prompt we chat-template it as a user turn, greedily generate a short
response, and capture the post-layer residual stream (all layers) at:
  * ``last_prompt`` — the last prompt token (prefill)
  * ``mean_gen``    — mean over the first ``gen_tokens`` generated tokens
``mean_gen`` is the primary training signal because it matches how the live panel
reads (on generated tokens).

Saved to data/activations/<model_key>/<emotion>.safetensors with tensors
{last_prompt, mean_gen, labels} of shape [N, n_layers, d_model] / [N], and
safetensors metadata carrying per-row family/style as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file, safe_open

from .backends import ModelBundle, REPO_ROOT

ACT_DIR = REPO_ROOT / "data" / "activations"
PROMPT_DIR = REPO_ROOT / "data" / "prompts"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.inference_mode()
def capture_prompt(bundle: ModelBundle, text: str, gen_tokens: int = 16):
    """Return (last_prompt[L,D], mean_gen[L,D]) post-layer residuals, all layers.

    Accumulates on-GPU and transfers to CPU once per prompt (not per token).
    """
    ids = bundle.encode_chat(bundle.user(text))
    cur, past = ids, None
    last_prompt = None
    gen_sum = None
    n_gen = 0
    for step in range(gen_tokens + 1):
        o = bundle.model(input_ids=cur, past_key_values=past, use_cache=True,
                         output_hidden_states=True)
        past = o.past_key_values
        stack = torch.stack([h[0, -1] for h in o.hidden_states[1:]])  # [L, D], on device
        if step == 0:
            last_prompt = stack
        else:
            gen_sum = stack if gen_sum is None else gen_sum + stack
            n_gen += 1
        nxt = int(o.logits[:, -1, :].argmax(-1))
        cur = torch.tensor([[nxt]], device=bundle.device)
        if nxt in bundle.eos_ids and n_gen >= 1:
            break
    mean_gen = (gen_sum / n_gen) if n_gen else last_prompt
    return last_prompt.float().cpu(), mean_gen.float().cpu()


def collect_emotion(bundle: ModelBundle, emotion: str, gen_tokens: int = 16,
                    verbose: bool = True) -> Path:
    rows = read_jsonl(PROMPT_DIR / f"{emotion}.jsonl")
    last_list, mean_list, labels, fams, styles = [], [], [], [], []
    for i, r in enumerate(rows):
        lp, mg = capture_prompt(bundle, r["text"], gen_tokens=gen_tokens)
        last_list.append(lp)
        mean_list.append(mg)
        labels.append(float(r["label"]))
        fams.append(r.get("family", str(i)))
        styles.append(r.get("style", ""))
        if verbose and (i + 1) % 20 == 0:
            print(f"    {emotion}: {i+1}/{len(rows)}")
    out_dir = ACT_DIR / bundle.key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{emotion}.safetensors"
    tensors = {
        "last_prompt": torch.stack(last_list),
        "mean_gen": torch.stack(mean_list),
        "labels": torch.tensor(labels),
    }
    meta = {"families": json.dumps(fams), "styles": json.dumps(styles),
            "emotion": emotion, "model": bundle.key, "n_layers": str(bundle.n_layers),
            "d_model": str(bundle.d_model)}
    save_file(tensors, str(out), metadata=meta)
    return out


def load_activations(model_key: str, emotion: str) -> dict:
    path = ACT_DIR / model_key / f"{emotion}.safetensors"
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return {
        "last_prompt": tensors["last_prompt"],
        "mean_gen": tensors["mean_gen"],
        "labels": tensors["labels"],
        "families": json.loads(meta.get("families", "[]")),
        "styles": json.loads(meta.get("styles", "[]")),
        "meta": meta,
    }


def collect_all(bundle: ModelBundle, emotions: list[str], gen_tokens: int = 16) -> list[Path]:
    paths = []
    for e in emotions:
        print(f"  collecting '{e}' ...")
        paths.append(collect_emotion(bundle, e, gen_tokens=gen_tokens))
    return paths
