"""Model-agnostic backend adapter.

Loads any registered model through HF transformers and exposes a uniform surface
to the rest of ``faced``:

* ``bundle.layers``    — the decoder-layer ``nn.ModuleList`` (the residual stream)
* ``bundle.n_layers`` / ``bundle.d_model``
* ``bundle.encode_chat(messages)`` — chat-templated ``input_ids`` on device
* ``bundle.forward(input_ids, ...)`` — a forward pass with hidden states

Per-model quirks (which Auto class, where the layers live, dtype, PLE flag) come
from ``config/models.yaml`` — never from code here.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def registry(path: Path | None = None) -> dict:
    """Load config/models.yaml, merging generated entries from models.local.yaml."""
    path = path or (CONFIG_DIR / "models.yaml")
    with open(path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    local = CONFIG_DIR / "models.local.yaml"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            lreg = yaml.safe_load(f) or {}
        reg.setdefault("models", {}).update(lreg.get("models", {}))
    return reg


def _auto_class(arch: str):
    from transformers import AutoModelForCausalLM
    if arch in ("causal_lm", "causal", "clm"):
        return AutoModelForCausalLM
    if arch in ("image_text_to_text", "conditional_generation", "vlm"):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    raise ValueError(f"Unknown arch '{arch}' in models.yaml")


def _resolve_path(root: Any, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _autodetect_layers(model) -> tuple[str, nn.ModuleList] | None:
    """Fallback: the largest ModuleList whose children look like decoder layers."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            child = mod[0]
            if any(hasattr(child, a) for a in ("self_attn", "mlp", "feed_forward")):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    return best


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelBundle:
    key: str
    hf_id: str
    model: Any
    tokenizer: Any
    layers: nn.ModuleList
    layer_path: str
    d_model: int
    n_layers: int
    device: str
    dtype: torch.dtype
    ple: bool = False
    chat: bool = True
    meta: dict = field(default_factory=dict)

    # -- input construction -------------------------------------------------
    def user(self, text: str) -> list[dict]:
        """A single-turn user message list."""
        return [{"role": "user", "content": text}]

    def encode_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> torch.Tensor:
        """Chat-template ``messages`` into ``input_ids`` on device (shape [1, T])."""
        if self.chat and getattr(self.tokenizer, "chat_template", None):
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                return_tensors="pt", return_dict=True,
            )
            ids = enc["input_ids"]
        else:  # raw fallback: concatenate contents
            text = "\n".join(m["content"] for m in messages)
            ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)

    # -- forward ------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True, **kw):
        return self.model(input_ids=input_ids, output_hidden_states=output_hidden_states,
                          use_cache=False, **kw)

    @property
    def eos_ids(self) -> list[int]:
        ids = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        # gemma uses <end_of_turn> to end assistant turns
        for tok in ("<end_of_turn>", "<|im_end|>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid not in ids:
                ids.append(int(tid))
        return ids


@functools.lru_cache(maxsize=3)
def load(key: str | None = None, device: str | None = None) -> ModelBundle:
    """Load a registered model. Cached, so repeated calls reuse the same bundle."""
    from transformers import AutoTokenizer

    reg = registry()
    key = key or reg["default"]
    if key not in reg["models"]:
        raise KeyError(f"Model '{key}' not in models.yaml (have: {list(reg['models'])})")
    spec = reg["models"][key]

    device = device or _pick_device()
    dtype = _DTYPES[str(spec.get("dtype", "float16")).lower()]
    auto = _auto_class(spec["arch"])

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    try:
        model = auto.from_pretrained(spec["hf_id"], dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:  # very old transformers used torch_dtype
        model = auto.from_pretrained(spec["hf_id"], torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()

    # Resolve decoder layers (explicit path, then auto-detect fallback).
    layer_path = spec.get("layer_path", "model.layers")
    try:
        layers = _resolve_path(model, layer_path)
        assert isinstance(layers, nn.ModuleList) and len(layers) > 0
    except (AttributeError, AssertionError):
        found = _autodetect_layers(model)
        if not found:
            raise RuntimeError(
                f"Could not resolve decoder layers for '{key}' at '{layer_path}' "
                "and auto-detect failed. Inspect model.named_modules()."
            )
        layer_path, layers = found

    cfg = model.config
    d_model = getattr(cfg, "hidden_size", None)
    if d_model is None and hasattr(cfg, "text_config"):
        d_model = getattr(cfg.text_config, "hidden_size", None)

    return ModelBundle(
        key=key, hf_id=spec["hf_id"], model=model, tokenizer=tok,
        layers=layers, layer_path=layer_path, d_model=int(d_model),
        n_layers=len(layers), device=device, dtype=dtype,
        ple=bool(spec.get("ple", False)), chat=bool(spec.get("chat", True)),
        meta=spec,
    )
