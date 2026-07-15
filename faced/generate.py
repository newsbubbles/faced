"""Manual streaming generation with per-token emotion telemetry.

Owns the decode loop (rather than model.generate) so we can read the residual
stream and optionally steer at every step. Yields one event per generated token:

    {"i": int, "t": str, "token_id": int, "meters": {emotion: {...}}}

The meters at step i reflect the model's internal state *as it produced* token i
(step 0 is the state at the end of the prompt — the reaction to the user).
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from .backends import ModelBundle
from .hooks import CaptureHook
from .readout import EmotionReader


def _sample(logits, temperature=0.0, top_p=1.0):
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits.float() / temperature, dim=-1).squeeze(0)
    if top_p < 1.0:
        srt, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(srt, 0)
        keep = cum <= top_p
        keep[0] = True
        srt = srt * keep
        srt = srt / srt.sum()
        choice = torch.multinomial(srt, 1)
        return int(idx[choice])
    return int(torch.multinomial(probs, 1))


@torch.inference_mode()
def stream(bundle: ModelBundle, prompt, reader: EmotionReader,
           max_tokens: int = 120, temperature: float = 0.0, top_p: float = 1.0,
           steer=None):
    """prompt: str (user turn) or list[messages]. steer: a SteerHook or None."""
    reader.reset()
    messages = bundle.user(prompt) if isinstance(prompt, str) else prompt
    ids = bundle.encode_chat(messages)
    cur, past = ids, None
    ctx = steer if steer is not None else nullcontext()
    # steer context is entered FIRST so the CaptureHook (entered second) fires
    # after it and reads the post-steer residual. We read via CaptureHook rather
    # than output_hidden_states because transformers 5.x snapshots hidden_states
    # before forward-hook edits (so steering wouldn't show up in the meters).
    with ctx:
        with CaptureHook(bundle.layers, reader.layers_used) as cap:
            for i in range(max_tokens):
                o = bundle.model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                meters = reader.read_captured(cap.captured)
                nxt = _sample(o.logits[:, -1, :], temperature, top_p)
                piece = bundle.decode(nxt)
                yield {"i": i, "t": piece, "token_id": nxt, "meters": meters}
                if nxt in bundle.eos_ids:
                    break
                cur = torch.tensor([[nxt]], device=bundle.device)
