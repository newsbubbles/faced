"""M0 smoke test.

Loads the local dev model, verifies the model-agnostic layer resolution, checks
that CaptureHook agrees with output_hidden_states, and runs a short manual greedy
loop printing the per-step residual-stream stack. Also reports VRAM.

    python scripts/smoke_m0.py [model_key]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from faced.backends import load
from faced.hooks import CaptureHook

PROMPT = "Can you review the contract I attached? Let me know if the payment terms look fair."


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    b = load(key)
    print(f"loaded '{b.key}' ({b.hf_id}) in {time.time()-t0:.1f}s")
    print(f"  device={b.device} dtype={b.dtype} | layers@'{b.layer_path}' "
          f"n_layers={b.n_layers} d_model={b.d_model} ple={b.ple}")

    ids = b.encode_chat(b.user(PROMPT))
    print(f"  prompt tokens: {ids.shape[1]}")

    # 1) CaptureHook must agree with output_hidden_states on middle layers.
    mid = b.n_layers // 2
    with CaptureHook(b.layers, [0, mid, b.n_layers - 1]) as cap:
        out = b.forward(ids)  # use_cache=False, output_hidden_states=True
    hs = out.hidden_states
    print(f"  hidden_states: len={len(hs)} (n_layers+1), each {tuple(hs[1].shape)}")
    for i in (0, mid):
        agree = torch.allclose(cap.captured[i].float(), hs[i + 1].float(), atol=1e-2)
        print(f"  hook==hidden_states[{i+1}] at layer {i}: {agree}")
    # last layer differs from hidden_states (post-final-norm) — report, don't assert
    print(f"  layer {b.n_layers-1}: hook norm={cap.captured[b.n_layers-1].float().norm():.1f} "
          f"vs hidden_states[-1] norm={hs[-1].float().norm():.1f} (post-final-norm, expected to differ)")

    # 2) Short manual greedy loop with per-step residual stack.
    print("\n  greedy 20-token sample:")
    cur, past, gen = ids, None, []
    with torch.inference_mode():
        for step in range(20):
            o = b.model(input_ids=cur, past_key_values=past, use_cache=True,
                        output_hidden_states=True)
            past = o.past_key_values
            nxt = int(o.logits[:, -1, :].argmax(-1))
            stack = torch.stack([h[0, -1] for h in o.hidden_states[1:]])  # [n_layers, d_model]
            if step == 0:
                print(f"    per-step residual stack shape: {tuple(stack.shape)}")
            gen.append(nxt)
            cur = torch.tensor([[nxt]], device=b.device)
            if nxt in b.eos_ids:
                break
    text = b.tokenizer.decode(gen, skip_special_tokens=True)
    print(f"    -> {text!r}")

    if b.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  peak VRAM: {peak:.2f} GB")
    print("\nM0 OK")


if __name__ == "__main__":
    main()
