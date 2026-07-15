"""Forward hooks over a model's decoder layers.

Two primitives the whole project hinges on:

* ``CaptureHook``  — records the post-layer residual stream at chosen layers.
* ``SteerHook``    — adds / suppresses / ablates a direction at a chosen layer
                     during generation (used in M4).

Layer outputs may be a bare tensor (transformers 5.x) or a ``tuple`` whose first
element is the hidden state (older transformers). Both are handled defensively so
the code is correct either way.
"""
from __future__ import annotations

import torch


def unpack_layer_output(output):
    """Return the hidden-state tensor from a decoder layer's forward output."""
    return output[0] if isinstance(output, tuple) else output


def repack_layer_output(output, new_hidden):
    """Put ``new_hidden`` back into the same container shape the layer returned."""
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class CaptureHook:
    """Capture residual-stream activations at ``indices`` during a forward pass.

    Usage::

        with CaptureHook(bundle.layers, [0, 12, 25]) as cap:
            bundle.forward(input_ids)
        cap.captured[12]  # -> tensor [batch, seq, d_model]
    """

    def __init__(self, layers, indices):
        self.layers = layers
        self.indices = list(indices)
        self.captured: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make(self, idx):
        def hook(module, inputs, output):
            self.captured[idx] = unpack_layer_output(output).detach()
        return hook

    def __enter__(self):
        self.captured.clear()
        for i in self.indices:
            self._handles.append(self.layers[i].register_forward_hook(self._make(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class SteerHook:
    """Add / suppress / ablate a direction at one layer during generation.

    ``vector`` is a direction in activation space. For ``add`` pass the RAW
    difference-of-means (mu_pos - mu_neg): moving by 1x that vector shifts the
    residual by one class-separation, so ``alpha`` is an interpretable coefficient
    (~1-3 is a strong-but-coherent steer). For ``ablate`` pass the direction you
    READ (the readout direction), so the meter provably drops.
    modes:
      * ``add``     : h += alpha * v_raw            (v_raw = mu_pos - mu_neg)
      * ``ablate``  : h -= (h . unit(v)) unit(v)    (project the direction out)
    ``positions='last'`` steers only the current (newest) token — the right choice
    during cached decode, where the hook sees shape [batch, 1, d_model] anyway.
    """

    def __init__(self, layers, index, vector, alpha=1.0, mode="add", positions="last"):
        self.layers = layers
        self.index = index
        self.alpha = float(alpha)
        self.mode = mode
        self.positions = positions
        v = vector.detach().float()
        self.raw = v
        self.unit = v / (v.norm() + 1e-8)
        self._handle = None

    def _apply(self, h):
        if self.positions == "last":
            sl = h[:, -1:, :]
        else:
            sl = h
        if self.mode == "add":
            sl = sl + self.alpha * self.raw.to(dtype=h.dtype, device=h.device)
        elif self.mode == "add_rms":
            # inject a consistent FRACTION of the local residual magnitude, so a
            # single alpha is comparable across emotions/layers of different scale.
            unit = self.unit.to(dtype=h.dtype, device=h.device)
            rms = sl.pow(2).mean(dim=-1, keepdim=True).sqrt()
            sl = sl + self.alpha * rms * unit
        elif self.mode == "ablate":
            unit = self.unit.to(dtype=h.dtype, device=h.device)
            proj = (sl * unit).sum(dim=-1, keepdim=True)
            sl = sl - proj * unit
        else:
            raise ValueError(f"Unknown steer mode '{self.mode}'")
        if self.positions == "last":
            h = h.clone()
            h[:, -1:, :] = sl
            return h
        return sl

    def _hook(self, module, inputs, output):
        h = unpack_layer_output(output)
        return repack_layer_output(output, self._apply(h))

    def __enter__(self):
        self._handle = self.layers[self.index].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()
            self._handle = None
        return False
