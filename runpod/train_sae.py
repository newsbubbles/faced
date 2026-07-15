"""Train a minimal top-k sparse autoencoder on harvested activations, then align
discovered features to the named emotion directions.

Dependency-free (no sae_lens needed). This is the M5b research bridge: it checks
whether the cheap contrastive directions correspond to features an unsupervised
SAE discovers, and surfaces candidate axes we didn't name.

    python runpod/train_sae.py [model_key] --layer 12 --features 4096 --k 32 --steps 2000

Reads artifacts/sae/<model>_L<layer>_acts.safetensors; writes the SAE + an
alignment report artifacts/sae/<model>_L<layer>_alignment.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from faced.backends import load as load_model, REPO_ROOT
from faced.config import axis_names


class TopKSAE(nn.Module):
    def __init__(self, d, n_features, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.enc = nn.Linear(d, n_features)
        self.dec = nn.Linear(n_features, d, bias=False)
        with torch.no_grad():  # tie init: decoder = encoder^T, unit-norm columns
            self.dec.weight.copy_(F.normalize(self.enc.weight.t(), dim=0))

    def encode(self, x):
        z = F.relu(self.enc(x - self.b_pre))
        topv, topi = z.topk(self.k, dim=-1)
        out = torch.zeros_like(z).scatter_(-1, topi, topv)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.b_pre, z


def train(acts, n_features, k, steps, bs, lr, device):
    d = acts.shape[1]
    sae = TopKSAE(d, n_features, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    acts = acts.to(device)
    n = acts.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        x = acts[idx]
        xhat, z = sae(x)
        loss = F.mse_loss(xhat, x)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():  # keep decoder columns unit-norm (standard SAE trick)
            sae.dec.weight.copy_(F.normalize(sae.dec.weight, dim=0))
        opt.step()
        if (step + 1) % max(1, steps // 5) == 0:
            var = acts.var().item()
            print(f"  step {step+1:5d}  mse={loss.item():.4f}  frac_var_unexplained={loss.item()/var:.3f}")
    return sae


def align(sae, model_key, layer, device):
    """For each named emotion direction, find the SAE feature with max |cosine|."""
    vecs = load_file(str(REPO_ROOT / "artifacts" / f"directions.{model_key}.safetensors"))
    with open(REPO_ROOT / "config" / f"calibration.{model_key}.json", encoding="utf-8") as f:
        calib = json.load(f)["axes"]
    D = F.normalize(sae.dec.weight.detach(), dim=0).to(device)  # [d, n_features]
    report = {}
    for e in axis_names():
        if f"{e}.v_steer" not in vecs:
            continue
        # SAE decoder features live in raw activation space, so compare against
        # the diff-of-means direction (also activation-space), not the whitened readout.
        v = F.normalize(vecs[f"{e}.v_steer"].to(device), dim=0)
        cos = (v @ D).abs()                    # [n_features]
        top = torch.topk(cos, 3)
        report[e] = {
            "axis_layer": calib[e]["layer"],
            "harvest_layer": layer,
            "best_feature": int(top.indices[0]),
            "best_cosine": float(top.values[0]),
            "top3_features": [int(i) for i in top.indices],
            "top3_cosine": [round(float(c), 3) for c in top.values],
            "aligned": bool(top.values[0] > 0.5 and calib[e]["layer"] == layer),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--features", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    b = load_model(a.model)
    layer = a.layer if a.layer is not None else b.n_layers // 2
    device = b.device
    acts_path = REPO_ROOT / "artifacts" / "sae" / f"{b.key}_L{layer}_acts.safetensors"
    acts = load_file(str(acts_path))["acts"]
    print(f"training top-{a.k} SAE: {acts.shape[0]} acts, d={acts.shape[1]}, "
          f"{a.features} features, {a.steps} steps")
    sae = train(acts, a.features, a.k, a.steps, a.bs, a.lr, device)

    out_dir = REPO_ROOT / "artifacts" / "sae"
    save_file({"enc_w": sae.enc.weight.detach().cpu(), "enc_b": sae.enc.bias.detach().cpu(),
               "dec_w": sae.dec.weight.detach().cpu(), "b_pre": sae.b_pre.detach().cpu()},
              str(out_dir / f"{b.key}_L{layer}_sae.safetensors"),
              metadata={"features": str(a.features), "k": str(a.k)})

    report = align(sae, b.key, layer, device)
    with open(out_dir / f"{b.key}_L{layer}_alignment.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n  feature<->axis alignment (axes fit at this layer):")
    for e, r in report.items():
        mark = "OK " if r["aligned"] else "   "
        note = "" if r["axis_layer"] == layer else f"(axis@L{r['axis_layer']})"
        print(f"  [{mark}] {e:12s} feat#{r['best_feature']:5d} cos={r['best_cosine']:.3f} {note}")
    n_al = sum(r["aligned"] for r in report.values())
    print(f"\n  {n_al} axes aligned (cos>0.5) at layer {layer}")


if __name__ == "__main__":
    main()
