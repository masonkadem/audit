"""Step 0: receptive field of the wav2sleep sequence mixer, theoretical and measured.

Usage: python -m audit.rf [--model hf://joncarter/wav2sleep] [--out results/rf.json]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from wav2sleep import load_model
from wav2sleep.models.blocks import DilatedConvBlock

T = 1200  # wav2sleep's fixed 10-hour frame, in 30-s epochs


def theoretical_mixer_rf(model) -> int:
    """One-sided receptive field of the sequence mixer in epochs.

    Each layer is symmetric (padding = (k_eff-1)/2), so it reaches (k-1)/2 * dilation to each side.
    """
    half = 0
    for block in model.sequence_mixer.dilated_convs:
        assert isinstance(block, DilatedConvBlock)
        half += sum((block.kernel_size - 1) // 2 * d for d in block.dilations)
    return half


def theoretical_encoder_rf(model) -> dict[str, float]:
    """Encoder receptive field per signal, in samples and in epochs."""
    out = {}
    for sig in model.valid_signals:
        enc = model.signal_encoders.get_encoder(sig)
        rf, jump = 1, 1
        for block in enc.cnn:
            for conv in (block.conv1, block.conv2, block.conv3):
                c = conv.conv
                rf += (c.kernel_size[0] - 1) * c.dilation[0] * jump
                jump *= c.stride[0]
        out[sig] = {'samples': rf, 'epochs': rf / enc.samples_per_epoch}
    return out


def measured_mixer_influence(model, device: str, n_draws: int = 8) -> np.ndarray:
    """|d out[t0] / d in[t0+lag]| for the sequence mixer alone, averaged over random inputs."""
    mixer = model.sequence_mixer.eval()
    F = model.feature_dim
    t0 = T // 2
    acc = np.zeros(T)
    g = torch.Generator(device='cpu').manual_seed(0)
    for _ in range(n_draws):
        z = torch.randn(1, T, F, generator=g).to(device).requires_grad_(True)
        y = mixer(z)[0, t0]
        y.pow(2).sum().backward()
        acc += z.grad[0].norm(dim=-1).detach().cpu().numpy()
    return acc / n_draws  # index = absolute epoch; lag = index - t0


@torch.no_grad()
def measured_full_model_influence(model, device: str, lags: list[int]) -> dict[int, float]:
    """Max change in output probabilities at epoch t0 when one input epoch at t0+lag is replaced.

    Uses Gaussian-noise inputs at native rates. Detects influence that bypasses the mixer RF,
    e.g. InstanceNorm statistics computed over the whole 10-h sequence.
    """
    model.eval()
    g = torch.Generator(device='cpu').manual_seed(1)
    x = {}
    for sig in model.valid_signals:
        spe = model.signal_encoders.get_encoder(sig).samples_per_epoch
        x[sig] = torch.randn(1, T * spe, generator=g, dtype=torch.float64).to(device)
    t0 = T // 2
    base = torch.softmax(model(x).double(), -1)[0, t0]
    out = {}
    for lag in lags:
        e = t0 + lag
        xp = {}
        for sig, v in x.items():
            spe = model.signal_encoders.get_encoder(sig).samples_per_epoch
            v = v.clone()
            v[0, e * spe : (e + 1) * spe] = torch.randn(spe, generator=g, dtype=torch.float64).to(device) * 3
            xp[sig] = v
        p = torch.softmax(model(xp).double(), -1)[0, t0]
        out[lag] = float((p - base).abs().max())
    return out


def real_night_erf(model, data: str, device: str) -> dict:
    """Pre-registered R: median over unaltered test nights of the 95%-mass radius of
    |d out[t0] / d z[t]| (mixer input z), t0 = middle of the scored night, all signals."""
    from glob import glob

    from .run_model import SUBSETS, altered_input, load_night

    radii = []
    for fp in sorted(glob(os.path.join(data, '*.parquet'))):
        sig, labels, n = load_night(fp)
        if not all(c in sig for c in SUBSETS['all']):
            continue
        x = altered_input(sig, np.arange(n), SUBSETS['all'], device)
        with torch.no_grad():
            z = model.epoch_mixer(model.signal_encoders(x))
        z = z.detach().requires_grad_(True)
        t0 = n // 2
        model.sequence_mixer(z)[0, t0].pow(2).sum().backward()
        infl = z.grad[0].norm(dim=-1).cpu().numpy()
        d = np.abs(np.arange(T) - t0)
        order = np.argsort(d, kind='stable')
        radii.append(int(d[order][np.searchsorted(np.cumsum(infl[order]) / infl.sum(), 0.95)]))
    return {'R_epochs': int(np.median(radii)), 'radius95_per_night': radii}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='hf://joncarter/wav2sleep')
    ap.add_argument('--out', default='results/rf.json')
    ap.add_argument('--data', default=None, help='wav2sleep MESA test folder: also compute R on real nights')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, device=device)
    model.eval()
    if args.data:
        res = real_night_erf(model, args.data, device)
        with open(args.out.replace('.json', '_real.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print('R =', res['R_epochs'], 'epochs')
        return

    half = theoretical_mixer_rf(model)
    enc = theoretical_encoder_rf(model)
    infl = measured_mixer_influence(model, device)
    t0 = T // 2
    nz = np.flatnonzero(infl > 0)
    measured_half = int(max(t0 - nz.min(), nz.max() - t0))
    lags = np.arange(-t0, T - t0)
    w = infl / infl.sum()
    order = np.argsort(np.abs(lags), kind='stable')
    r95 = int(np.abs(lags[order])[np.searchsorted(np.cumsum(w[order]), 0.95)])
    # float64 on CPU so the result is deterministic (GPU kernels are not bit-reproducible)
    full = measured_full_model_influence(model.to('cpu').double(), 'cpu', [0, 1, 5, 30, 100, 124, 125, 200, 300, 590])

    res = {
        'model': args.model,
        'mixer_theoretical_one_sided_epochs': half,
        'mixer_theoretical_total_epochs': 2 * half + 1,
        'mixer_theoretical_one_sided_minutes': half / 2,
        'mixer_measured_nonzero_one_sided_epochs': measured_half,
        'mixer_random_input_95pct_mass_radius_epochs': r95,
        'encoder_rf': enc,
        'full_model_max_dprob_at_lag': full,
        'causal': bool(model.signal_encoders.causal),
        'feature_dim': model.feature_dim,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    np.save(args.out.replace('.json', '_mixer_influence.npy'), infl)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
