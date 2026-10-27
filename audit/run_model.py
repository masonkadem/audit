"""Run wav2sleep on unaltered and altered MESA test nights; save per-epoch probabilities.

Reads wav2sleep's own preprocessed MESA test partition ($WAV2SLEEP_DATA/mesa/test/*.parquet,
produced by preprocessing/1_ingest.py + 2a_create_splits_paper.py) and refuses any night not
listed in wav2sleep's MESA test split.

Usage: python -m audit.run_model --data $WAV2SLEEP_DATA/mesa/test --out results/w2s
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from wav2sleep import load_model
from wav2sleep.data.nsrr import get_split
from wav2sleep.settings import COLS_TO_SAMPLES_PER_EPOCH, INTEGER_LABEL_MAPS, LABEL

from . import interventions as iv

T = 1200
SUBSETS = {'ppg': ['PPG'], 'ecg': ['ECG'], 'all': ['ECG', 'PPG', 'ABD', 'THX']}


def load_night(fp: str) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Signals reshaped to [epochs, samples_per_epoch], 4-class labels (-1 unscored), valid length n."""
    df = pd.read_parquet(fp)
    labels = df[LABEL].dropna().map(INTEGER_LABEL_MAPS[4]).fillna(-1).to_numpy().astype(int)[:T]
    scored = np.flatnonzero(labels >= 0)
    n = int(scored[-1]) + 1
    sig = {}
    for col in SUBSETS['all']:
        if col in df.columns:
            spe = COLS_TO_SAMPLES_PER_EPOCH[col]
            x = df[col].dropna().to_numpy().astype(np.float32)[: T * spe]
            sig[col] = x.reshape(-1, spe)
            # Night ends where the ingest's zero fill begins, not at the last scored epoch:
            # unscored epochs with real signal move with content like any other epoch.
            live = np.flatnonzero(np.abs(sig[col]).max(1) > 0)
            if len(live):
                n = max(n, int(live[-1]) + 1)
    labels = np.r_[labels, np.full(max(0, n - len(labels)), -1)]
    return sig, labels, n


def altered_input(sig: dict[str, np.ndarray], src: np.ndarray, cols: list[str], device: str):
    """Rebuild the 10-h frame from moved epochs, zero-fill tail, z-score exactly as ParquetDataset does."""
    x = {}
    for col in cols:
        if col not in sig:
            spe = COLS_TO_SAMPLES_PER_EPOCH[col]
            x[col] = torch.full((1, T * spe), float('-inf'), device=device)
            continue
        e = sig[col][src[:T]]
        frame = np.zeros((T, e.shape[1]), np.float32)
        frame[: len(e)] = e
        v = torch.from_numpy(frame.reshape(1, -1)).to(device)
        std = v.std()
        x[col] = (v - v.mean()) / (std if std > 1e-6 else 1e-6)
    return x


@torch.inference_mode()
def run_night(model, fp: str, device: str) -> dict[str, np.ndarray]:
    return run_arrays(model, *load_night(fp), device)


@torch.inference_mode()
def run_arrays(model, sig: dict[str, np.ndarray], labels: np.ndarray, n: int, device: str) -> dict[str, np.ndarray]:
    lab = labels[:n]
    plans = iv.build(lab)
    out: dict[str, np.ndarray] = {'labels': lab, 'n': np.array(n)}
    for name, (src, null) in plans.items():
        src_t = src[:T]  # content pushed past 10 h is truncated, as in the original pipeline
        pos = iv.content_position(src_t, n)
        out[f'{name}/pos'] = pos
        out[f'{name}/dist'] = iv.seam_distance(src_t)
        out[f'{name}/src'] = src_t
        for subset, cols in SUBSETS.items():
            if not all(c in sig for c in cols):
                continue
            logits = model(altered_input(sig, src_t, cols, device))[0, : len(src_t)]
            out[f'{name}/{subset}/prob'] = torch.softmax(logits.float(), -1).cpu().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='wav2sleep MESA test folder')
    ap.add_argument('--model', default='hf://joncarter/wav2sleep')
    ap.add_argument('--out', default='results/w2s')
    args = ap.parse_args()
    test_ids = set(get_split('mesa', 'test'))
    fps = sorted(glob(os.path.join(args.data, '*.parquet')))
    ids = [os.path.basename(f).replace('.parquet', '').replace('.issues', '') for f in fps]
    bad = [i for i in ids if i not in test_ids]
    if bad:
        raise SystemExit(f'{len(bad)} files are not in the wav2sleep MESA test split, e.g. {bad[:3]}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, device=device).eval()
    os.makedirs(args.out, exist_ok=True)
    for fp, sid in tqdm(list(zip(fps, ids))):
        o = os.path.join(args.out, f'{sid}.npz')
        if not os.path.exists(o):
            np.savez_compressed(o, **run_night(model, fp, device))


if __name__ == '__main__':
    main()
