"""Secondary robustness check: the same 200 MESA test nights from an alternative extraction
(band-pass filtered, clipped at +/-3 SD, z-scored; PPG and ECG only).

Usage: python -m audit.run_h5 --ppg mesa_ppg_with_labels.h5 --ecg mesa_real_ecg.h5 --index mesa_subject_index.h5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
from wav2sleep import load_model
from wav2sleep.data.nsrr import get_split

from .run_model import T, run_arrays


def load_h5_night(ppg, ecg, idx: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    order = np.sort(idx)
    if order[-1] - order[0] + 1 != len(order):
        raise ValueError('window_indices of a night are not contiguous rows')
    sig = {'PPG': ppg['ppg'][order[0] : order[-1] + 1].astype(np.float32),
           'ECG': ecg['ecg'][order[0] : order[-1] + 1].astype(np.float32)}
    labels = ppg['labels'][order[0] : order[-1] + 1].astype(int)[:T]
    if not (ecg['labels'][order[0] : order[-1] + 1][:T] == labels).all():
        raise ValueError('PPG and ECG files disagree on labels')
    scored = np.flatnonzero(labels >= 0)
    n = int(scored[-1]) + 1
    for v in sig.values():
        live = np.flatnonzero(np.abs(v).max(1) > 0)
        n = max(n, int(live[-1]) + 1) if len(live) else n
    labels = np.r_[labels, np.full(max(0, n - len(labels)), -1)]
    return sig, labels, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppg', required=True)
    ap.add_argument('--ecg', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--model', default='hf://joncarter/wav2sleep')
    ap.add_argument('--out', default='results/h5')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, device=device).eval()
    os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.ppg, 'r') as ppg, h5py.File(args.ecg, 'r') as ecg, h5py.File(args.index, 'r') as ix:
        for sid in tqdm(get_split('mesa', 'test')):
            key = sid.replace('mesa-sleep-', '')
            o = os.path.join(args.out, f'{sid}.npz')
            if key not in ix['subjects'] or os.path.exists(o):
                continue
            idx = ix['subjects'][key]['window_indices'][:]
            res = run_arrays(model, *load_h5_night(ppg, ecg, idx), device)
            np.savez_compressed(o, **res)


if __name__ == '__main__':
    main()
