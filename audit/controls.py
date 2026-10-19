"""Step 6 controls: windowed-feature LightGBM with and without an explicit time-since-onset feature.

Features are the existing per-epoch windowed HRV / RRV / EDR features (windows <= 270 s) from the
MESA actigraphy-overlap pipeline. Interventions permute feature rows with the same index maps as
wav2sleep; windows that would straddle a seam are not recomputed, so epochs within
SEAM_RADIUS of a seam are excluded downstream.

Training uses wav2sleep's MESA train split (val split for early stopping); evaluation uses only
nights in wav2sleep's MESA test split.

Usage: python -m audit.controls --features F.parquet --labels L.parquet --out results/controls
"""

from __future__ import annotations

import argparse
import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from wav2sleep.data.nsrr import get_split

from . import interventions as iv
from .prior import sleep_onset

SEAM_RADIUS = 5  # epochs; longest feature window is 270 s = +/-4.5 epochs
FEATURE_RE = re.compile(r'^(_hrv_|\d+_(RRV|EDR)_)')  # cardio-respiratory only, no actigraphy
MODELS = ('lgbm_notime', 'lgbm_time')


def onset_minutes(labels: np.ndarray) -> np.ndarray:
    return (np.arange(len(labels)) - sleep_onset(labels)) / iv.EPOCHS_PER_MIN


def load(features_fp: str, labels_fp: str):
    lab = pd.read_parquet(labels_fp, columns=['subject_id', '4stage'])
    cols = [c for c in pq.ParquetFile(features_fp).schema.names if FEATURE_RE.match(c)] + ['subject_id']
    feat = pd.read_parquet(features_fp, columns=cols)
    if not (feat['subject_id'].to_numpy() == lab['subject_id'].to_numpy()).all():
        raise ValueError('features and labels are not row-aligned')
    feat = feat.drop(columns='subject_id')
    for c in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[c]):
            # e.g. _hrv_higuchi_fd was saved as the repr of neurokit's (value, info) tuple
            feat[c] = pd.to_numeric(feat[c].astype(str).str.extract(r'^\(?\s*([-+0-9.eE]+|nan)', expand=False),
                                    errors='coerce')
    Xall = feat.to_numpy(np.float32)
    yall = lab['4stage'].to_numpy()
    yall = np.where(np.isnan(yall), -1, yall).astype(int)
    nights = {}
    for sid, idx in lab.groupby('subject_id', sort=False).indices.items():
        nights[f'mesa-sleep-{sid}'] = (Xall[idx], yall[idx])
    return nights, [c for c in cols if c != 'subject_id']


def design(X: np.ndarray, y: np.ndarray, time: bool) -> np.ndarray:
    return np.column_stack([X, onset_minutes(y)]) if time else X


def fit(nights, ids, val_ids, time: bool, seed: int = 0) -> lgb.LGBMClassifier:
    def stack(keys):
        Xs, ys = zip(*[(design(*nights[k], time), nights[k][1]) for k in keys])
        X, y = np.vstack(Xs), np.concatenate(ys)
        return X[y >= 0], y[y >= 0]

    Xtr, ytr = stack(ids)
    Xva, yva = stack(val_ids)
    clf = lgb.LGBMClassifier(
        objective='multiclass', n_estimators=2000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed, verbose=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


def run_night(models: dict, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    n = len(y)
    out: dict[str, np.ndarray] = {'labels': y, 'n': np.array(n)}
    for name, (src, _null) in iv.build(y).items():
        out[f'{name}/pos'] = iv.content_position(src, n)
        out[f'{name}/dist'] = iv.seam_distance(src)
        out[f'{name}/src'] = src
        Xa, ya = X[src], y[src]  # time feature recomputed from the altered night's own onset
        for m, clf in models.items():
            out[f'{name}/{m}/prob'] = clf.predict_proba(design(Xa, ya, m == 'lgbm_time')).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', default='results/controls')
    args = ap.parse_args()
    nights, cols = load(args.features, args.labels)
    split = {s: [k for k in get_split('mesa', s) if k in nights] for s in ('train', 'val', 'test')}
    print({s: len(v) for s, v in split.items()}, f'{len(cols)} features')
    models = {m: fit(nights, split['train'], split['val'], time=(m == 'lgbm_time')) for m in MODELS}
    os.makedirs(args.out, exist_ok=True)
    for sid in tqdm(split['test']):
        np.savez_compressed(os.path.join(args.out, f'{sid}.npz'), **run_night(models, *nights[sid]))


if __name__ == '__main__':
    main()
