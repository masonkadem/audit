"""Step 4: per-epoch null-corrected probability change regressed on change in position prior.

Usage:
  python -m audit.analysis fit-prior --source overlap --labels L.parquet --out results/prior
  python -m audit.analysis fit-prior --source ingest --data $WAV2SLEEP_DATA/mesa/train --out results/prior
  python -m audit.analysis slopes --runs results/controls --prior results/prior --out results/controls_slopes
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .prior import STAGES, Prior, onset_bins, start_bins

DIST_EDGES_MIN = [0, 5, 15, 30, 62, 120, np.inf]  # 62 min = theoretical mixer RF (124 epochs)
MIXER_RF = 124  # epochs, from results/rf.json
RELOCATIONS = ['swap_thirds', 'reverse_30', 'reverse_60']
OFFSETS = ['offset_15', 'offset_30', 'offset_60']
N_BOOT = 2000
EQUIV_MARGIN = 0.10


# ---------------------------------------------------------------- priors
def train_labels_overlap(labels_fp: str) -> dict[str, np.ndarray]:
    from wav2sleep.data.nsrr import get_split

    lab = pd.read_parquet(labels_fp, columns=['subject_id', '4stage'])
    train = set(get_split('mesa', 'train'))
    out = {}
    for sid, d in lab.groupby('subject_id'):
        k = f'mesa-sleep-{sid}'
        if k in train:
            out[k] = d['4stage'].fillna(-1).to_numpy().astype(int)
    return out


def train_labels_ingest(folder: str) -> dict[str, np.ndarray]:
    from wav2sleep.data.nsrr import get_split
    from wav2sleep.settings import INTEGER_LABEL_MAPS, LABEL

    train = set(get_split('mesa', 'train'))
    out = {}
    for fp in glob(os.path.join(folder, '*.parquet')):
        k = os.path.basename(fp).replace('.parquet', '').replace('.issues', '')
        if k not in train:
            raise SystemExit(f'{k} is not in the wav2sleep MESA train split')
        y = pd.read_parquet(fp, columns=[LABEL])[LABEL].dropna().map(INTEGER_LABEL_MAPS[4]).fillna(-1)
        y = y.to_numpy().astype(int)
        scored = np.flatnonzero(y >= 0)
        out[k] = y[: scored[-1] + 1] if len(scored) else y[:0]
    return out


def train_labels_xml(folder: str) -> dict[str, np.ndarray]:
    """Labels for wav2sleep's MESA train split, parsed with wav2sleep's own XML parser."""
    from wav2sleep.data.nsrr import get_split
    from wav2sleep.data.xml import parse_xml_annotations
    from wav2sleep.settings import INTEGER_LABEL_MAPS

    out = {}
    for k in get_split('mesa', 'train'):
        fp = os.path.join(folder, f'{k}-nsrr.xml')
        if not os.path.exists(fp):
            continue
        y = parse_xml_annotations(fp).map(INTEGER_LABEL_MAPS[4]).fillna(-1).to_numpy().astype(int)
        scored = np.flatnonzero(y >= 0)
        out[k] = y[: scored[-1] + 1] if len(scored) else y[:0]
    return out


def load_priors(folder: str) -> dict[str, Prior]:
    return {a: Prior(pd.read_csv(os.path.join(folder, f'prior_{a}.csv'), index_col='bin')) for a in ('onset', 'start')}


# ---------------------------------------------------------------- per-epoch table
def epoch_table(npz_fp: str, models: list[str], priors: dict[str, Prior]) -> tuple[pd.DataFrame, list[dict]]:
    z = np.load(npz_fp)
    sid = os.path.basename(npz_fp).replace('.npz', '')
    y = z['labels']
    n = len(y)
    conds = sorted({k.split('/')[0] for k in z.files if k.endswith('/src')})
    nulls = {c: _matched_null(c) for c in conds}
    rows, metrics = [], []
    ob, sb = onset_bins(y), start_bins(y)
    for c in conds:
        src = z[f'{c}/src']
        ya = y[src]
        for m in models:
            if f'{c}/{m}/prob' not in z.files:
                continue
            p = z[f'{c}/{m}/prob']
            ok = ya >= 0
            metrics.append({'sid': sid, 'cond': c, 'model': m, 'y': ya[ok], 'pred': p[ok].argmax(1)})
        null = nulls[c]
        if null is None or null not in conds:
            continue
        pos, pos0, pid = z[f'{c}/pos'], z[f'{null}/pos'], z['identity/pos']
        i = np.flatnonzero((pos >= 0) & (pos0 >= 0) & (pid >= 0) & (y >= 0))
        j, j0 = pos[i], pos0[i]
        dist = z[f'{c}/dist'][j]
        m_alt = len(src)
        edge = np.minimum(np.minimum(i, n - 1 - i), np.minimum(j, m_alt - 1 - j))  # nearer of either night's edges
        d_onset = priors['onset'](onset_bins(ya)[j]) - priors['onset'](ob[i])
        d_start = priors['start'](start_bins(ya)[j]) - priors['start'](sb[i])
        for m in models:
            key = f'{c}/{m}/prob'
            if key not in z.files:
                continue
            p, p0, pu = z[key][j], z[f'{null}/{m}/prob'][j0], z[f'identity/{m}/prob'][pid[i]]
            df = pd.DataFrame({'sid': sid, 'cond': c, 'model': m, 'epoch': i, 'dist': dist, 'edge': edge})
            for s, name in enumerate(STAGES):
                df[f'dp_{name}'] = p[:, s] - p0[:, s]  # (p_I - p_U) - (p_null - p_U)
                df[f'dpraw_{name}'] = p[:, s] - pu[:, s]
                df[f'dnull_{name}'] = p0[:, s] - pu[:, s]
                df[f'dxo_{name}'] = d_onset[:, s]
                df[f'dxs_{name}'] = d_start[:, s]
            rows.append(df)
    return (pd.concat(rows) if rows else pd.DataFrame()), metrics


def _matched_null(cond: str) -> str | None:
    if cond.startswith('reverse_'):
        return 'null_jitter_' + cond.split('_')[1]
    if cond == 'swap_thirds':
        return 'null_jitter_thirds'
    if cond.startswith('offset_'):
        return 'null_jitter_offset_' + cond.split('_')[1]
    return None


# ---------------------------------------------------------------- slopes with participant bootstrap
def _suff(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    g = pd.DataFrame({'sid': df['sid'], 'x': df[x], 'y': df[y]})
    g['xx'], g['xy'] = g.x * g.x, g.x * g.y
    s = g.groupby('sid').agg(n=('x', 'size'), x=('x', 'sum'), y=('y', 'sum'), xx=('xx', 'sum'), xy=('xy', 'sum'))
    return s


def _slope(s: np.ndarray) -> float:
    n, x, y, xx, xy = s
    den = n * xx - x * x
    return float((n * xy - x * y) / den) if den > 1e-12 else np.nan


def slope_ci(df: pd.DataFrame, x: str, y: str, rng: np.random.Generator) -> dict:
    s = _suff(df, x, y)
    if len(s) < 2:
        return {'slope': np.nan, 'lo': np.nan, 'hi': np.nan, 'n_epochs': int(s['n'].sum()), 'n_participants': len(s)}
    a = s.to_numpy()
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    boots = np.array([_slope(a[b].sum(0)) for b in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {'slope': _slope(a.sum(0)), 'lo': lo, 'hi': hi, 'n_epochs': int(a[:, 0].sum()), 'n_participants': len(a)}


def dist_bin(d_epochs: pd.Series) -> pd.Series:
    labels = [f'{a:g}-{b:g}' for a, b in zip(DIST_EDGES_MIN[:-1], DIST_EDGES_MIN[1:])]
    return pd.cut(d_epochs / 2, DIST_EDGES_MIN, right=False, labels=labels)


def all_slopes(tab: pd.DataFrame, min_dist_epochs: int, primary_dist_epochs: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tab = tab[tab['dist'] >= min_dist_epochs].copy()
    tab['dbin'] = dist_bin(tab['dist'])
    out = []

    def add(sub, family, cond, dbin, anchor, ycol='dp'):
        xp = 'dxo' if anchor == 'onset' else 'dxs'
        for m, g in sub.groupby('model'):
            for s in STAGES:
                r = slope_ci(g, f'{xp}_{s}', f'{ycol}_{s}', rng)
                out.append({'model': m, 'family': family, 'cond': cond, 'dist_bin': dbin, 'anchor': anchor,
                            'y': ycol, 'stage': s, **r})

    reloc = tab[tab['cond'].isin(RELOCATIONS)]
    offs = tab[tab['cond'].isin(OFFSETS)]
    # Primary (pre-registered): relocations pooled, onset prior, distance >= primary radius.
    add(reloc[reloc['dist'] >= primary_dist_epochs], 'primary', 'relocation_pooled', f'>={primary_dist_epochs}ep', 'onset')
    add(offs[offs['dist'] >= primary_dist_epochs], 'primary_offset', 'offset_pooled', f'>={primary_dist_epochs}ep', 'start')
    # Secondary: per condition, per distance bin, both anchors, uncorrected change.
    for c, g in tab[tab['cond'].isin(RELOCATIONS + OFFSETS)].groupby('cond'):
        for anchor in ('onset', 'start'):
            add(g, 'by_condition', c, 'all', anchor)
            for b, gb in g.groupby('dbin', observed=True):
                add(gb, 'by_distance', c, str(b), anchor)
        add(g, 'uncorrected', c, 'all', 'onset' if c in RELOCATIONS else 'start', ycol='dpraw')
        far = g[g['dist'] >= MIXER_RF]
        for label, ge in (('edge_exposed', far[far['edge'] < MIXER_RF]), ('interior', far[far['edge'] >= MIXER_RF])):
            add(ge, 'by_edge', c, label, 'onset' if c in RELOCATIONS else 'start')
    return pd.DataFrame(out)


def verdicts(sl: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered decision rules to the primary relocation slopes, per model."""
    rows = []
    for m, g in sl[(sl.family == 'primary')].groupby('model'):
        g = g.set_index('stage')
        faithful = bool(((g.lo >= -EQUIV_MARGIN) & (g.hi <= EQUIV_MARGIN)).all())
        top2 = set(g.slope.nlargest(2).index)
        shortcut = bool((g.loc[['Deep', 'REM'], 'lo'] > 0).all() and top2 == {'Deep', 'REM'})
        rows.append({'model': m, 'faithful_confirmed': faithful, 'shortcut_confirmed': shortcut})
    return pd.DataFrame(rows)


def accuracy_table(metrics: list[dict]) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame(metrics)
    for (c, m), g in df.groupby(['cond', 'model']):
        y, p = np.concatenate(g.y.to_list()), np.concatenate(g.pred.to_list())
        acc_night = [np.mean(a == b) for a, b in zip(g.y, g.pred) if len(a)]
        rows.append({'cond': c, 'model': m, 'n_nights': len(g), 'acc_pooled': np.mean(y == p),
                     'acc_mean_night': np.mean(acc_night), 'kappa_pooled': cohen_kappa_score(y, p)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('fit-prior')
    a.add_argument('--source', choices=['xml', 'overlap', 'ingest'], required=True)
    a.add_argument('--labels')
    a.add_argument('--data')
    a.add_argument('--out', default='results/prior')
    b = sp.add_parser('slopes')
    b.add_argument('--runs', required=True)
    b.add_argument('--prior', default='results/prior')
    b.add_argument('--models', nargs='+', required=True)
    b.add_argument('--min-dist', type=int, default=0, help='exclude epochs closer than this to a seam')
    b.add_argument('--primary-dist', type=int, required=True, help='pre-registered radius R (epochs)')
    b.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.cmd == 'fit-prior':
        loaders = {'xml': train_labels_xml, 'overlap': train_labels_overlap, 'ingest': train_labels_ingest}
        nights = loaders[args.source](args.labels if args.source == 'overlap' else args.data)
        os.makedirs(args.out, exist_ok=True)
        for anchor in ('onset', 'start'):
            Prior.fit(nights, anchor).table.to_csv(os.path.join(args.out, f'prior_{anchor}.csv'))
        json.dump({'source': args.source, 'n_nights': len(nights)}, open(os.path.join(args.out, 'meta.json'), 'w'))
        print(f'prior from {len(nights)} training nights ({args.source})')
        return

    priors = load_priors(args.prior)
    tabs, mets = [], []
    for fp in sorted(glob(os.path.join(args.runs, '*.npz'))):
        t, m = epoch_table(fp, args.models, priors)
        tabs.append(t)
        mets += m
    tab = pd.concat(tabs)
    os.makedirs(args.out, exist_ok=True)
    sl = all_slopes(tab, args.min_dist, args.primary_dist)
    sl.to_csv(os.path.join(args.out, 'slopes.csv'), index=False)
    verdicts(sl).to_csv(os.path.join(args.out, 'verdicts.csv'), index=False)
    accuracy_table(mets).to_csv(os.path.join(args.out, 'accuracy.csv'), index=False)
    null_size = tab.groupby(['model', 'cond'])[[f'dnull_{s}' for s in STAGES]].apply(lambda g: g.abs().mean())
    null_size.to_csv(os.path.join(args.out, 'null_abs_change.csv'))
    print(verdicts(sl).to_string())


if __name__ == '__main__':
    main()
