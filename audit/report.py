"""Figures for report.md: stage-by-position, and slope by seam distance.

Usage: python -m audit.report --prior results/prior --runs results/w2s:all results/controls:lgbm_time \
           --slopes results/w2s_slopes results/controls_slopes --out figures
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import load_priors
from .prior import BIN_EPOCHS, STAGES, onset_bins

# Reference categorical slots 1-4 (validated adjacent-pair set); text stays in ink tokens.
COLORS = dict(zip(STAGES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _label_ends(ax, x, curves):
    for s, y in curves.items():
        ax.annotate(s, (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK)


def unaltered_by_position(run_dir: str, model: str, max_bin: int):
    """Mean predicted P(stage) and observed frequency by onset bin, on unaltered nights."""
    pred, obs = [], []
    for fp in sorted(glob(os.path.join(run_dir, '*.npz'))):
        z = np.load(fp)
        if f'identity/{model}/prob' not in z.files:
            continue
        y = z['labels']
        b = onset_bins(y)
        p = z[f'identity/{model}/prob'][: len(y)]
        ok = (y >= 0) & (b >= 0) & (b <= max_bin)
        pred.append(pd.DataFrame(p[ok], columns=STAGES).assign(bin=b[ok]))
        obs.append(pd.get_dummies(pd.Categorical(y[ok], categories=range(4))).set_axis(STAGES, axis=1).assign(bin=b[ok]))
    if not pred:
        return None, None
    return pd.concat(pred).groupby('bin').mean(), pd.concat(obs).astype(float).groupby('bin').mean()


def fig_stage_by_position(prior_dir: str, runs: list[tuple[str, str]], out: str, max_bin: int = 15):
    prior = load_priors(prior_dir)['onset'].table
    prior = prior[(prior.index >= 0) & (prior.index <= max_bin)]
    ncol = 1 + len(runs)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.4), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    x = prior.index.to_numpy() * BIN_EPOCHS / 120  # hours since onset
    ax = axes[0]
    _style(ax)
    for s in STAGES:
        ax.plot(x, prior[s], color=COLORS[s], lw=2)
    _label_ends(ax, x, {s: prior[s].to_numpy() for s in STAGES})
    ax.set_title('Position prior (MESA train)', fontsize=9, color=INK, loc='left')
    ax.set_ylabel('P(stage)', fontsize=8, color=INK2)
    for ax, (run_dir, model) in zip(axes[1:], runs):
        _style(ax)
        pred, obs = unaltered_by_position(run_dir, model, max_bin)
        ax.set_title(f'{model}: predicted (solid) vs scored (dashed)', fontsize=9, color=INK, loc='left')
        if pred is None:
            ax.text(0.5, 0.5, 'not yet run', transform=ax.transAxes, ha='center', color=INK2)
            continue
        xb = pred.index.to_numpy() * BIN_EPOCHS / 120
        for s in STAGES:
            ax.plot(xb, pred[s], color=COLORS[s], lw=2)
            ax.plot(xb, obs[s], color=COLORS[s], lw=1.2, ls='--')
        _label_ends(ax, xb, {s: pred[s].to_numpy() for s in STAGES})
    for ax in axes:
        ax.set_xlabel('Hours since sleep onset', fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_slope_by_distance(slope_dirs: list[str], out: str):
    sl = pd.concat([pd.read_csv(os.path.join(d, 'slopes.csv')) for d in slope_dirs if os.path.exists(os.path.join(d, 'slopes.csv'))])
    sl = sl[(sl.family == 'by_distance') & (((sl.cond.str.startswith('offset')) & (sl.anchor == 'start'))
                                           | ((~sl.cond.str.startswith('offset')) & (sl.anchor == 'onset')))]
    models, conds = sorted(sl.model.unique()), sorted(sl.cond.unique())
    fig, axes = plt.subplots(len(models), len(conds), figsize=(2.6 * len(conds), 2.2 * len(models)),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    order = ['0-5', '5-15', '15-30', '30-62', '62-120', '120-inf']
    for r, m in enumerate(models):
        for c, cond in enumerate(conds):
            ax = axes[r, c]
            _style(ax)
            ax.axhline(0, color=INK2, lw=0.8)
            g = sl[(sl.model == m) & (sl.cond == cond)]
            for k, s in enumerate(STAGES):
                h = g[g.stage == s].set_index('dist_bin').reindex(order)
                xs = np.arange(len(order)) + (k - 1.5) * 0.15
                ax.errorbar(xs, h.slope, yerr=[h.slope - h.lo, h.hi - h.slope], fmt='o', ms=4, lw=1,
                            color=COLORS[s], label=s if (r == 0 and c == 0) else None)
            ax.set_xticks(range(len(order)), order, rotation=60, fontsize=7)
            if r == 0:
                ax.set_title(cond, fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f'{m}\nslope', fontsize=8, color=INK2)
    axes[-1, len(conds) // 2].set_xlabel('Seam distance (min)', fontsize=8, color=INK2)
    fig.legend(loc='upper center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', default='results/prior')
    ap.add_argument('--runs', nargs='*', default=[], help='dir:model pairs for unaltered panels')
    ap.add_argument('--slopes', nargs='*', default=[])
    ap.add_argument('--out', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [tuple(r.rsplit(':', 1)) for r in args.runs]
    fig_stage_by_position(args.prior, runs, os.path.join(args.out, 'stage_by_position.png'))
    if args.slopes:
        fig_slope_by_distance(args.slopes, os.path.join(args.out, 'slope_by_distance.png'))


if __name__ == '__main__':
    main()
