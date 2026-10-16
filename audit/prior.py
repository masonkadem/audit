"""Position prior P(stage | time since sleep onset) and P(stage | time since recording start)."""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = ['Wake', 'Light', 'Deep', 'REM']  # wav2sleep 4-class order
BIN_EPOCHS = 60  # 30-minute bins
MIN_BIN_EPOCHS = 1000  # sparser bins are pooled into the nearest populated bin


def sleep_onset(labels: np.ndarray) -> int:
    """Index of the first scored non-Wake epoch (len(labels) if never asleep)."""
    asleep = np.flatnonzero(labels > 0)
    return int(asleep[0]) if len(asleep) else len(labels)


def onset_bins(labels: np.ndarray) -> np.ndarray:
    """30-min bin of time since sleep onset for each epoch (negative = before onset)."""
    return np.floor_divide(np.arange(len(labels)) - sleep_onset(labels), BIN_EPOCHS)


def start_bins(labels: np.ndarray) -> np.ndarray:
    return np.arange(len(labels)) // BIN_EPOCHS


class Prior:
    """Lookup table bin -> P(stage), with out-of-range bins clamped to the populated range."""

    def __init__(self, table: pd.DataFrame):
        self.table = table  # index: bin, columns: STAGES (+ n)
        self.lo, self.hi = int(table.index.min()), int(table.index.max())
        self._p = table[STAGES].to_numpy()

    def __call__(self, bins: np.ndarray) -> np.ndarray:
        return self._p[np.clip(bins, self.lo, self.hi) - self.lo]

    @classmethod
    def fit(cls, nights: dict[str, np.ndarray], anchor: str = 'onset') -> 'Prior':
        binner = onset_bins if anchor == 'onset' else start_bins
        rows = []
        for lab in nights.values():
            ok = lab >= 0
            rows.append(pd.DataFrame({'bin': binner(lab)[ok], 'stage': lab[ok].astype(int)}))
        df = pd.concat(rows, ignore_index=True)
        counts = pd.crosstab(df['bin'], df['stage']).reindex(columns=range(4), fill_value=0)
        n = counts.sum(axis=1)
        populated = n.index[n >= MIN_BIN_EPOCHS]
        lo, hi = populated.min(), populated.max()
        counts = counts.groupby(np.clip(counts.index, lo, hi)).sum()
        table = counts.div(counts.sum(axis=1), axis=0)
        table.columns = STAGES
        table['n'] = counts.sum(axis=1)
        table.index.name = 'bin'
        return cls(table)
