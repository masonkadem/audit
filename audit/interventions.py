"""Night-level interventions expressed as epoch index maps.

Every intervention returns `src`: altered epoch j carries the content (signals and label)
of original epoch src[j]. Signals are never modified within an epoch, so labels move with
content by construction. Seams are altered indices j where src[j] != src[j-1] + 1.
"""

from __future__ import annotations

import numpy as np

EPOCHS_PER_MIN = 2
WAKE = 0


def _blocks(n: int, block: int) -> list[np.ndarray]:
    return [np.arange(s, min(s + block, n)) for s in range(0, n, block)]


def identity(n: int) -> np.ndarray:
    return np.arange(n)


def null_literal(n: int, block: int) -> np.ndarray:
    """Cut into blocks and reassemble in order. Bit-identical to the original night."""
    return np.concatenate(_blocks(n, block))


def null_jitter(n: int, cuts: list[int]) -> np.ndarray:
    """In-order reassembly with a real waveform discontinuity at each cut.

    Alternates deleting the first epoch after a cut and duplicating the last epoch before it,
    so every cut is a seam while cumulative position drift stays within one epoch.
    """
    out, prev = [], 0
    for k, c in enumerate(sorted(cuts)):
        seg = np.arange(prev, c)
        if k % 2 == 0:
            out.append(seg)
            prev = c + 1  # drop epoch c
        else:
            out.append(np.concatenate([seg, seg[-1:]]))  # duplicate epoch c-1
            prev = c
    out.append(np.arange(prev, n))
    return np.concatenate(out)


def block_cuts(n: int, block: int) -> list[int]:
    return list(range(block, n, block))


def third_cuts(n: int) -> list[int]:
    return [n // 3, 2 * n // 3]


def swap_thirds(n: int) -> np.ndarray:
    a, b = third_cuts(n)
    return np.concatenate([np.arange(b, n), np.arange(a, b), np.arange(0, a)])


def reverse_blocks(n: int, block: int) -> np.ndarray:
    return np.concatenate(_blocks(n, block)[::-1])


def offset(labels: np.ndarray, n_epochs: int) -> np.ndarray | None:
    """Prepend the night's own first `n_epochs` epochs. Requires they are all scored Wake."""
    if n_epochs > len(labels) or not (labels[:n_epochs] == WAKE).all():
        return None
    return np.concatenate([np.arange(n_epochs), np.arange(len(labels))])


def seams(src: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(src) != 1) + 1


def seam_distance(src: np.ndarray, extra_seams: np.ndarray | None = None) -> np.ndarray:
    """Distance (epochs) from each altered epoch to the nearest seam; 0 = adjacent to a seam."""
    s = seams(src)
    if extra_seams is not None:
        s = np.union1d(s, extra_seams)
    j = np.arange(len(src))
    if len(s) == 0:
        return np.full(len(src), np.iinfo(np.int32).max)
    right = np.searchsorted(s, j, side='right')  # first seam > j
    d_left = np.where(right > 0, j - s[np.maximum(right - 1, 0)], np.inf)  # j at/after seam s
    d_right = np.where(right < len(s), s[np.minimum(right, len(s) - 1)] - 1 - j, np.inf)  # j before seam
    return np.minimum(d_left, d_right).astype(np.int64)


def content_position(src: np.ndarray, n: int) -> np.ndarray:
    """Altered index of each original epoch (last occurrence), -1 if deleted.

    Last occurrence means prepended offset copies are excluded and the real night is analysed.
    """
    pos = np.full(n, -1, dtype=np.int64)
    pos[src] = np.arange(len(src))  # later assignments win
    return pos


def build(labels: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    """All interventions for one night: name -> (src, matched null name)."""
    n = len(labels)
    out: dict[str, tuple[np.ndarray, str]] = {'identity': (identity(n), '')}
    for mins in (30, 60):
        b = mins * EPOCHS_PER_MIN
        out[f'null_literal_{mins}'] = (null_literal(n, b), '')
        out[f'null_jitter_{mins}'] = (null_jitter(n, block_cuts(n, b)), '')
        out[f'reverse_{mins}'] = (reverse_blocks(n, b), f'null_jitter_{mins}')
    out['null_jitter_thirds'] = (null_jitter(n, third_cuts(n)), '')
    out['swap_thirds'] = (swap_thirds(n), 'null_jitter_thirds')
    for mins in (15, 30, 60):
        k = mins * EPOCHS_PER_MIN
        src = offset(labels, k)
        if src is not None:
            out[f'null_jitter_offset_{mins}'] = (null_jitter(n, [k]), '')
            out[f'offset_{mins}'] = (src, f'null_jitter_offset_{mins}')
    return out
