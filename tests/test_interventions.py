import numpy as np

from audit import interventions as iv
from audit.prior import Prior, onset_bins


def night(n=960, onset=40):
    y = np.r_[np.zeros(onset, int), np.random.default_rng(0).integers(1, 4, n - onset)]
    return y


def test_null_literal_is_identity():
    assert (iv.null_literal(901, 60) == np.arange(901)).all()
    assert len(iv.seams(iv.null_literal(901, 60))) == 0


def test_null_jitter_has_seams_and_bounded_drift():
    n = 960
    cuts = iv.block_cuts(n, 60)
    src = iv.null_jitter(n, cuts)
    assert len(iv.seams(src)) == len(cuts)
    pos = iv.content_position(src, n)
    kept = pos >= 0
    assert np.abs(pos[kept] - np.arange(n)[kept]).max() <= 1
    assert abs(len(src) - n) <= 1


def test_relocations_are_permutations():
    n = 961
    for src in (iv.swap_thirds(n), iv.reverse_blocks(n, 60), iv.reverse_blocks(n, 120)):
        assert sorted(src) == list(range(n))
    a, b = iv.third_cuts(n)
    s = iv.swap_thirds(n)
    assert s[0] == b and s[-1] == a - 1
    assert list(iv.seams(s)) == [n - b, n - a]


def test_reverse_blocks_keeps_within_block_order():
    src = iv.reverse_blocks(150, 60)
    assert list(src[:30]) == list(range(120, 150))
    assert list(src[30:90]) == list(range(60, 120))


def test_offset_requires_wake_and_keeps_real_night():
    y = night(onset=40)
    assert iv.offset(y, 60) is None
    src = iv.offset(y, 30)
    pos = iv.content_position(src, len(y))
    assert (pos == np.arange(len(y)) + 30).all()
    # time since onset is unchanged by an offset, by construction
    ya = y[src]
    assert (onset_bins(ya)[pos] == onset_bins(y)).all()


def test_seam_distance():
    src = np.r_[np.arange(10, 20), np.arange(0, 10)]
    d = iv.seam_distance(src)
    assert d[9] == 0 and d[10] == 0 and d[0] == 9 and d[19] == 9


def test_prior_clamps_and_sums_to_one():
    nights = {str(k): night(onset=10 + k) for k in range(30)}
    p = Prior.fit(nights)
    assert np.allclose(p.table[['Wake', 'Light', 'Deep', 'REM']].sum(1), 1)
    assert p(np.array([-999, 999])).shape == (2, 4)
