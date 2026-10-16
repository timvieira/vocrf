"""
Value tests for the group-lasso prox in `OnlineProx` (vocrf/updates/spom.pyx).

When groups do not overlap, the prox is exact: each group's weights are
rescaled by max(0, 1 - threshold/||w_g||) (block soft-thresholding). The
budget test in spom_test.py only checks which groups end up at zero, so it does
not see whether the surviving groups shrink.

Requires only numpy and the compiled extension (python setup.py build_ext -i).
"""
import numpy as np
from vocrf.updates.spom import OnlineProx


def block_soft_threshold(w, groups, threshold):
    "Exact prox of threshold * sum_g ||w_g||_2 for non-overlapping groups."
    w = w.copy()
    for g in groups:
        n = np.linalg.norm(w[g])
        w[g] = 0 if n <= threshold else w[g] * (n - threshold) / n
    return w


def disjoint_groups(rng, d, G):
    "Random partition of range(d) into G non-empty groups."
    perm = rng.permutation(d)
    cuts = np.sort(rng.choice(np.arange(1, d), size=G-1, replace=False))
    return [list(map(int, g)) for g in np.split(perm, cuts)]


def make(groups, d, w):
    dense = OnlineProx(groups, d, C=0, L=2, eta=1.0, fudge=1)
    dense.w[:] = w.copy()
    return dense


def test_prox_threshold_disjoint():
    rng = np.random.RandomState(0)
    for _ in range(50):
        d = rng.randint(2, 30)
        G = rng.randint(1, d+1)
        groups = disjoint_groups(rng, d, G)
        w = rng.uniform(-1, 1, size=d)
        norms = np.array([np.linalg.norm(w[g]) for g in groups])
        # A threshold between the smallest and largest group norm, so that some
        # groups are zeroed and the rest must shrink.
        threshold = rng.uniform(0, norms.max())
        dense = make(groups, d, w)
        dense.prox_threshold(threshold)
        want = block_soft_threshold(w, groups, threshold)
        assert np.allclose(np.asarray(dense.w), want), (groups, threshold)


def test_prox_threshold_singletons_is_l1():
    # With one coordinate per group the prox is the L1 prox (soft-thresholding).
    rng = np.random.RandomState(1)
    d = 20
    w = rng.uniform(-1, 1, size=d)
    for threshold in [0.1, 0.5, 0.9]:
        dense = make([[k] for k in range(d)], d, w)
        dense.prox_threshold(threshold)
        want = np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
        assert np.allclose(np.asarray(dense.w), want), threshold


def test_prox_threshold_skips_ungrouped():
    # Coordinates that are in no group are not touched.
    rng = np.random.RandomState(2)
    d = 10
    w = rng.uniform(-1, 1, size=d)
    groups = [[5, 6], [7, 8, 9]]
    dense = make(groups, d, w)
    dense.prox_threshold(0.1)
    got = np.asarray(dense.w)
    assert np.array_equal(got[:5], w[:5])
    assert np.allclose(got, block_soft_threshold(w, groups, 0.1))


def test_prox_budget_disjoint():
    # The budget heuristic keeps the `budget` largest groups, with the threshold
    # halfway between the last kept norm and the first dropped norm, and shrinks
    # the kept groups by that threshold.
    rng = np.random.RandomState(3)
    for _ in range(50):
        d = rng.randint(2, 30)
        G = rng.randint(2, d+1)
        groups = disjoint_groups(rng, d, G)
        w = rng.uniform(-1, 1, size=d)
        norms = np.sort([np.linalg.norm(w[g]) for g in groups])[::-1]
        budget = rng.randint(1, G)
        threshold = (norms[budget-1] + norms[budget]) / 2
        dense = make(groups, d, w)
        dense.prox_budget(budget)
        got = np.asarray(dense.w)
        assert np.allclose(got, block_soft_threshold(w, groups, threshold)), (groups, budget)
        assert sum(np.any(got[g] != 0) for g in groups) == budget


if __name__ == '__main__':
    test_prox_threshold_disjoint()
    test_prox_threshold_singletons_is_l1()
    test_prox_threshold_skips_ungrouped()
    test_prox_budget_disjoint()
    print('[spom prox] pass')
