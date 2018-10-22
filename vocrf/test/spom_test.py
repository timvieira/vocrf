
import pylab as pl
import numpy as np
from arsenal import colors

from vocrf.model import VoCRF
from vocrf.updates.spom import OnlineProx
from vocrf.util import prefix_closure


class SPOM_tests(VoCRF):
    """
    Test model with the lazy weight updater.
    """

    def __init__(self, *args, **kw):
        super(SPOM_tests, self).__init__(*args, **kw)
        self.H = len(self.states)*self.A

    def ideal_runtime(self, w):
        "Evaluate the ideal runtime penalty."
        return len(prefix_closure({c for c in self.C if w[self.context_feature_id(c)] != 0}))

    def L0_group_norm_proxy(self, dense):
        "Evaluate idealized group lasso penalty based on L0."
        dense.update_group_norm()
        active = np.asarray(dense.group_norm) > 0
        group_size = np.asarray(dense.group_size)
        return group_size[active].sum()

    def test_prox_budget_heuristic(self, w=None, verbose=0):
        # Approx prox update we should have |active(groups)| <= budget. In the
        # case of overlapping groups, we penalize some groups too many times so
        # the bound will be looser.

        # What can we say about the prox operator? We can test that the prox
        # operator shrinks parameters to be under the group budget (even though
        # the constraint is tight if there are nonoverlapping groups and loose
        # otherwise).

        #groups = [[i] for i in range(self.H)]   # reduces to ordinary L1 penalty
        groups = self.group_structure()

        coverage = {k for g in groups for k in g}
        assert len(coverage) == len(self.C)   # all features must appear in some group.

        if verbose:
            print('[prox]', groups)

        if w is None:
            w_orig = np.random.uniform(-1, 1, size=self.H)
        else:
            print('H =', self.H, '|raw contexts| =', len(w))
            w_orig = np.zeros(self.H)
            for k in w:
                w_orig[self.context_feature_id(k)] = w[k]
        del w

        print((w_orig != 0).sum(), 'active features')
        print(sum(np.abs(w_orig[list(G)]).sum() > 0 for G in groups), 'active groups')
        print(len(groups), 'total groups')

        dense = OnlineProx(groups, self.H, C=0, L=2, eta=1.0, fudge=1)
        dense.w[:] = w_orig.copy()

        def L0(threshold):
            dense.w[:] = w_orig.copy()
            dense.prox_threshold(threshold)
            return self.L0_group_norm_proxy(dense)

        # Check that the find_threshold gives a conservative estimate for L0 budget
        f = {}
        est = {}
        M = len(groups) #sum(len(g) for g in groups)
        for budget in range(M+1):
            dense.w[:] = w_orig.copy()
            est[budget] = dense.find_threshold(budget)
            f[budget] = L0(est[budget])
            assert f[budget] <= budget, [budget, f[budget], est[budget]]
            # check that L0 on this group structure is what we wanted
            assert self.ideal_runtime(dense.w) == f[budget]

        # Check end points
        #
        # The maximum number of active groups. Is upper bounded by the number of
        # groups with a nonzero norm, which might <<= tne number of groups.
        max_active = sum(np.abs(w_orig[list(G)]).sum() > 0 for G in groups)
        #print('max_active:', max_active, 'number of groups:', len(groups))
        assert f[0] == 0
        assert f[max_active] == max_active
        assert f[M] == max_active

        # Check coverage against a numerical sweep.
        numerical_x = np.linspace(0, M+1, 10000)
        numerical_y = np.array([L0(threshold) for threshold in numerical_x])
        heuristic_x = np.array(sorted(est.values()))
        heuristic_y = np.array([L0(threshold) for threshold in heuristic_x])

        if 0:
            pl.title('threshold vs L0 coverage')
            keep = numerical_y > 0
            pl.plot(numerical_x[keep], numerical_y[keep], c='b', alpha=0.5, lw=2, label='numerical')
            pl.plot(heuristic_x, heuristic_y, alpha=0.5, c='r', lw=2, label='heuristic')
            pl.scatter(heuristic_x, heuristic_y, lw=0)
            pl.legend(loc='best')
            pl.show()

        numerical_points = list(sorted(set(numerical_y)))
        heuristic_points = list(sorted(set(heuristic_y)))

        # How many operating points (budgets) do we miss that the numerical
        # method achieves?
        #
        # Note that we don't expect perfect coverage because the heuristic
        # pretends that groups don't overlap.
        #
        #  ^^ We appear to be getting great coverage. Should we revise this
        #     statement?

        print(numerical_points)
        print(heuristic_points)

        recall = len(set(numerical_points) & set(heuristic_points)) / len(set(numerical_points))
        print('recall: %.2f' % recall)
        assert recall >= 0.99, recall

        if 0:
            pl.title('Ability to conservatively meet the budget')
            xs, ys = list(zip(*sorted(f.items())))
            pl.plot(xs, xs, c='k', alpha=0.5, linestyle=':')
            pl.plot(xs, ys, alpha=0.5, c='r', lw=2)
            pl.scatter(xs, ys, lw=0)
            pl.show()

        print('[test budget]', colors.light.green % 'pass')


def test():
    # The following example include multiple backoffs.
    sigma1 = 'abcdefg'
    C1 = list(map(tuple, [
        'aaa',
        'bba',
    ]))

    # The folling example is a conventional first-order tagger. It does not use
    # failure transition. I've added this test, as it was useful for debugging.
    sigma2 = 'ab'
    C2 = list(map(tuple, [
        'aa',
        'ab',
        'ba',
        'bb',
    ]))

    sigma3 = 'abc'
    C3 = list(map(tuple, [
        'aaaaaaac',
        'baaaaaaa',
        'baaaaaab',
        'bcacab',
        'aba',
        'abb',
        'baaba',
        'bb',
    ]))

    print()
    print('[prox] Context 1')
    SPOM_tests(sigma1, C1).test_prox_budget_heuristic()

    print()
    print('[prox] Context 2:')
    SPOM_tests(sigma2, C2).test_prox_budget_heuristic()

    print()
    print('[prox] Context 3:')
    SPOM_tests(sigma3, C3).test_prox_budget_heuristic()

    if 0:
        filename = 'weights.pkl'
        print()
        print('[prox] Context %s' % filename)
        import pickle
        with file(filename) as f:
            w = pickle.load(f)
        SPOM_tests({a for c in w for a in c}, list(w.keys())).test_prox_budget_heuristic()     # with random weights
        print()
        print('===================================')
        SPOM_tests({a for c in w for a in c}, list(w.keys())).test_prox_budget_heuristic(w=w)  # with saved weights


if __name__ == '__main__':
    test()
