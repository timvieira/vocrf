
import pylab as pl
import numpy as np
from vocrf.updates.tree_prox import TreeProx, Tree
from arsenal import colors
from itertools import product as xprod
from vocrf.util import prefix_closure


class TreeTest(Tree):

    def __init__(self):
        """
        Corrected Figure a)
        http://aclweb.org/anthology/D13-1024
        """
        super(TreeTest, self).__init__(())

        # new tree
        self.add((3,))
        self.add((3,4))
        self.add((3,4,6))
        self.add((3,4,7))
        self.add((3,4,6,6))
        self.add((3,4,7,7))
        self.add((3,4,6,6,5))
        self.add((3,4,6,6,7))

        post = [z.z for z in self.postorder()]
        assert post == [(3, 4, 6, 6, 5),
                        (3, 4, 6, 6, 7),
                        (3, 4, 6, 6),
                        (3, 4, 6),
                        (3, 4, 7, 7),
                        (3, 4, 7),
                        (3, 4),
                        (3,),
                        ()]


def random_contexts(sigma, depth, size):
    """Generate random contexts, such that `|W|=size` (before closure) of
    |w|=`depth` for w in W, from a |alphabet|=sigma.

    """
    sigma = list(range(sigma))
    possible = list(xprod(*(sigma,)*depth))
    np.random.shuffle(possible)
    return list(prefix_closure(possible[:size]))


class TreeProxTest(TreeProx):
    """Test cases for tree-structured group lasso.

    test_D13
    Corrected Figure a
    http://aclweb.org/anthology/D13-1024

    """

    def __init__(self):
        arity = 3
        super(TreeProxTest, self).__init__(arity)

        if 0:
            # (3,) is replaced with ()
            self.add((4,))
            self.add((4,6))
            self.add((4,7))
            self.add((4,6,6))
            self.add((4,7,7))
            self.add((4,6,6,5))
            self.add((4,6,6,7))

        else:
            C = random_contexts(self.arity, 10, 10)
            for c in C:
                self.add(c)

        self.update_groups()

    def test_prox(self):
        D = len(self.topo)
        kappa = 0.8

        def obj(w):
            "Proximal objective."
            d = (w_orig - w)
            return 0.5 * d.dot(d) + kappa * self.group_norm(w)

        w_orig = np.random.uniform(-1,1,size=D)

        #w_orig = np.array([3.0, 4.0, 6.0, 6.0, 4.0, 5.0, 7.0, 7.0])
        #w_paper = asarray([2.8, 3.5, 4.8, 4.3, 2.3, 3.0, 5.6, 4.9])
        w = w_orig.copy()
        self.prox(kappa, w)

        from scipy.optimize import minimize
        opt = minimize(obj, w_orig, method='bfgs').x
        #opt = fmin(obj, w_orig)

        print('opt', obj(opt))
        print('ours', obj(w))
        #print('paper', obj(w_paper))
        #err = abs(opt-w).max()
        #print('L-inf error:', err)

        assert obj(w) <= obj(opt) #+ 1e-4
        print('[test prox]', colors.light.green % 'pass')

    def test_budget(self):
        "Test the budget heuristic."
        w = np.random.uniform(-1,1,size=self.D)
        w_orig = w.copy()

        def L0(threshold):
            ww = w_orig.copy()
            self.prox(threshold, ww)
            return (np.abs(ww) > 0).sum()

        # Check that the find_threshold gives a conservative estimate for L0 budget
        M = len(w)
        f = {}
        est = {}
        for budget in range(M+1):
            est[budget] = self.find_threshold(budget, w)
            l0 = L0(est[budget])
            f[budget] = l0
            assert l0 <= budget
        # Check end points
        assert f[0] == 0
        assert f[M] == M

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

        # How many operating points (budgets) do we miss that the numerical
        # method achieves?
        #
        # Note that we don't expect perfect coverage because the heuristic
        # pretends that groups don't overlap.
        #
        #  ^^ We appear to be getting great coverage. Should we revise this
        #     statement?
        numerical_points = list(sorted(set(numerical_y)))
        heuristic_points = list(sorted(set(heuristic_y)))
        print('numerical:', numerical_points)
        print('heuristic:', heuristic_points)

        recall = len(set(numerical_points) & set(heuristic_points)) / len(set(numerical_points))
        print('recall: %.2f' % recall)

        if 0:
            # This plot is for debugging the conservativeness of the budget
            # heuristic, which is now asserted above.
            pl.title('Ability to conservatively meet the budget')
            xs, ys = list(zip(*sorted(f.items())))
            pl.plot(xs, xs, c='k', alpha=0.5, linestyle=':')
            pl.plot(xs, ys, alpha=0.5, c='r', lw=2)
            pl.scatter(xs, ys, lw=0)
            pl.show()

        print('[test budget]', colors.light.green % 'pass')


if __name__ == '__main__':
    TreeTest()
    tpt = TreeProxTest()
    tpt.test_prox()
    tpt.test_budget()
