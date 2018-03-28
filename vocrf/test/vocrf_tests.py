from __future__ import division
import numpy as np
from collections import defaultdict
from arsenal import colors

from vocrf.model import VoCRF
from vocrf.score import BasicEdgeScoreDict
from vocrf.sparse import SparseBinaryVector
from vocrf.updates.spom import OnlineProx
from vocrf.score import ScoringModel
from vocrf.util import fdcheck
from lazygrad.adagrad import LazyRegularizedAdagrad


class FailureArcTest(VoCRF):

    def __init__(self, *args, **kw):
        super(FailureArcTest, self).__init__(*args, **kw)

    def test_gradient(self, T):
        S = BasicEdgeScoreDict(defaultdict(lambda: np.random.uniform(-1, 1)))

        y = np.random.randint(0, self.A, T).astype(np.int32)
        self.gradient(T, y, S)

        c = fdcheck(lambda: self.objective(T, y, S), S.weights, S.d_weights) #.show()
        assert c.pearson >= 0.999999
        assert c.max_err <= 1e-8
        assert np.allclose(c.expect, c.got)
        print '[test gradient]', colors.light_green % 'pass'
        print


class MockInstance:
    """
    Simple instance with random features for each token.

    `ScoringModel` expects an instance with these fields.

    """
    def __init__(self, N, A, D, K, y=None):
        """
        N: sequence length
        A: size of alphabet
        D: token property dimensionality
        K: number of active features
        """
        self.N = N
        self.tags = np.random.randint(0, A, N).astype(np.int32) if y is None else y
        self.properties = {}
        for t in range(N):
            self.properties[t] = SparseBinaryVector(np.random.randint(D, size=K))


class LazyFailureArcTest(VoCRF):
    """
    Test model with the lazy weight updater.
    """

    def __init__(self, *args, **kw):
        super(LazyFailureArcTest, self).__init__(*args, **kw)
        self.H = len(self.states)*self.A

    def test_gradient(self, T):

        D = 100

        # Give the updater 'trivial' parameters so that it doesn't make the test
        # complicated.
        sparse = LazyRegularizedAdagrad(D*self.A, C=0, L=2, eta=1.0, fudge=1)
        sparse.w[:] = np.random.uniform(-1, 1, size=sparse.d)
        sparse.step = 0

        #groups = [[i] for i in range(self.H)]
        groups = self.group_structure()

        dense = OnlineProx(groups, self.H, C=0, L=2, eta=1.0, fudge=1)
        dense.w[:] = np.random.uniform(-1, 1, size=dense.d)

        # Since updates are done inplace. We need to copy the original
        # parameters so that we can later 'infer' the gradient step.
        sparse_W_copy = np.array(sparse.w, copy=1)
        dense_W_copy = np.array(dense.w, copy=1)

        x = MockInstance(T, self.A, D = D, K = 5)

        S = ScoringModel(x, self.A, self.feature_backoff, sparse, dense)
        self.gradient(T, x.tags, S)
        S.backprop()

        def func():
            S = ScoringModel(x, self.A, self.feature_backoff, sparse, dense)
            return self.objective(T, x.tags, S)

        if 0:
            # Note: we don't run this test because it doesn't pass! This is
            # because lazy adagrad manipulates the stepsize somewhat
            # unpredictably in order to get the benefit of inlining (avoiding
            # allocating temproary datastructures to buffer adjoints before
            # propagating them.)

            g = sparse_W_copy - sparse.finalize()
            sparse.w[:] = sparse_W_copy
            dense.w[:] = dense_W_copy
            [keys] = np.nonzero(g)
            fdcheck(func, sparse.w, g, keys)

        # figure out what the gradient step must have been.
        g = dense_W_copy - dense.w   # updater is descent

        sparse.w[:] = sparse_W_copy
        dense.w[:] = dense_W_copy
        c = fdcheck(func, dense.w, g)
        assert c.pearson >= 0.999999
        assert c.max_err <= 1e-8
        assert np.allclose(c.expect, c.got)
        print '[test gradient]:', colors.light_green % 'pass'

    def test_overfitting(self, T, y=None):
        D = 100
        groups = []
        dense = OnlineProx(groups, self.H, C=0, L=2, eta=1.0, fudge=1)
        dense.w[:] = np.random.uniform(-1, 1, size=dense.d)

        sparse = LazyRegularizedAdagrad(D*self.A, C=0, L=2, eta=1.0, fudge=1)
        sparse.w[:] = np.random.uniform(-1, 1, size=sparse.d)

        x = MockInstance(T, self.A, D=D, K=5, y=y)

        print
        #print '[test overfitting]'
        for _ in range(10):
            S = ScoringModel(x, self.A, self.feature_backoff, sparse, dense)
            self.gradient(T, x.tags, S)
            S.backprop()
            y = self.predict(x.N, S)
            #print 'obj: %g, acc: %.2f' % (self.objective(T, x.tags, S),
            #                              (y==x.tags).mean())

        y = self.predict(x.N, S)
        assert (y==x.tags).all()
        print '[test overfitting]', colors.light_green % 'pass'


def test_stateful(sigma1, C1, sigma2, C2):
    "Test that update method isn't stateful for some unintended reason."
    M1 = VoCRF(sigma1, C1)   # use set 1
    M2 = VoCRF(sigma2, C2)   # use set 2
    M1.update(sigma2, C2)    # use update method M1
    # Check that all attributes are equal.
    assert M1.S == M2.S
    assert M1.A == M2.A
    assert M1.C == M2.C
    assert M1.states == M2.states
    assert M1.sigma == M2.sigma
    assert np.allclose(M1.backoff, M2.backoff)
    assert np.allclose(M1.backoff_a, M2.backoff_a)
    assert np.allclose(M1.transition, M2.transition)
    assert M1.outgoing == M2.outgoing


def test():
    #np.random.seed(0)

    # The following example include multiple backoffs.
    sigma1 = 'abcdefg'
    C1 = map(tuple, [
        'aaa',
        'bba',
    ])

    # The folling example is a conventional first-order tagger. It does not use
    # failure transition. I've added this test, as it was useful for debugging.
    sigma2 = 'ab'
    C2 = map(tuple, [
        'aa',
        'ab',
        'ba',
        'bb',
    ])

    test_stateful(sigma1, C1, sigma2, C2)

    # Small gradient example on lazy and eager EdgeScore modules.
    T = 10
    print
    FailureArcTest(sigma1, C1).test_gradient(T)
    print
    LazyFailureArcTest(sigma1, C1).test_gradient(T)

    # Note: In order for this test case to pass everytime we need a rich enough
    # model to pick out any (randomly chosen) target output. This depends on the
    # model structure. Failure arcs make it tricky to pick out specific examples
    # because they don't get features.
    LazyFailureArcTest(sigma2, C2).test_overfitting(T)

    M = LazyFailureArcTest(sigma1, C1)
    M.test_overfitting(6, y=np.array(M.sigma.map('bbabba'), dtype=np.int32))
    M.test_overfitting(6, y=np.array(M.sigma.map('aaabba'), dtype=np.int32))


if __name__ == '__main__':
    test()
