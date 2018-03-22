from __future__ import division

import cPickle
import numpy as np
from path import Path

from arsenal import colors, iterview
from arsenal.math import compare
from arsenal.fsutils import mkdir
from arsenal.iterextras import groupby2

from vocrf.model import VoCRF
from vocrf.score import ScoringModel
from vocrf.updates.spom import OnlineProx
from vocrf.pos.instance import MAGIC
from vocrf.util import prefix_closure, last_char_sub_closure
from lazygrad.adagrad import LazyRegularizedAdagrad


class ActiveSet(VoCRF):

    def __init__(self, corpus, Y, train, dev, initial_contexts,
                 outer_iterations, inner_iterations,
                 group_budget, regularizer, allowed_contexts, dump,
                 no_failure_arcs=0):

        self.no_failure_arcs = no_failure_arcs   # if true, runs model with last-char subst closure.

        # Create initial pattern set.
        VoCRF.__init__(self, Y, initial_contexts)

        self.dump = None
        if dump is not None:
            self.dump = Path(dump)
            mkdir(self.dump)

        self.corpus = corpus
        self.dev_best = -np.inf

        # the set of allowed contexts must be prefix closed to make sense.
        self.allowed_contexts = None
        if allowed_contexts is not None:
            self.allowed_contexts = set(prefix_closure(allowed_contexts))

        self.train = train
        self.dev = dev

        # max number of higher-order features =
        #              budget        [green nodes - the max number of 'active' contexts at any time]
        #  x       extensions = |Y|  [yellow nodes - a little room to grow]
        #  x number of labels        [because that's how we encode features]   XXX: I think this is an overestimate we want |states| x |labels|
        self.H = max(group_budget * len(Y), len(self.C)) * self.A
        self.D = MAGIC * self.A

        self.group_budget = group_budget
        self.regularizer = regularizer / len(self.train)

        L = 2 if regularizer > 0 else -1
        self.sparse = LazyRegularizedAdagrad(self.D, L=L, C=self.regularizer)
        self.dense = OnlineProx(self.group_structure(), self.H, L=L, C=self.regularizer)

        self.inner_iterations = inner_iterations
        self.outer_iterations = outer_iterations

        self.log = []

#    def ideal_runtime(self, w):
#        return len(prefix_closure({c for c in self.C if w[self.context_feature_id(c)] != 0}))

#    def check_L0_group_norm_proxy(self, dense):
#        dense.update_group_norm()
#        active = np.asarray(dense.group_norm) > 0
#        group_size = np.asarray(dense.group_size)
#        v = group_size[active].sum()
#        ideal = self.ideal_runtime(dense.w)
#        assert v == ideal, 'ideal %g, got %g' % (ideal, v)

    def update(self, sigma, C):
        if self.no_failure_arcs:
            C = set(prefix_closure(C))
            C.update((a,) for a in sigma)
            b4 = len(C)
            C = set(last_char_sub_closure(sigma, C))
            C.add(())
            print '[last-char closure] before: %s, after: %s' % (b4, len(C))
        return VoCRF.update(self, sigma, C)

    def active_features(self, verbose=1):

        # XXX: We probably don't want to do this here. However, we *do* run it
        # here so that the active set is guaranteed to be the right size.
        if self.group_budget is not None:
            self.dense.prox_budget(self.group_budget)

        active = [c for c in self.C if self.dense.w[self.context_feature_id(c)] != 0]

        #assert len(active) == np.sum(w != 0), 'active %s, nonzero %s' % (len(active), np.sum(w != 0))
        #self.check_L0_group_norm_proxy(self.dense)

        if verbose:
            print '%s: %s out of %s' % (colors.yellow % 'active', len(active), len(self.C)),
            B = groupby2(active, len)
            print '(budget %s, sizes %s)' % (self.group_budget,
                                             ', '.join('%s: %s' % (z, len(B[z])) for z in sorted(B)))

        return active

    def active_set(self):
        for outer in xrange(1, self.outer_iterations+1):
            print
            print colors.green % '====================='
            print colors.green % 'Outer %s' % outer

            self.inner_optimization(self.inner_iterations)

            if outer != self.outer_iterations:
                print
                print colors.yellow % 'Grow %s' % outer

                # old feature index
                old = {c: self.context_feature_id(c) for c in self.C}
                w = self.dense.w.copy()
                q = np.array(self.dense.q, copy=1)

                TEST_EXPECT = 0

                if TEST_EXPECT:
                    # Record expectations under previous model. Technically,
                    # this is observed-expected features.
                    predictions = []
                    for x in self.train:
                        S = ScoringModel(x, self.A, self.feature_backoff, self.sparse, self.dense)
                        self.gradient(x.N, x.tags, S)   # don't backprop thru scoring model because we don't change the parameters.
                        predictions.append({k: S.d_dense[i] for k,i in old.iteritems()})

                # "Grow" Z by extending active features with on more character.
                active = self.active_features()

                # Heuristic: Use an intelligent guess for 'new' q values in the
                # next iterations.
                #
                # This improves active set's ability to monotonically improve
                # after growing. Otherwise, adagrad will update too aggressively
                # compared to the sensible alternative of start at the last seen
                # value (if possible) or at the fudge value.
                #
                # In other words, new features get huge learning rates compared
                # to existing ones. Features that used to exist also get pretty
                # big learning rates too. This is because adagrad learning rates
                # decrease quickly with time as they are 1/sqrt(sum-of-squares).
                #
                # I found that guessing the mean q works better than min or max.
                self.dense.w[:] = 0
                self.dense.q[:] = q.mean()

                # Grow active contexts to the right.
                cc = {p+(y,) for p in active for y in self.sigma}

                ####
                # Note that just because we extended a bunch of active elements
                # by all elements of sigma, this does not mean that we are
                # last-character closed.
                #
                # Feel free to check via the following (failing) assertion
                #
                #   assert set(prefix_closure(cc)) == set(last_char_sub_closure(self.sigma, prefix_closure(cc)))
                #
                # The reason is that some elements go to zero and, thus, get
                # pruned. This is the same reason why `active` is not
                # automatically prefix closed.

                ####
                # Is the growing set prefix closed by construction?
                #
                # No. The grown set is also not prefix closed either because
                # it's possible for a parent to be zero with nonzero children.
                #
                # Here is an assertion that will fail.
                #
                # assert set(prefix_closure(cc)) == set(cc)
                #
                #cc = set(prefix_closure(cc))

                ####
                # XXX: In general, we probably do not want to do last-char-sub
                # closure. I've added it in because it seems to help use
                # more-closely preserve the distribution after manipulating the
                # active set.
                #cc = set(last_char_sub_closure(self.sigma, cc))

                # Filter active set by allowed-context constraints, if supplied.
                if self.allowed_contexts:
                    cc &= set(self.allowed_contexts)

                # Update DFA and group lasso data structures.
                self.update(self.sigma, cc)
                self.dense.set_groups(self.group_structure())
                print colors.yellow % '=> new', '|C| = %s' % len(self.C)

                # Copy previous weights
                for c in self.C:
                    i = self.context_feature_id(c)
                    if c in old:
                        o = old[c]
                        self.dense.w[i] = w[o]
                        self.dense.q[i] = q[o]

                if 0:
                    print
                    print colors.light_red % 'is accuracy the same???????'
                    self.after_inner_pass()
                    print colors.light_red % '^^^^^^^^^^^^^^^^^^^^^^^^^^^'
                    print

                if TEST_EXPECT:
                    # DEBUGGING: check that expections match
                    #
                    # I'm not sure this test is implemented perfectly because we
                    # need to compute the expected value of all the old features
                    # under the new model.
                    #
                    # We get away using the new model because it has backoff
                    # features.
                    #
                    # In the case of a unigram model (order-0 model), this test
                    # fails. Why? are the unigrams used incorrectly?
                    #
                    new = {c: self.context_feature_id(c) for c in self.C}

                    for x, want in zip(self.train, predictions):
                        S = ScoringModel(x, self.A, self.feature_backoff, self.sparse, self.dense)
                        self.gradient(x.N, x.tags, S)    # don't backprop thru scoring model because we don't change the parameters.

                        # just check on *old* features.
                        E = {k: 0 for k in want}
                        E.update({k: S.d_dense[new[k]] for k in want if k in new})

                        # XXX: filter down to features in both vectors, I guess?
                        E = {k: v for k, v in E.iteritems() if k in new}
                        want = {k: v for k, v in want.iteritems() if k in new}

                        c = compare(want, E, verbose=1)

                        if c.cosine < .99:
                            c.show()

    def inner_optimization(self, iterations, prox_every=25):
        budget = self.group_budget
        for t in xrange(iterations):
            print
            np.random.shuffle(self.train)
            for x in iterview(self.train, colors.green % 'Pass %s' % (t+1)):

                S = ScoringModel(x, self.A, self.feature_backoff, self.sparse, self.dense)
                self.gradient(x.N, x.tags, S)
                S.backprop()

                if budget is not None and self.sparse.step % prox_every == 0:
                    self.dense.prox_budget(budget)

                self.sparse.step += 1

            assert np.isfinite(self.sparse.w).all()
            assert np.isfinite(self.dense.w).all()

            # make sure to call prox udate before finishing this pass. This will
            # keep the number of features within the budget.
            if budget is not None:
                self.dense.prox_budget(budget)

            self.after_inner_pass()

    def predict(self, x):
        "Predict tags for `Instance x`."
        S = ScoringModel(x, self.A, self.feature_backoff, self.sparse, self.dense)
        y = super(ActiveSet, self).predict(x.N, S)
        return self.sigma.lookup_many(y)

    def after_inner_pass(self, verbosity=1):
        "Called after each pass of inner optimization."
        train = self.corpus.evaluate(self.predict, self.train, 'train', verbosity=verbosity)
        dev = self.corpus.evaluate(self.predict, self.dev, 'dev', verbosity=verbosity)
        self.active_features()
        if not self.dev: dev = train   # train is dev, when there is no dev.

        self.log.append({
            'epoch': 1+len(self.log),
            'train': train,
            'dev': dev,
            'contexts': {c: self.dense.w[self.context_feature_id(c)] for c in self.C if c},
        })

        with file(self.dump / 'log.pkl', 'wb') as f:
            cPickle.dump(self.log, f)

        if dev > self.dev_best:          # save model only when dev performance increases
            print colors.light_green % 'New best!'
            if self.dump is not None:
                with file(self.dump / 'state.pkl', 'wb') as f:
                    cPickle.dump(self.__getstate__(), f)
                print '>>> wrote model to', f.name

    def __getstate__(self):
        return {'H': self.H,
                'D': self.D,
                'A': self.A,
                'C': self.C,
                'sigma': self.sigma,
                'sparse': np.array(self.sparse.w),
                'dense': np.array(self.dense.w),
                'groups': self.group_structure()}

#    @classmethod
#    def load(cls, pkl):
#        with file(pkl) as f:
#            s = cPickle.load(f)
#
#        A = s['A']
#        feature_backoff = s['feature_backoff']
#        sparse = LazyRegularizedAdagrad(s['D'], L=0, C=0)
#        sparse.w[:] = s['sparse']
#        dense = OnlineProx(s['groups'], s['H'], L=0, C=0)
#        dense.w[:] = s['dense']
#        sigma = s['sigma']
#
#        m = VoCRF(sigma, s['C'])
#
#        def predict(x):
#            "Predict tags for `Instance x`."
#            S = ScoringModel(x, A, feature_backoff, sparse, dense)
#            y = m.predict(x.N, S)
#            return sigma.lookup_many(y)
#
#        return predict
