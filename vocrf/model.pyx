#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: cdivision=True
#cython: infertypes=True
#cython: wraparound=False
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: boundscheck=False
#cython: nonecheck=False
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

cimport cython
import numpy as np
from numpy import zeros_like, zeros, empty, int32
from libc.math cimport exp, log, log1p, sqrt
from numpy cimport ndarray

from vocrf.score cimport EdgeScore
from vocrf.sparse cimport SparseBinaryVector
from vocrf.util import longest_suffix_in, prefix_closure, groups, suffixes

from arsenal.alphabet import Alphabet
from lazygrad.adagrad cimport LazyRegularizedAdagrad

from libcpp.vector cimport vector

ctypedef (int,int) intpair


cdef class VoCRF(object):
    """
    Variable-order conditional random field (VoCRF) with failure arcs.
    """

    cdef public:
        object C
        object sigma
        object states
        int[:] backoff
        int[:,:] backoff_a
        int[:,:] transition
        vector[vector[intpair]] outgoing
        int[:] feature_backoff
        int S, A

    def __init__(self, sigma, raw_contexts):
        self.update(sigma, raw_contexts)

    @cython.boundscheck(True)
    @cython.wraparound(True)
    cpdef int context_feature_id(self, tuple c):
        assert c    # no empty context
        s = c[:-1]
        a = c[-1]
        return self.states[s]*self.A + self.sigma[a]

    def group_structure(self):
        G = groups(self.C)
        return list(map(lambda g: list(sorted(map(self.context_feature_id, g))), G.values()))

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def update(self, sigma, raw_contexts):
        """Create machine.

        Will take prefix closure of `raw_contexts`. It will not take
        "last-character substitution" closure because that is handled by failure
        arcs.

        """

        C = set(prefix_closure(raw_contexts))
        C.update((a,) for a in sigma)

        if () in C:
            C.remove(())

        # Set of states is C minus the last character.
        states = {c[:-1] for c in C}

        # Backoff to the longest proper suffix in C.
        # you can't be your own backoff. (Note that () backs off to () below.)
        backoff = {c: longest_suffix_in(c[1:], states) for c in C}
        backoff[()] = ()

        feature_backoff = {}
        for c in C:
            feature_backoff[c] = ()
            for b in reversed(sorted(suffixes(c), key=len)):   # longest first
                if b != c and b in C:   # proper suffix, which is also in C.
                    feature_backoff[c] = b
                    break

        # Transition function.
        arcs = {s: [] for s in states}
        transition = {}
        for c in C:
            if len(c) == 0:
                continue
            s, a = c[:-1], c[-1]
            arcs[s].append(a)
            # Figure out where this action takes us.
            transition[s, a] = longest_suffix_in(c, states)

        # empty state has explicit transitions for each element of sigma.
        # TODO: This should be a default (rho) arc, right?
        for y in sigma:
            if y not in arcs[()]:
                arcs[()].append(y)

        self._update(sigma, C, backoff, states, arcs, transition, feature_backoff)

    def _update(self, sigma, C, backoff, states, arcs, transition, feature_backoff):
        cdef:
            vector[intpair] tmp
            intpair p
            int si, spi, ai

        self.C = C
        self.S = len(states)   # number of states
        self.A = len(sigma)    # size of alphabet (action space)

        self.sigma = Alphabet.from_iterable(sigma)
        # sorting of `states` is important because it puts backoff states in the correct order.
        self.states = Alphabet.from_iterable(sorted(states, key=lambda x: (len(x), x)))
        assert self.states[()] == 0
        self.sigma.freeze()
        self.states.freeze()

        self.feature_backoff = np.full(self.S*self.A, -1, dtype=np.int32)
        for c in C:
            b = feature_backoff[c]
            if b:
                assert b in C, b
                self.feature_backoff[self.context_feature_id(c)] = self.context_feature_id(b)

        self.transition = np.full((self.S, self.A), -1, dtype=np.int32)
        self.backoff_a = np.full((self.S, self.A), -1, dtype=np.int32)  # Precompute where a state backs-off to on a given arc.
        self.backoff = np.full(self.S, -1, dtype=np.int32)

        self.outgoing.clear()   # makes sure we clear out old stuff.
        for _ in range(self.S):
            self.outgoing.push_back(tmp)

        for s in self.states:
            si = self.states[s]
            for a in arcs[s]:
                sp = transition[s, a]
                ai = self.sigma[a]
                spi = self.states[sp]
                p = (ai, spi)
                self.outgoing[si].push_back(p)
                self.transition[si, ai] = spi

            self.backoff[si] = self.states[backoff[s]]

            for a in self.sigma:
                b = backoff[s]
                while a not in arcs[b]:
                    b = backoff[b]
                ai = self.sigma[a]
                self.backoff_a[si, ai] = self.states[b]

    cpdef double observed(self, int T, int[:] y, EdgeScore S, int with_grad=1):
        cdef int t, s, a, sp
        cdef double score
        score = 0.0
        s = 0
        for t in range(T):
            a = y[t]
            sp = self.transition[s, a]
            if sp < 0:
                s = self.backoff_a[s,a]     # go to backoff state.
                sp = self.transition[s, a]  # transition from backoff state
            score += log(S.edge(t, s, a))
            if with_grad:
                S.d_edge(t, s, a, 1.0, -1.0)
            s = sp
        return -score

    cpdef double objective(self, int T, int[:] y, EdgeScore S):
        "Negative log-likelihood."
        cdef double a = self.observed(T, y, S, with_grad=0)
        cdef double[:,:] A = self.forward(T, S)
        for t in range(T+1):
            a += log(A[t,self.S])
        return a

    cpdef double gradient(self, int T, int[:] y, EdgeScore S):
        "Propagate gradient of objective."
        self.observed(T, y, S)
        cdef double[:,:] A = self.forward(T, S)
        self.backward(T, S, A)

    cdef double[:,:] backward(self, int T, EdgeScore S, double[:,:] A):
        cdef:
            double[:,:] v
            double w, u, rescale
            int s, t, b, bp
            intpair e
        v = np.zeros((T+1, self.S+1))
        for s in range(self.S):
            v[T, s] = 1.0
        for t in reversed(range(1, T+1)):
            rescale = A[t, self.S]
            for s in range(self.S):          # uses special sorting
                b = self.backoff[s]
                if s != b:
                    v[t-1, s] += v[t-1, b]
                for e in self.outgoing[s]:
                    a, sp = e
                    w = S.edge(t-1, s, a)
                    u = v[t, sp] / rescale
                    v[t-1, s] += w * u
                    S.d_edge(t-1, s, a, w, A[t-1, s] * u)
                    b = self.backoff_a[s, a]
                    if s != b:
                        bp = self.transition[b, a]
                        w = S.edge(t-1, b, a)
                        u = v[t, bp] / rescale
                        v[t-1, s] -= w * u
                        S.d_edge(t-1, b, a, w, -A[t-1, s] * u)
        return v

    cdef double[:,:] forward(self, int T, EdgeScore S):
        cdef:
            double[:,:] v
            double w
            int t, s, b, a, sp, bp
            intpair e

        v = np.zeros((T+1, self.S+1))
        v[0, 0] = 1.0   # 0 is guaranteed to be empty state.

        for t in range(1, T+1):
            self.rescale_t(v, t-1)

            for s in reversed(range(self.S)):
                b = self.backoff[s]
                if s != b:
                    v[t-1, b] += v[t-1, s]  # don't rescale here b/c it's within a time-step.
                for e in self.outgoing[s]:
                    (a, sp) = e
                    v[t, sp] += v[t-1, s] * S.edge(t-1, s, a)
                    b = self.backoff_a[s, a]
                    if s != b:
                        bp = self.transition[b, a]
                        v[t, bp] -= v[t-1, s] * S.edge(t-1, b, a)

        self.rescale_t(v, T)

        return v

    cdef inline double rescale_t(self, double[:,:] v, int t):
        """Rescale chart at the time `t` inplace while recording normalization
        constant.

        This method stores the rescaling coefficient in the array just passed
        the actualy values at position `v[t,S]`, where S in the number of
        states. Thus, a  (T x |S|+1) array is required.

        The rescaling coefficient we use is the sum of the original array's
        values. This choice makes computing the overall normalizing constant
        easier.

        """
        cdef double rescale
        cdef int s

        # rescale parameter is equal the total mass at time t.
        # Note: We could chose something else, e.g., the max value.
        rescale = 0.0
        for s in range(self.S):
            rescale += v[t, s]

        # store the rescaling parameter in this fake state to avoid passing
        # arround and allocating a separate array.
        v[t, self.S] = rescale

        # rescale
        for s in range(self.S):
            v[t, s] /= rescale

        return rescale

    cpdef ndarray[int,ndim=1] predict(self, int T, EdgeScore S):
        return self.mbr(T, S)

    cpdef mbr(self, int T, EdgeScore S):
        "Minimum Bayes risk decoding under hamming loss."
        # Implementation note: We get marginal probabilities using the adjoints
        # at each edge, which computes d \log Z / d weight(t, a) = p(y[t] = a).
        #
        # Note: Unlike the full gradient computation, we don't call backprop on
        # `S` for efficiency because we don't need it.
        self.backward(T, S, self.forward(T, S))
        return np.argmax(S.d_U, axis=1)

    cpdef ndarray[int,ndim=1] buggy_viterbi(self, int T, EdgeScore S):
        """The algorithm implemented here subtle bug.

        Viterbi with failure arcs is a little nontrivial because max doesn't
        have an inverse, so the subtraction trick doesn't quite work. There are
        workarounds which require something like a maxheap to (efficiently)
        support (weak) subtraction of overridden aggregands.

        """
        cdef:
            double[:,:] v
            double w, score
            int t, s, b, a, sp, bp
            intpair e
            int[:,:,:] B
            int ps, pa, pt
        v = np.zeros((T+1, self.S))
        v[0, 0] = 1.0      # 0 is guaranteed to be empty state.

        B = np.full((T+1, self.S, 3), -1, dtype=np.int32)
        B[0, 0, 0] = 0   # state
        B[0, 0, 1] = 0   # action
        B[0, 0, 1] = 0   # time

        for t in range(1, T+1):
            for s in reversed(range(self.S)):    # XXX: is this the right order?
                for e in self.outgoing[s]:
                    (a, sp) = e
                    score = v[t-1, s] * S.edge(t-1, s, a)
                    if score > v[t, sp]:
                        v[t, sp] = score
                        B[t, sp, 0] = s
                        B[t, sp, 1] = a
                        B[t, sp, 2] = t-1

                # XXX: Do we need to backoff multiple times here?
                #
                # take the free edge to back off state
                sp = self.backoff[s]
                score = v[t, s]
                if score > v[t, sp]:
                    v[t, sp] = score
                    B[t, sp, 0] = s
                    B[t, sp, 1] = -1
                    B[t, sp, 2] = t

        cdef int[:] best = np.full(T+1, -1, dtype=np.int32)

        # find the best final state.
        cdef double val = 0.0
        cdef (int,int,int) arg = (-1,-1,-1)
        for s in reversed(range(self.S)):
            if v[T, s] > val:
                val = v[T, s]
                arg = (s,-1,T)

        while arg[2] > -1:
            s, a, t = arg
            if a >= 0:
                best[t] = a
            ps = B[t,s,0]
            pa = B[t,s,1]
            pt = B[t,s,2]
            arg = (ps, pa, pt)

        return np.asarray(best[:T])
