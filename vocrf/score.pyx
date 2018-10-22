#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: wraparound=False, boundscheck=False
#cython: nonecheck=False
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

import numpy as np
from libc.math cimport exp
from lazygrad.adagrad cimport LazyRegularizedAdagrad
from vocrf.updates.adagrad cimport RegularizedAdagrad
from vocrf.sparse cimport SparseBinaryVector
from collections import defaultdict


cdef class EdgeScore:
    "Base case for edge scoring models."
    cdef double edge(self, int t, int s, int a):
        return np.nan
    cdef void d_edge(self, int t, int s, int a, double value, double adjoint):
        pass


cdef class BasicEdgeScoreDict(EdgeScore):
    """
    Simple EdgeScore class for debugging.
    """

    cdef public:
        object weights, d_weights

    def __init__(self, weights):
        self.weights = weights
        self.d_weights = defaultdict(float)

    cdef double edge(self, int t, int s, int a):
        return exp(self.weights[t,s,a])

    cdef void d_edge(self, int t, int s, int a, double value, double adjoint):
        self.d_weights[t,s,a] += value * adjoint


cdef class RandomEdgeFeatures(EdgeScore):
    """
    Simple EdgeScore class for debugging.
    """

    cdef public:
        object weights, d_weights, F, E, d_E

    def __init__(self, weights, F):
        self.weights = weights
        self.d_weights = np.zeros_like(weights)
        self.d_E = defaultdict(float)
        self.E = {}
        self.F = F

    cdef double edge(self, int t, int s, int a):
        v = exp(sum([self.weights[k] for k in self.F[t,s,a]]))
        self.E[t,s,a] = v
        return v

    cdef void d_edge(self, int t, int s, int a, double value, double adjoint):
        self.d_E[t,s,a] += value * adjoint
        for k in self.F[t,s,a]:
            self.d_weights[k] += value * adjoint


# TODO: Support backedoff tags (probably needed for morphological tagging)
cdef class ScoringModel(EdgeScore):
    """
    Factored scoring model.

    1. Unary factors for context features at token `t`

    2. Higher-order features for tag context as well as backoff version of the
       tag context.

    """

    cdef public:
        object x
        int A, N
        LazyRegularizedAdagrad sparse
        RegularizedAdagrad dense
        double[:,:] U, d_U
        int[:] backoff
        double[:] d_dense

    def __init__(self, object x, int A, int[:] backoff,
                 LazyRegularizedAdagrad sparse,
                 RegularizedAdagrad dense):
        self.x = x
        self.A = A
        self.N = self.x.N
        self.backoff = backoff
        self.sparse = sparse
        self.dense = dense
        self.U = self._unary_potentials()   # precompute unary potentials
        self.d_U = np.zeros_like(self.U)
        self.d_dense = np.zeros(self.dense.d)

    cdef double[:,:] _unary_potentials(self):
        """
        Compute unary potentials -- theses are (non-stationary) features which only
        depend on the label at time `t` and the 'properties' of the context
        around the tag at `t`.
        """
        cdef SparseBinaryVector f
        cdef int a, t, k
        cdef double w
        cdef double[:,:] U = np.zeros((self.N, self.A))
        for t in range(self.N):
            f = self.x.properties[t]
            for a in range(self.A):
                w = 0.0
                for i in range(f.length):
                    # conjunction with just the label
                    k = f.keys[i]*self.A + a
                    w += self.sparse.catchup(k)
                U[t,a] = w
        return U

    cdef double edge(self, int t, int s, int a):
        cdef:
            int k
            double w

        # TODO: We should change this up. As it is, we allocate too many
        # features! Specifically, the set of features is closed under
        # last-character substitution!
        #
        # I think the way to solve this (as well as another problem to do with
        # features shifting names across active set iterations!) is to have a
        # dictionary mapping state-action pairs to features indices.
        #
        #   ** A features should only be requested if its indeed in
        #      C. Otherwise, we have bug in the failure arc code. ***

        # transition feature weight
        k = s*self.A + a
        w = self.dense.w[k]

        # weight of all backoffs from transition features
        k = self.backoff[k]
        while k >= 0:
            w += self.dense.w[k]
            k = self.backoff[k]

        return exp(w + self.U[t,a])

    cdef void d_edge(self, int t, int s, int a, double value, double adjoint):
        "Backpropagate value into scoring module."
        cdef int k

        # transition feature weight
        k = s*self.A + a
        self.d_dense[k] += value*adjoint

        # weight of all backoffs from transition features
        k = self.backoff[k]
        while k >= 0:
            self.d_dense[k] += value*adjoint
            k = self.backoff[k]

        # other features
        self.d_U[t, a] += value*adjoint

    cpdef backprop(self):
        "Propagates any buffered updates to their destination."
        cdef SparseBinaryVector f
        cdef int i, t, a, k
        for t in range(self.N):
            f = self.x.properties[t]
            for a in range(self.A):
                for i in range(f.length):
                    k = f.keys[i]*self.A + a   # conjunction with the label
                    self.sparse.update_active(k, self.d_U[t, a])
        # propagate dense adjoints to the parameters
        self.dense.update(self.d_dense)
