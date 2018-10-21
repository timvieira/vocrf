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
#distutils: extra_compile_args = ["-std=c++11"]

import numpy as np
from vocrf.updates.adagrad cimport RegularizedAdagrad
from libcpp.vector cimport vector
from libc.math cimport sqrt
from libc.stdint cimport int32_t, int64_t

cdef inline double inf = float('inf')


cdef class OnlineProx(RegularizedAdagrad):
    """
    Sparse proximal online updates for overlapping group lasso.
    """

    cdef public:
        vector[vector[int]] groups
        int G
        cdef double[:] group_norm
        cdef int[:] group_size

    def __init__(self, list groups, int d, int L, double C, double eta = 0.1, double fudge = 1e-4):
        self.set_groups(groups)
        super(OnlineProx, self).__init__(d=d, L=L, C=C, eta=eta, fudge=fudge)

    def set_groups(self, list groups):
        self.G = len(groups)
        self.groups.clear()    # clear out old version!
        for gl in groups:
            self.groups.push_back(gl)
        self.group_norm = np.zeros(self.G)
        self.group_size = np.zeros(self.G, dtype=np.int32)

        for i in range(self.G):
            g = self.groups[i]
            # [2016-10-28 Fri] The budget heuristic described in Martins+2011
            # paper uses the following definition for group size.
            #
            #self.group_size[i] = g.size()
            #
            # In the VoCRF setting, we want to minimize |C|, while taking prefix
            # closure into account (hence tree0-structured groups). We don't use
            # Margins-style group size because it count elements of C many times
            # (once for each group it participates in) rather than just once.
            #
            self.group_size[i] = 1

    cpdef update_group_norm(self):
        "Update group norms."
        cdef double z
        for i in range(self.G):
            g = self.groups[i]
            z = 0.0
            for k in g:
                z += self.w[k]*self.w[k]
            self.group_norm[i] = sqrt(z)

    cpdef double find_threshold(self, double budget):
        """Figure out the threshold (sigma) to make the number of active groups
        equal to <= budget.

        NOTE: This method assumes group norms are up-to-date, i.e.,
        call `update_group_norm` before calling this method.

        This value of sigma guarantees that the total size of the number of
        active groups is <= budget.

        Note that when groups overlap we will be a little too aggressive with
        this thresholding procedure.

        Martins+'11 give some convergence criteria for this heuristic
        algorithm. Essentially, it boils down to whether or not the sequence of
        sigma values /eventually/ stabilizes (the asymptotic average converges)
        to a constant value.

        """
        cdef int j
        cdef double z
        cdef int64_t[:] S

        self.update_group_norm()

        # TODO: use an inplace sorting algorithm that reuses memory
        S = np.argsort(self.group_norm)[::-1]    # group indices by decreasing norm.

        # Find the threshold that will keep the total size within budget.
        z = 0.0
        j = -1
        while j < self.G-1:
            z += self.group_size[S[j+1]]
            if z > budget:
                break
            j += 1

        # handle corner cases.
        if j == -1:           # everything must go.
            return inf
        elif j >= self.G-1:   # keep everything.
            return 0
        else:
            return (self.group_norm[S[j]] + self.group_norm[S[j+1]]) / 2

    cpdef void prox_budget(self, int budget):
        """Apply online proximal update for group lasso where the threshold is based on
        the budget heuristic.

        """
        self.__prox(self.find_threshold(budget))

    cpdef void prox_threshold(self, double threshold):
        "Apply online proximal update for group lasso with numerical `threshold`."
        self.update_group_norm()
        self.__prox(threshold)

    cdef void __prox(self, double threshold):
        """Apply online proximal update for group lasso.

        Note: Caller must run `update_group_norm`. (Designed this way to avoid
        unnecessarily running `update_group_norm` multiple times.)

        Effectively, groups with norm <= `threshold` go directly to zero, all 
        other groups shrink by the amount `threshold`.

        This procedure is only exact if groups do not overlap. However, there
        is a relatively good guarantee (in terms of a PAC bound) in the 
        overlapping case described in Martins et al (2011).

        For example, if
          threshold = 1
          group_norm[0] = 0.5
          group_norm[1] = 3

        The new group norms will be

          group_norm[0] -> 0.5 <= threshold -> 0
          group_norm[1] -> 3 > threshold    -> 3 - threshold = 2

        This is completely analogous to the L1-prox operator.
        In fact if groups do not overlap and they have one element, 
        it is precisely the same.
        """
        if threshold == 0:
            return
        cdef int i, k
        cdef double rescale
        for i in range(self.G):
            if self.group_norm[i] <= threshold:
                for k in self.groups[i]:
                    self.w[k] = 0
            else:
                rescale = (self.group_norm[i] - threshold) / self.group_norm[i]
                for k in self.groups[i]:
                    self.w[i] *= rescale
