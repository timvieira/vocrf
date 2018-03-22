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

from libc.math cimport sqrt


cdef inline double sign(double x) nogil:
    return -1 if x < 0 else +1


cdef inline double abs(double x) nogil:
    return -x if x < 0 else x


cdef class RegularizedAdagrad:
    """
    Proximally regularized adagrad.
    """

    cdef public:
        double C, eta, fudge, etaC
        int L, d
        double[:] w, q

    cdef inline void update(self, double[:] g) nogil:
        cdef double d, z, sq
        cdef int k
        for k in range(self.d):
            sq = sqrt(self.q[k])
            self.q[k] += g[k]*g[k]
            if self.L == 2:
                self.w[k] = (self.w[k]*sq - self.eta*g[k])/(self.etaC + sq)
            elif self.L == 1:
                z = self.w[k] - self.eta*g[k]/sq
                d = abs(z) - self.etaC/sq
                self.w[k] = sign(z) * max(0, d)
            else:
                self.w[k] -= self.eta*g[k]/sq
