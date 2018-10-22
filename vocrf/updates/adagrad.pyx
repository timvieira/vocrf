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

import numpy as np

cdef class RegularizedAdagrad:
    """
    Proximally regularized adagrad.
    """

    def __init__(self, int d, int L, double C, double eta = 0.1, double fudge = 1e-4):
        self.L = L
        self.d = d
        self.fudge = fudge
        self.q = np.zeros(d, dtype=np.double) + fudge
        self.w = np.zeros(d, dtype=np.double)
        self.C = C
        self.eta = eta
        self.etaC = eta*C
