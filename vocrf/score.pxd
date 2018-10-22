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

cdef class EdgeScore:
    "Base case for edge scoring models."
    cdef double edge(self, int t, int s, int a)
    cdef void d_edge(self, int t, int s, int a, double value, double adjoint)
