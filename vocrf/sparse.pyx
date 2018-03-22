#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=False
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]
#distutils: include_dirs=['/home/timv/anaconda/include/']

import numpy as np
from numpy import asarray, array, empty

cdef class SparseBinaryVector(object):

    def __cinit__(self, keys):
        self.keys = asarray(keys, dtype=np.int32)
        self.length = self.keys.shape[0]

    def __iter__(self):
        return iter(self.keys)

    cpdef double dot(self, double[:] w, int offset=0, int inc=1):
        return self._dot(w, offset=offset, inc=inc)

    cdef double _dot(self, double[:] w, int offset=0, int inc=1) nogil:
        cdef:
            int i
            double x
        x = 0
        for i in range(self.length):
            x += w[self.keys[i]*inc + offset]
        return x

    cpdef pluseq(self, double[:] w, double coeff, int offset=0, int inc=1):
        """
        Compute += update, dense vector ``w`` is updated inplace.

           w += coeff*this

        More specifically the sparse update:

          w[self.keys] += coeff

        """
        self._pluseq(w, coeff, offset=offset, inc=inc)

    cdef void _pluseq(self, double[:] w, double coeff, int offset=0, int inc=1) nogil:
        cdef:
            int i
        for i in range(self.length):
            w[self.keys[i]*inc + offset] += coeff

    def transform_keys(self, int offset, int inc):
        return SparseBinaryVector([self.keys[i]*inc + offset for i in range(self.length)])

    def __repr__(self):
        keys = list(self.keys)
        return 'SparseBinaryVector(%s)' % keys

    def __reduce__(self):
        # can't pickle memoryview slice, but can pickle ndarray
        keys = array(self.keys, dtype=int)
        return (SparseBinaryVector, (keys,), {})

    def __setstate__(self, _):
        pass
