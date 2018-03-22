#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']

cdef class SparseBinaryVector(object):
    cdef readonly int[:] keys
    cdef readonly int length
    cpdef double dot(self, double[:] w, int offset=?, int inc=?)
    cdef double _dot(self, double[:] w, int offset=?, int inc=?) nogil
    cpdef pluseq(self, double[:] w, double coeff, int offset=?, int inc=?)
    cdef void _pluseq(self, double[:] w, double coeff, int offset=?, int inc=?) nogil
