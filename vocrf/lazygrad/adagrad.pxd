#!python
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++

cdef class LazyRegularizedAdagrad:

    cdef public:
        double[:] w   # weight vector
        double[:] q   # sum of squared weights
        double eta    # learning rate (assumed constant)
        double C      # regularization constant
        int[:] u      # time of last update
        int L         # regularizer type in {1,2}
        int d         # dimensionality
        double fudge  # adagrad fudge factor paramter
        int step      # time step of the optimization algorithm (caller is
                      # responsible for incrementing)

    cdef inline double catchup(self, int k) nogil
    cdef inline void update_active(self, int k, double g) nogil

    cpdef update(self, int[:] keys, double[:] vals)
    cdef inline void _update(self, int[:] keys, double[:] vals) nogil

    cpdef double dot(self, int[:] keys)
    cdef inline double _dot(self, int[:] keys) nogil
