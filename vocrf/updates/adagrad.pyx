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
