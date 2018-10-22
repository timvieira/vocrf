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
#distutils: extra_compile_args = ["-std=c++11"]

"""
Tree-structured group lasso.
"""
import numpy as np
from arsenal.alphabet import Alphabet
from libc.math cimport sqrt


cdef inline double inf = float('inf')


cdef class Tree(object):
    """ Tree structured group for structured sparse penalty """

    cdef public:
        object z
        list children

    def __init__(self, z):
        self.z = z
        self.children = []

    def add(self, z, depth=0):
        "Adds the pattern z."
        z_head = z[:depth+1]
        found = False
        for child in self.children:
            if z_head == child.z:
                child.add(z, depth+1)
                found = True
                break
        if not found and len(z) == depth+1:
            self.children.append(Tree(z))

    def postorder(self):
        "Post-order traversal of subtree."
        nodes = []
        for child in self.children:
            nodes += child.postorder()
        nodes.append(self)
        return nodes


cdef class TreeProx:
    """
    Group Lasso proximal update for tree-structured groups.

    Implements Alg. 2 of Nelakanti et al., 2013.
    http://www.di.ens.fr/~fbach/anil_emnlp.pdf
    """

    cdef public:
        int arity
        int[:,:] children
        int[:] topo
        dict Z
        int D
        Tree tree
        double[:] eta, gamma   # Allocate space for eta and gamma that we can reuse it.

    def __init__(self, Y):
        self.arity = Y
        self.tree = Tree(())
        self.Z = {(): 0}
        self.D = 1

    def add(self, z):
        """ adds the pattern z """
        if z not in self.Z:
            self.Z[z] = len(self.Z)
        self.tree.add(z)

    cpdef prox(self, double kappa, double[:] w):
        "Proximal Operator, implements Alg 2."
        cdef int t, c, x, h
        cdef double gamma_h
        self.gamma[:] = 0
        self.eta[:] = 1

        assert w.shape[0] == self.D

        # bottom up
        for t in range(self.D):
            x = self.topo[t]
            self.gamma[x] = w[x]*w[x]
            for c in xrange(self.arity):
                h = self.children[x, c]
                if h == -1:
                    break
                self.gamma[x] += self.gamma[h]
            self.eta[x] = max(0, 1-kappa/sqrt(self.gamma[x]))
            self.gamma[x] *= self.eta[x]*self.eta[x]

        # top down
        for t in range(1,self.D+1):
            x = self.topo[self.D-t]
            w[x] *= self.eta[x]
            for c in xrange(self.arity):
                h = self.children[x, c]
                if h == -1:
                    break
                self.eta[h] *= self.eta[x]

    def update_groups(self):
        cdef int t, x

        post = self.tree.postorder()
        topo = [self.Z[z.z] for z in post]

        self.topo = np.array(topo, dtype=np.int32)
        self.D = self.topo.shape[0]

        self.gamma = np.zeros(self.D)
        self.eta = np.zeros(self.D)

        self.children = np.full((self.D, self.arity), -1, dtype=np.int32)
        for node in post:
            x = self.Z[node.z]
            for t, c in enumerate(node.children):
                self.children[x, t] = self.Z[c.z]

    cpdef double group_norm(self, double[:] w):

        cdef int t, c, x, h
        cdef double z

        self.gamma[:] = 0

        # compute group norm in linear time by essentially recursion on the tree.
        for t in range(self.D):
            x = self.topo[t]
            self.gamma[x] = w[x]*w[x]
            for c in range(self.arity):
                h = self.children[x, c]
                if h == -1:
                    break
                self.gamma[x] += self.gamma[h]

        z = 0.0
        for t in range(self.D):
            z += sqrt(self.gamma[t])

        return z

    cpdef double find_threshold(self, double budget, double[:] w):
        cdef int x, h, t
        cdef double[:] size = np.zeros(self.D)

        # compute group norm in linear time by essentially recursion on the tree.
        for t in range(self.D):
            x = self.topo[t]
            self.gamma[x] = w[x]*w[x]
            size[x] = 1
            for c in range(self.arity):
                h = self.children[x, c]
                if h == -1:
                    break
                self.gamma[x] += self.gamma[h]
                size[x] += size[h]

        for t in range(self.D):
            self.gamma[t] = sqrt(self.gamma[t])

        cdef int j
        cdef double z
        cdef long[:] S

        S = np.argsort(self.gamma)[::-1]    # group indices by decreasing norm.

        # Find the threshold that will keep the total size within budget.
        #z = 0.0
        #j = -1
        #while j < self.D-1:
        #    #z += size[j+1]
        #    z += 1
        #    if z > budget:
        #        break
        #    j += 1
        j = int(budget)-1

        # handle corner cases.
        if j == -1:      # everything must go.
            return inf
        elif j >= self.D-1:   # keep everything.
            return 0
        else:
            return (self.gamma[S[j]] + self.gamma[S[j+1]]) / 2
