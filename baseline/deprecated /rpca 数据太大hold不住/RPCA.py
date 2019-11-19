#!/usr/bin/python
# -*- coding:  utf-8 -*-
from __future__ import division, print_function
import numpy as np 

from r_pca import R_pca

'''
Robust Principal Component Analysis (RPCA): 
https://github.com/dganguli/robust-pca
'''
# generate low rank synthetic data


N = 100
num_groups = 3
num_values_per_group = 40
p_missing = 0.2

Ds = []
for k in range(num_groups):
    d = np.ones((N, num_values_per_group)) * (k + 1) * 10
    Ds.append(d)

D = np.hstack(Ds)

# decimate 20% of data 
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < 0.2] = 0
print(D,D.shape)
# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = R_pca(D)
L, S = rpca.fit(max_iter=10000, iter_print=100)
print(L,S,L.shape,S.shape)