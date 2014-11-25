from __future__ import division
import operator
import numpy as np
from polynomes import *

import matplotlib.pyplot as plt

def calcul_proba(table_proba, Vp, alpha=0.):
    return np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])

# E[ k ^ N]
def esp_min_k_N(k, proba_N):
    return np.dot(np.array(range(k)), proba_N[0][:k]) + k * (1 - sum(proba_N[0][:k]))

def optimize(alpha,p,  K=20):
    Vp = np.array([p**k for k in range(30)])
    Vq = np.array([(1-p)**k for k in range(30)])

    # Loi de N
    proba_N = (1-alpha) * calcul_proba(P_N, Vp) + alpha * calcul_proba(P_N, Vq)

    # Loi de S_k
    proba_S = (1-alpha) *  calcul_proba(P_S_X0_1, Vp) + alpha * calcul_proba(P_S_X0__1, Vq)

    # E[S_k^n]
    E_S = alpha * E_S__1 + (1-alpha) * E_S_1


    # E[ c(k ^ N) + (1+S_{k^N} ]
    tests = [ (esp_min_k_N(k, proba_N) if k < 10 else 0,
               1+E_S[int(10*p)][k] )
              for k in range(K) ]

    A, B = map(np.array, zip(*tests))

    c_range = np.arange(0, 0.3, 0.05)
    plt.subplot(221)
    for c in c_range:
        plt.plot( B + c*A, 'o-', label='$c = %.2f$' % c) 
    plt.legend(loc=2, bbox_to_anchor=(1.2, 1))
    
    plt.ylabel('$E[ c * min(k, N) + (1+S_{min(k, N)}) ]$' )
    plt.xlabel('$k$')

    
    plt.title('$alpha = %d, p = %.2f$' % (alpha, p))
    dir = "graphs/"
    plt.savefig(dir + "alpha = %d and p = %.2f .png" % (alpha, p) )
    plt.clf()

# alpha = P(X0 = -1)
for alpha in (0, 1):    
    for p in range(0, 11):
        optimize(alpha, p=p/10, K = 10)    

