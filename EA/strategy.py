from __future__ import division
import operator
import numpy as np
from polynomes import *

import matplotlib.pyplot as plt

def calcul_proba(table_proba, Vp, alpha=0.):
    # P(. | X0 = -1)
    #proba_1 = np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])
    # P(. | X0 = 1)
    #proba_2 = np.array([ np.array([np.dot(Vq[:len(Q)], Q) for Q in PP]) for PP in table_proba])
    #return proba_1 * alpha +  proba_2 * (1-alpha)
    return np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])

# E[ k ^ N]    
def esp_min_k_N(k, proba_N):
    return np.dot(np.array(range(k)), proba_N[0][:k]) + k * (1 - sum(proba_N[0][:k]))

# P(k < N_tilde | S0_tilde = Sk) 
def proba_k_inf_N_cond_S_k(k, Sk):
    if Sk == -1: return 1
    # P(k < N | S0 = -s-2) = P( k < N_tilde | S0 = s)
    if Sk >= 0: return 1-sum(proba_inv_N[Sk][:k+1])
    return 1-sum(proba_N[-Sk-2][:k+1])

# E[ 1_{k<N} S_k ] = E[ S_k * P(k < N_tilde | S0_tilde = Sk) ]
def esp_1_mult_S_k(k):
    return sum([ j * proba_k_inf_N_cond_S_k(k, Sk=j) * proba_S[k][j+k] for j in range(-k, k+1)]) 

# alpha = P(X0 = -1)

    
def optimize(alpha,p,  K=20):

    Vp = np.array([p**k for k in range(30)])
    Vq = np.array([(1-p)**k for k in range(30)])

    # Loi de N
    proba_N = (1-alpha) * calcul_proba(P_N, Vp) + alpha * calcul_proba(P_N, Vq)

    # Loi de S_k
    proba_S = (1-alpha) *  calcul_proba(P_S_X0__1, Vp) + alpha * calcul_proba(P_S_X0_1, Vq)

    # E[S_k^n]
    E_S = alpha * E_S_1 + (1-alpha) * E_S__1
    
    # E[ c(k ^ N) + (1+S_{k^N} ]
    tests = [ (esp_min_k_N(k, proba_N) if k < 10 else 0,
               1+E_S[int(10*p)][k] )
              for k in range(K) ]

    A, B = map(np.array, zip(*tests))

    #plt.plot( A , 'r*-')
    #plt.plot( B )
    #plt.plot( B + A , 'g*-')
    c_range = [0, 1, 5, 10]

    plt.subplot(221)
    for c in c_range:
        plt.plot( c*B + A, 'o-', label='$c = %.2f$' % c) 
    plt.legend(loc=2, bbox_to_anchor=(1.2, 1))
    
    plt.ylabel('$E[ c * min(k, N) + (1+S_{min(k, N)}) ]$' )
    plt.xlabel('$k$')

    
    plt.title('$alpha = %d, p = %.2f$' % (alpha, p))
    dir = "graphs/"
    #plt.show()
    plt.savefig(dir + "alpha = %d and p = %.2f .png" % (alpha, p) )
    plt.clf()

for alpha in (0, 1):    
    for p in range(1, 11):
        optimize(alpha, p=p/10, K = 10)    


