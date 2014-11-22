from __future__ import division
import operator
import numpy as np
from polynomes import *

p = 3/4
Vp = np.array([p**k for k in range(30)])
Vq = np.array([(1-p)**k for k in range(30)])

def calcul_proba(table_proba, Vp, alpha=0.):
    # P(. | X0 = -1)
    #proba_1 = np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])
    # P(. | X0 = 1)
    #proba_2 = np.array([ np.array([np.dot(Vq[:len(Q)], Q) for Q in PP]) for PP in table_proba])
    #return proba_1 * alpha +  proba_2 * (1-alpha)
    return np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])
    
# alpha = P(X0 = -1)
alpha = 1

# Loi de N
proba_N = calcul_proba(P_N, Vp, alpha)
proba_inv_N = calcul_proba(P_N, Vq, alpha)

# Loi de S_k

proba_S = calcul_proba(P_S_X0__1, Vp,  alpha)
proba_inv_S = calcul_proba(P_S_X0__1, Vq, alpha)

print 'P( S_0 = .) =' , proba_S[3]
print 'Somme = ', sum(proba_S[2])

for k, Q in enumerate(P_N[0]):
    if k % 2:
        print 'P(N =', k, ' | S_0 =', 0, ' ) =', proba_N[0][k]

# E[ k ^ N]    
def esp_min_k_N(k):
    return np.dot(np.array(range(k)), proba_N[0][:k]) + k * (1 - sum(proba_N[0][:k]))

# P(k < N_tilde | S0_tilde = Sk) 
def proba_k_inf_N_cond_S_k(k, Sk):
    if Sk == -1: return 0
    # P(k < N | S0 = -s-2) = P( k < N_tilde | S0 = s)
    if Sk >= 0: return sum(proba_inv_N[Sk][:k])
    return sum(proba_N[-Sk-2][:k])

# E[ 1_{k < N} S_k
def esp_1_mult_S_k(k):
    return sum([ j * proba_k_inf_N_cond_S_k(k, Sk=j) * proba_S[k][j] for j in range(0, k+1)]) + sum([ j * proba_k_inf_N_cond_S_k(k, Sk=j) * proba_inv_S[k][-j] for j in range(-k,0)]) 
    
def optimize(c=0.1, M=4):
    # E[ c(k ^ N) + (1+S_{k^N} ]
    tests = [ (esp_min_k_N(k) , 
               1 + esp_1_mult_S_k(k) - sum(  proba_N[0][:k+1] ) )
              for k in range(M) ]
    print 'min = ',   tests

print esp_min_k_N(1)
optimize(M = 4)
