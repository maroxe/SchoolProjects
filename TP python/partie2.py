import numpy as np
import numpy.random as npr
from scipy.stats import norm

import matplotlib.pyplot as plt

S0 = 112
K = 130
r = 0.04
sigma = 0.33
T = 0.5
N = 26
h = T/N
M = 10000

# disrectiser le temps
t = np.linspace(0, T, num=N, endpoint=True)
def estime_S():
    # accroissement du brownien
    dW = np.sqrt(h) * npr.randn(N-1)
    ratio_S = np.exp( (r-sigma**2/2) * h + sigma * dW)
    S = [S0, ]
    Si = S0
    for ratio in ratio_S:
        Si *= ratio
        S.append(Si)
    return np.array(S)


def d(S, t, signe=1):
    arg = S * np.exp(r*(T-t))/K
    sqrt_inv_t = np.sqrt(T-t)
    return 1/(sigma * sqrt_inv_t) * np.log(arg) + signe * sigma/2 * sqrt_inv_t

def d0(S, t): return d(S, t, -1)
def d1(S, t): return d(S, t, 1)    
def N_repartition(y): return norm.cdf(y)

def estime_V(S):
    strategie = map(N_repartition, d1(S, t))
    V0 = S0 * N_repartition(d1(S0, 0)) - K*np.exp(-r*T)*N_repartition(d0(S0, 0))

    V = [V0, ]
    Vi = V0
    for i, s in enumerate(strategie[:-1]):
        Vi = Vi + s*(S[i+1] - S[i]) + (Vi - s*S[i])*r*h
        V.append(Vi)
    return V

def tracking_error():
    S = estime_S()
    V = estime_V(S)    
    eT = (V - np.maximum(S- K, 0))[-1]
    return eT

print np.mean([tracking_error() for _ in range(M)])
