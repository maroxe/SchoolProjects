from scipy.stats import bernoulli
import numpy as np

# parametres de la simulation
RAND_SIZE = int(2e6)
K = 2
M = int(1e6) # Monte Carlo

p = 0.5

X = {1: iter([]), -1: iter([])}

def simule_S(k, X0):
    i = 0
    S = 0
    for i in range(k):
        try:
            X0 =  X[X0].next()
        except StopIteration:
            X[X0] = iter(2*bernoulli.rvs( p if X0 > 0 else (1-p), size=RAND_SIZE)-1)
            print 'iter'
        S += X0
        if S <= -1: return S
    return S


for k in range(K):
    print 'E(S_{%d^n}) = ' % k, np.mean( np.array([simule_S(k, 1) for _ in range(M) ]))
