
import numpy as np
import numpy.random as npr

from scipy.stats import norm
import matplotlib.pyplot as plt
from operator import itemgetter

# Nombre de tirages independants
M = 10000
S0 = 100
r = 0.03
sigma = 0.35
T = 0.5
P = 0.95

def intervalle_de_confiance(K):
    # Q1: Simuler la variable aleatoire St a partir de Wt   
    W = np.sqrt(T) * npr.randn(M) 
    S = S0 * np.exp( (r-sigma**2/2)*T + sigma * W)
    
    #Xi = e^(-rT) * (Si - K)+
    X = np.exp(-r*T) * np.maximum(S - K, 0)
    estim_E = np.mean( X )
    estim_Var = (1 + 1. / (M-1)) * ( np.mean(np.power(X, 2)) - estim_E**2)
    
    quantile = - norm.ppf( (1-P)/2 )
    
    # (centre, rayon)
    intervalle_conf = ( estim_E, quantile * np.sqrt(estim_Var/M) ) 
    return intervalle_conf

def intervalle_de_confiance_reduction(K):
    M_prime = 1000
    W = np.sqrt(T) * npr.randn(M) 
    S = S0 * np.exp( (r-sigma**2/2)*T + sigma * W)
    X = np.exp(-r*T) * np.maximum(S - K, 0)
    
    rho = 1. / M_prime * np.dot(X[:M_prime], S[:M_prime]) - np.mean(X[:M_prime]) * S0 * np.exp(r*T)
    rho /= np.mean( np.power(S[:M_prime], 2)) -  (S0 * np.exp(r*T))**2
    
    Y = X[M_prime:] - rho * ( S[M_prime:] - S0 * np.exp(r*T)) 
    estim_E = np.mean(Y)
    estim_Var = (1 + 1. / (M-M_prime-1)) * ( np.mean(np.power(Y, 2)) - estim_E**2)
    # ppf vs cdf
    quantile = - norm.cdf( (1-P)/2 )
    intervalle_conf = ( estim_E, quantile * np.sqrt(estim_Var/(M-M_prime)) ) 
    
    return intervalle_conf

Ks = np.linspace(50, 100, 100)
Is = map(intervalle_de_confiance, Ks)
Is_2 = map(intervalle_de_confiance_reduction, Ks)

Gain = np.array(map(itemgetter(1), Is)) / np.array(map(itemgetter(1), Is_2))
# prix du call en fonction du strike
plt.figure()
plt.errorbar(Ks, map(itemgetter(0), Is), 
           map(itemgetter(1), Is))

plt.errorbar(Ks, map(itemgetter(0), Is_2), 
           map(itemgetter(1), Is_2))
plt.figure()
plt.plot(Ks, Gain)
plt.axhline(y=1)
plt.show()
