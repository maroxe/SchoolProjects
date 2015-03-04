import numpy as np
import matplotlib.pyplot as plt



M = 10000 # Nombre de tirages independants
S0 = 100
K = 90
sigma = 0.35
T = 0.5

# Generation de la variable log normale
Wt = np.random.normal(scale=np.sqrt(T), size=M)
X = np.maximum(S0 * np.exp( sigma * Wt - 0.5 * sigma**2 * T) -K, 0)
a = 10.307

# Les valeurs qu'on affichera par la suite
X_M = []
f_bar_sup = []
f_bar_inf = []

intervalle = list(range(10, M, M/100))
for i in intervalle:

    # Calcul des moyennes partielles
    X_bar_1 =  np.mean(X[:i/2])
    X_bar_2 =  np.mean(X[i/2:i])
    X_i = (X_bar_1 + X_bar_2) / 2. 

    X_M.append(X_i)
    f_bar_sup.append( np.maximum(X_i, a) )
    f_bar_inf.append(X_bar_2 if X_bar_1 >= a else a)
    

# Affichage
plt.figure()

plt.xlabel('$M$')

plt.plot(intervalle, X_M, 'go-', label='$X_M$')
plt.plot(intervalle, f_bar_inf, 'b*-', label='$f_M sup$')
plt.plot(intervalle, f_bar_sup, 'ro-', label='$f_M inf$')

plt.legend()
plt.show()
