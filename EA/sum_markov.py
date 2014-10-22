# Sn = sum Xi
# objectif: P(Sn = k)
import math
from binomial_table import BinomTable

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


n = 5

p0 = 0.5 # p0 = P(X1 = a | X0 = a)
R = 1 # R = P(X0 = 1)
table = BinomTable(n).table

# X0 == 1
def pr_S_1(k, n):
    if k == n: return p0**n
    C1 = int(n + 0.5 - abs(2*k-0.5-n)) if k  < n else 0
    C1 = n
    r = 0
    for C in range(1, C1+1):
        a, b = map(lambda x: int(math.ceil(x)), (0.5 * C - 1, 0.5*C))
        if a <= k and b <= n-k:
            r += table[k][a] * table[n-k-1][b-1] * (p0/(1-p0))**(a+b)
    return r * (1-p0)**n
    
def pr_S_0(k, n):
    C0 = int(n + 0.5 - abs(2*k-0.5-n))
    C0 = n
    r = 1 if k == 0 else 0
    
    for C in range(1, C0+1):
        a, b = map(lambda x: int(math.ceil(x)), (0.5 * C - 1, 0.5*C))
        if a <= k and b <= n-k:
            r += table[k-1][b-1] * table[n-k][a] * (p0/(1-p0))**(a+b)
    return r * (1-p0)**n

def pr_S_cond(k, n, X0):
    if X0 == 0: return pr_S_0(k, n)
    return pr_S_1(k, n)

def pr_S(k, n):
    if k > n: return 0
    return R*pr_S_cond(k, n, 1) + (1-R) * pr_S_cond(k, n, 0)
    
Pr = np.array([ [pr_S(k, m) for k in range(n+1)]  for m in range(n+1)])
print Pr
print map(sum, Pr)
X = np.arange(n+1)
Y = np.arange(n+1)
xpos, ypos = np.meshgrid(X, Y)

zpos =  np.zeros(n+1)
dx =  0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = Pr
        
fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xpos, ypos,  dz)
#plt.show()
