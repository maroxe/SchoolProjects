from __future__ import division
import operator
from polynomes import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

p = 0.2
alpha =1 
def calcul_proba(table_proba, Vp, alpha=0.):
    return np.array([ np.array([np.dot(Vp[:len(Q)], Q) for Q in PP]) for PP in table_proba])

def fill(A, k=20):
    A= map(lambda L: np.append(L, [0]*int((k-len(L))/2)), A)
    A= map(lambda L: np.append([0]*((k-len(L))), L), A)
    return arrify(A)

# Loi de S_k
Vp = np.array([p**k for k in range(30)])
Vq = np.array([(1-p)**k for k in range(30)])

proba_S = (1-alpha) *  calcul_proba(P_S_X0__1, Vp) + alpha * calcul_proba(P_S_X0_1, Vq)
proba_S = fill(proba_S[:-1])
print len(proba_S)

K = int(len(proba_S))
dx = 0.25
dy = dx
dz = proba_S.flatten()
ypos = arrify(list(range(K))) 
xpos = list(range( len(proba_S[0])))
xpos, ypos = np.meshgrid(xpos, ypos)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
plt.show()
