import numpy as np
import matplotlib.pyplot as plt

data = []
with open('mext.dat') as f:
    data = np.array(map(float, f.read().splitlines()))


l1 = 400
l2 = l1 + 1

# calculate phi
mean = data.mean()
X = data - mean
N = len(X)
cov = np.array([ np.mean( [X[k] * X[k+i] for k in range(N-i)] )for i in range(N)])

p = 2

gamma = np.array( [ [cov[abs(i-j)] for i in range(p)] for j in range(p)] )
phi = np.linalg.solve(gamma, cov[1:p+1])
print 'phi', phi, cov[1]/cov[0]

row = np.array([-phi[p-i-1] if i < p else 1 if i == p else 0 for i in range(l2-l1+2*p+1)])
PHI = np.array([np.roll(row, j) for j in range(l2-l1+p+1)])
print PHI

PHI_B = PHI[:, :p]
PHI_M = PHI[:, p:p+l2-l1+1]
PHI_A = PHI[:, p+l2-l1+1:]
print PHI_A
X_B = X[l1-p-1:l1-1]
X_M = X[l1-1:l2]
X_A = X[l2:l2+p]


inv =  np.linalg.inv( np.dot(PHI_M.T, PHI_M) )
concat_PHI = np.concatenate((PHI_A, PHI_B), axis=1)
concat_X =  np.concatenate((X_B, X_A))

#XX_M = - reduce(np.dot, (inv, PHI_M.T, concat_PHI, concat_X))
XX_M = - reduce(np.dot, (inv, PHI_M.T, concat_PHI, concat_X))
print 'error = ', XX_M - X_M

plt.plot(range(N), X, 'g-*')
plt.plot(range(l1-1, l2), XX_M, 'ro-')
#plt.plot(range(l1, l2+1), X_M)

plt.show()
