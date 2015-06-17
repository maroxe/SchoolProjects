import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as _colors

data = []
with open('mext.dat') as f:
    data = np.array(map(float, f.read().splitlines()))


l1 = 250
l2 = l1+10
p = 2

mean = np.mean(data)
X = data - mean
N = len(X)

# Estimate covariance matrix gamma and regression coefficients phi
cov = np.array([ np.mean( [X[k] * X[k+i] for k in range(N-i)] )for i in range(N)])
gamma = np.array( [ [cov[abs(i-j)] for i in range(p)] for j in range(p)] )
phi = np.linalg.solve(gamma, cov[1:p+1])

row = np.array([-phi[p-i-1] if i < p else 1 if i == p else 0 for i in range(l2-l1+2*p+1)])
PHI = np.array([np.roll(row, j) for j in range(l2-l1+p+1)])


PHI_B = PHI[:, :p]
PHI_M = PHI[:, p:p+l2-l1+1]
PHI_A = PHI[:, p+l2-l1+1:]

X_B = X[l1-p-1:l1-1]
X_M = X[l1-1:l2]
X_A = X[l2:l2+p]


inv =  np.linalg.inv( np.dot(PHI_M.T, PHI_M) )
concat_PHI = np.concatenate((PHI_B, PHI_A), axis=1)
concat_X =  np.concatenate((X_B, X_A))
# XX_M is the best predictor of X_M knowing X_B, X_A
XX_M = - reduce(np.dot, (inv, PHI_M.T, concat_PHI, concat_X))


plt.xlabel('$l_1=%d \, l_2=%d \, p=%d$' % (l1, l2, p))
plt.plot(range(N), X, 'b-+', label='$X$')
plt.plot(range(l1-1, l2), XX_M, 'go-', label='$X_M$')
plt.plot(range(l2, l2+p), X_A, 'ko-', label='$X_A$')
plt.plot(range(l1-p-1, l1-1), X_B, 'ro-', label='$X_B$')

plt.legend()
plt.show()
