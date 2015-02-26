from __future__ import division
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from matplotlib import colors
colors_ = iter(colors.cnames)



N = 1000
X = bernoulli.rvs(0.5, size=N)
X = 1 - 2*X
W = []
St = 0

for n, x in enumerate(X):
    St += x
    W.append(St / np.sqrt(n+1))


steps = [0, 20, 100, 200]
for i, j in zip(steps[:-1], steps[1:]):
    plt.plot(range(i, j+1), W[i: j+1], next(colors_))



plt.show()
