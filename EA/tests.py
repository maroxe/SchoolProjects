import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
X = np.arange(0, 10, step=0.4)
Y = np.arange(0, 10, step=0.4)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, np.exp(-X*X - Y*Y))
plt.show()





