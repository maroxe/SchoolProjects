import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

n = 30
def eigen1(z, p):
    return 1/(p* z)

def eigen2(z, p):    
    return 1./(2*p*z) * (1 + 2 * z**2 * p - z**2 + np.sqrt(1+ 4 * (z**2) * p
                                                     - 2 * z**2 + 4 * z**4 * p**2 - 4 * z**4 * p
                                                     + z**4 - 4 * z**2 * p**2))

def eigen3(z, p):
    return 1./(2*p*z) * (1 + 2 * z**2 * p - z**2 - np.sqrt(1+ 4 * (z**2) * p
                                                     - 2 * z**2 + 4 * z**4 * p**2 - 4 * z**4 * p
                                                     + z**4 - 4 * z**2 * p**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

P = np.arange(0.1, 0.9, 1./n)
Z = np.arange(0.1, 0.9, 1./n) 
P, Z = np.meshgrid(P, Z)

ax.plot_wireframe(P, Z, 0.1 + (eigen1(P, Z) < 1), color="black")
ax.plot_wireframe(P, Z, 0.2 + (eigen2(P, Z) < 1), color="green")
ax.plot_wireframe(P, Z, eigen3(P, Z) > -1, color="red")

plt.show()

