import numpy as np



p = 1./3.
z = 1./3
q = 1-p

u = 1/(z*p) + z*(2-1/p)
u /= 2.
print 'u=', u

x = u + np.sqrt(u**2-1)

print x**2 - 1/(q*z)*x + p/q
