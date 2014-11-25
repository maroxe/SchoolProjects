from __future__ import division
import numpy as np
import scipy
from scipy import interpolate

data = np.array([
[0, 1],
[0, 1, -2, 1],
[0, 1, -4, 7, -6, 2],
[0, 1, -6, 18, -32, 34, -20, 5],
[0, 1, -8, 34, -92, 166, -200, 155, -70, 14],
[0, 1, -10, 55, -200, 510, -932, 1220, -1120, 686, -252, 42],
[0, 1, -12, 81, -370, 1220, -2992, 5524, -7672, 7910, -5880, 2982, -924, 132],
[0, 1, -14, 112, -616, 2492, -7672, 18298, -34064, 49462, -55524, 47292, -29568, 12804, -3432, 429],
[0, 1, -16, 148, -952, 4564, -16912, 49462, -115516, 216514, -325488, 390012,  -367752, 267036, -144144, 54483, -12870, 1430],
[0, 1, -18, 189, -1392, 7716, -33432, 115836, -325488, 747762, -1409492,  2178762, -2749296, 2806188, -2282280, 1444872, -686400, 230230, -48620, 4862],    
    ]
)

N = 20
mat = np.zeros(shape=(N, N), dtype=int)

for i, L in enumerate(data):
    for j, x in enumerate(L):
        mat[i][j] = x

for j in range(7):        
    L = mat.T[j]
    L = L[L != 0]
    print L
    P = scipy.interpolate.lagrange(range(1, len(L)+1), L)

    print 'P_', j, ' = ' , map(lambda x: 0 if abs(x) < 1e-3 else x, P.c)









