from __future__ import division
import pickle
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob




for data_file in glob('./data/xFGBL/*.pkl'):
    with open(data_file, 'rb') as input:
        r=pickle.load(input)[:10000]
        X = r['BidPrice']
        X = X[1:] - X[:-1]
        T = r["Time"][abs(X) > 1e-3]
        T = T[1:] - T[:-1]
        print ' T= ', np.mean(T), len(T)
        




