import pickle
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt



data_file = 'xFGBL20130702.pkl'

with open(data_file, 'rb') as input:
        r=pickle.load(input)
        X = r['Time']
        Y = r['AskPrice'] - r['BidPrice']

        plt.plot(X, Y)
        plt.show()



