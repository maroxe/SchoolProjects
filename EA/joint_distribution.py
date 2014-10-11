####################
# Joint distribution of Ask/Bid Qty
####################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d


data_directory = 'data/xFGBL'
img_directory = 'images/'
data_file = 'xFGBL20130702.pkl'

for fn in os.listdir(data_directory):
    with open(os.path.join(data_directory, fn), 'rb') as input:
        r=pickle.load(input)

        X = r['AskQty']
        Y = r['BidQty']
        bins = np.arange(0, 600, 20)
        hist, xedges, yedges = np.histogram2d(Y, X, bins=bins, normed=True)

        fig = plt.figure()
        fig.suptitle(fn, fontsize=20)

        ax = fig.add_subplot(111, projection='3d')
        elements = (len(xedges) - 1) * (len(yedges) - 1)
        X, Y = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
        ax.plot_wireframe(X, Y, hist)


        # xpos = X.flatten()
        # ypos = Y.flatten()
        # zpos =  np.zeros(elements)
        # dx =  10 * np.ones_like(zpos)
        # dy = dx.copy()
        # dz = hist.flatten()

        #ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        #ax.scatter(xpos, ypos, dz)


        #plt.show()
        plt.savefig(os.path.join(img_directory, fn + '.png'))



