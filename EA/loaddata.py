from __future__ import division
import pickle
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt


data_file = 'xFGBL20130702.pkl'

with open(data_file, 'rb') as input:
        r=pickle.load(input)
        X = r['BidPrice']
        X = X[1:] - X[:-1]
        X = X[ abs(X) > 1e-3] # Keep only the entries where the price changes
        X = np.array(map(lambda x: 1 if x > 0 else -1, X))
        p = np.count_nonzero(X[1:] == X[:-1]) / len(X)
        print 'p = ', p

        import matplotlib.pyplot as plt

        K = 10
        c = 0.1

        # Y_1[k] Stores the mean cost when applying strategy(k) | X0 = 1
        Y_1 = []
        # Y__1[k] Stores the mean cost when applying strategy(k) | X0 = -1
        Y__1 = []

        for k in range(K):
                # count[1] stores the cost of strategy(k) when X0=1
                # the same for count[-1]
                cout = {1: [], -1: [] } 
                for i, X0 in enumerate(X):
                        S = 0
                        j = 1
                        try:
                                # strategy(k) continues while the number of changes < k
                                # and S has not reach -1
                                while j <= k and S >= 0:
                                        S += X[i + j]
                                        j += 1
                                cout[X0].append(  (min(j, k), S+1) )
                        except IndexError:
                                # End of data, the strategy is interrupted
                                continue
                                
                Y_1.append(np.mean(map(lambda (N, S): N*c+S, cout[1])))
                Y__1.append(np.mean(map(lambda (N, S): N*c+S, cout[-1])))
                
        plt.plot(K, Y_1, 'b*-', label="$X_0 = 1$")
        plt.plot(K, Y__1, 'ro-', label="$X_0 = -1$")
        plt.ylabel('mean of $c * min(k, N) + (1+S_{min(k, N)})$' )
        plt.xlabel('$k$')
        plt.legend(loc=2,)
        plt.savefig("graphs/esp_market.png")
        plt.savefig("Rapport/img/esp_market.png")
        plt.show()

