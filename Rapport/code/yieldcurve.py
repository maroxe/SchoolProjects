import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing 

class YieldCurve:
    curve = []

    def start(self, T=365, n=8):
        self.pool = Pool()
        step = T/n
        self.curve = self.pool.map(work, [ range(i, i+step) for i in range(0, T, step) ])

def work(myrange):
    name = multiprocessing.current_process().name
    print name, 'Starting'
    print name, 'Exiting'
    return [ (t, price(np.ones((2*t+1, 2*t+1)))) for t in myrange]

def get_curve():
    mycurve = YieldCurve()
    mycurve.start(T=30, n=5) 
    L =  sum(mycurve.curve, [])
    plt.plot(zip(*L)[1], 'k^-')
    plt.show()
