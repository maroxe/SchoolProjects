import pickle
import matplotlib.pyplot as plt
import numpy as np

data_file = 'xFGBL20130702.pkl'


with open(data_file, 'rb') as input:
    r=pickle.load(input)[:1000]
    
    T = r['Time'][r['AskCan']]
    frequence  = 1./(np.mean( T[1:] - T[:-1]) )
    print 'freq = ', frequence


    for queue in ('Bid', 'Ask'):

        qty = np.unique(r[queue+'Qty'])
        
        fig, ax = plt.subplots()

        for order_type in ('Lim', 'Mar', 'Can'):
            limit_orders = r[r[queue + order_type]]
            qty_levels =   np.array([ len(r[ r[queue+'Qty'] == n]) for n in qty ])
            freq_lim = np.array([ len(limit_orders[limit_orders[ queue+'Qty'] == n]) for n in qty ], dtype=float)
            freq_lim /= qty_levels
            freq_lim *= frequence
            plt.plot(qty, freq_lim, label= '%s %s intensity' % (queue, order_type))

        ax.legend(loc='upper center', shadow=True)            
    plt.show() 


