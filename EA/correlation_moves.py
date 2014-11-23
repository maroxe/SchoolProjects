##############################
# Correlation des Xi
##############################

import sys
import time
import numpy as np


def gen_events(freq):
    return np.random.exponential(scale=1./freq, size=size)

def reinit_q(rev=False):
    x, y = np.random.poisson(lam=5., size=2)
    x *= 3
    if rev:
        y,x = x, y
    return {'ask': x, 'bid': y}
    
# (t, V^a, V^b)
size = 100000
freq = 50

ask_can_events = map(lambda t: (t, -1, 0), gen_events(freq))
ask_lim_events = map(lambda t: (t, 1, 0), gen_events(freq))
bid_can_events = map(lambda t: (t, 0, -1), gen_events(freq))
bid_lim_events = map(lambda t: (t, 0, 1), gen_events(freq))

events = (ask_can_events, ask_lim_events, bid_can_events, bid_lim_events)
events = sum(events, [])
events.sort()

Q = reinit_q()
moves = []
for (dt, Va, Vb) in events:
    Q['ask'] += Va
    Q['bid'] += Vb

    if Q['ask'] == 0:
        Q = reinit_q()
        moves.append(1)
        
    if Q['bid'] == 0:
        Q = reinit_q(rev=True)
        moves.append(-1)

moves = np.array(moves)

print np.mean(moves)
print 'cov(X1, X2) = ', np.mean(moves[1:]*moves[:-1])

