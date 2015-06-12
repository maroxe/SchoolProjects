import numpy as np
sigma = .2
nu = .3
rho = -.5
a = 0.02
b  = 0.01
sqrtDt = np.sqrt(1./ 365)
dx = sigma  * sqrtDt
dy = nu  * sqrtDt
directions = [ (1, 1), (1, -1), (-1, 1), (-1, -1) ] 

def calc_proba(x, y):
    return [ (1+rho) / 4 + i * (b*sigma*y + j * a*nu*x) * sqrtDt / (4*sigma*nu)
                for (i, j) in [ (-1, 1), (1, -1), (-1, -1), (1, 1) ]]

def calc_node(x, y):
    return [ (x+i*dx, y+j*dy) for (i, j) in [ (1, 1), (1, -1), (-1, 1), (-1, -1) ] ]

def next_slice(curr_slice):
    def work_node(node):
             (x, y), _ = node
             subnodes = calc_node(x,y)
             return map(lambda n: (n, calc_proba(*n)), subnodes)
    return sum(map(work_node, curr_slice), [])

def draw_payoff(payoff):
    plt.pcolormesh(payoff)

def discount(payoff, proba_table):
    disc_payoff = np.zeros_like(payoff)
    for i in range(n):
        for j in range(n):
            for (c, (dir_x, dir_y)) in enumerate(directions):
                if 0 <= i-dir_x < n and 0 <= j-dir_y < n:
                    p = proba_table[i-dir_x][j-dir_y][c]
                    disc_payoff[i][j] += p * payoff[i-dir_x][j-dir_y]
    return disc_payoff

curr_slice = [((0, 0), calc_proba(0,0)),]
curr_slice = next_slice(curr_slice)

n = 100
proba_table = np.array([[calc_proba(i*dx, j*dy) for j in range(-n/2, n/2)] for i in range(-n/2, n/2)])
payoff = np.ones((n, n))
payoff = discount(payoff, proba_table)
    
