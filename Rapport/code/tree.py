import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import multiprocessing 
import scipy.stats as ss

directions = [ (1, 1), (1, -1), (-1, 1), (-1, -1) ] 

class TreePricer:

    def __init__(self, sigma=.02, nu=.03, rho=0.5, draw=False):
        self.sigma = sigma
        self.nu = nu
        self.rho = rho
        self.a = 0.02
        self.b  = 0.01
        self.Dt = 1. / 365
        self.sqrtDt = np.sqrt(self.Dt)
        self.dx = self.sigma  * self.sqrtDt
        self.dy = self.nu  * self.sqrtDt
        self.draw = draw

    def calc_proba(self, x, y):
        return [ (1+j*self.rho) / 4 + i * (self.b*self.sigma*y + j * self.a*self.nu*x) * self.sqrtDt / (4*self.sigma*self.nu)
                    for (i, j) in [ (-1, 1), (1, -1), (-1, -1), (1, 1) ]]

    def calc_node(self, x, y):
        return [ ((x+i)*self.dx, (y+j)*self.dy) for (i, j) in directions ]

    def discount(self, payoff, proba_table, t):
        disc_payoff = np.zeros_like(payoff)
        n = len(payoff)
        for i in range(t+1, n-t-1):
            for j in range(t+1, n-t-1):
                neighbours = self.calc_node(i, j)
                for (c, (dir_x, dir_y)) in enumerate(directions):
                    p = proba_table[i-dir_x][j-dir_y][c]
                    disc_factor =  np.exp(-sum(neighbours[c])*self.Dt)
                    disc_payoff[i][j] += p * disc_factor * payoff[i-dir_x][j-dir_y]
        return disc_payoff

    def price(self, t0, payoff):
        n = len(payoff)
        proba_table = np.array([[self.calc_proba(i*self.dx, j*self.dy) for j in range(-n/2, n/2)] for i in range(-n/2, n/2)])
        for t in range(n/2-t0):
            payoff = self.discount(payoff, proba_table, t)
            if self.draw:
                draw_payoff(payoff)

        return payoff

    def Mx(self, t, T):
        return (1 - np.exp(-self.a * (T-t)))/self.a

    def My(self, t, T):
        return (1 - np.exp(-self.b * (T-t)))/self.b

    def V(self, t, T):
        a = self.a
        b = self.b
        V1 = (self.sigma / a)**2 * (T-t + 2*(np.exp(-a*(T-t))-1)/a - (np.exp(-2*a*(T-t))-1)/(2*a) )
        V2 = (self.nu / b)**2 * (T-t + 2*(np.exp(-b*(T-t))-1)/b - (np.exp(-2*b*(T-t))-1)/(2*b) )
        V3 = 2*(self.sigma  *self.nu) / (a*b) * (T-t + (np.exp(-a*(T-t))-1)/a + (np.exp(-b*(T-t))-1)/b - (np.exp(-(a+b)*(T-t))-1)/(a+b) )
        return V1 + V2 + V3

    def P(self, t, T, x, y): 
        return np.exp( -self.Mx(t, T) * x- self.My(t, T) * y + 0.5 * self.V(t, T))

    def tree_caplet(self, t, T, K):
        nsteps = T
        X = np.arange(-nsteps, nsteps+1) * self.sigma
        Y = np.arange(-nsteps, nsteps+1) * self.nu
        X, Y = np.meshgrid(X, Y)
        R = X + Y
        payoff = np.maximum(R - K, 0) * (T - t)
        return self.price(t, payoff)


    def caplet_sigma(self, t, S, T):
        Mx = self.Mx(T, S)
        My = self.My(T, S)
        a = self.a
        b = self.b
        
        Sigma = self.sigma **2 * Mx * Mx * (1-np.exp(-2*a*(T-t))) / (2*a) \
                + self.nu **2 * My * My * (1-np.exp(-2*b*(T-t))) / (2*b) \
                + 2 * self.rho * self.sigma * self.nu * Mx * My * (1-np.exp(-(a+b)*(T-t))) / (a+b)
        return Sigma

    def cf_caplet(self, t, T, S, x, y, K):
        bondT = self.P(t, T, x, y)
        bondS = self.P(t, S, x, y)
        Sigma = self.caplet_sigma(t, T, S)
        d1 = np.log( (K * bondT) / bondS) / Sigma - 0.5 * Sigma
        d2 = d1 + Sigma
        phi = ss.norm.cdf
        return -bondS * phi(d1) + K*bondT*phi(d2)

def draw_payoff(payoff):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(payoff); 
    fig.colorbar(mesh); 

def price_caplet(t, T, K, sigma, nu, rho):
    pricer = TreePricer(sigma, nu, rho)
    return pricer.tree_caplet(t, T, K)[T-t:T+t+1, T-t:T+t+1]

def price_caplet_cf(t, T, S, K, sigma, nu, rho):
    pricer = TreePricer(sigma, nu, rho)
    return pricer.cf_caplet(t, T, S, 0, 0, K)
    
def draw_surface(H):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.mgrid[:len(H), :len(H[0])]
    ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
        linewidth=0, antialiased=False)
    ax.scatter(X.ravel(), Y.ravel(), H.ravel(), c=H.ravel())
    plt.show()

def draw_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('T')
    ax.set_ylabel('K')
    ax.set_zlabel('Caplet')
    ax.set_title('Price surface')

    H = np.array([[ price_caplet_cf(t=0, T=10, S=10+i, K=0.1*j, sigma=0.01, nu=0.02, rho=-0.5) for j in range(1, 10) ] for i in range(1, 10) ])
    X,Y = np.mgrid[:len(H), :len(H[0])]
    ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.scatter(X.ravel(), Y.ravel(), H.ravel(), c=H.ravel())


    plt.show()

!

draw_surface()
