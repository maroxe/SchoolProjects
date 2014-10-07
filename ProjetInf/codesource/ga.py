import numpy as np
import random
from scipy.stats import bernoulli

def default_mutation(l, x):
    bits_to_change = random.sample(range(len(x)), l)
    x_mut = list(x)
    for b in bits_to_change:
        x_mut[b] = 1 - x_mut[b]
    return x_mut

def default_crossover(c, x, xx):
    parent = [x, xx]
    parent_choice = bernoulli.rvs(c, size=len(x))
    return [ parent[p][i] for i, p in enumerate(parent_choice) ]

def default_rule(offspring_size, better_solution_found):
    return offspring_size

def one_fifth_rule(offspring_size, better_solution_found):
    F = 1.5
    if better_solution_found:
        return max(1, offspring_size / F)
    else:
        return offspring_size * (F**(0.25))

class EA:
    def __init__(self, fitness, mutation=default_mutation, crossover=default_crossover, 
                 l_distribution=np.random.binomial):
        self.mutation = mutation
        self.crossover = crossover
        self.fitness = fitness
        self.l_distribution = l_distribution

    def run(self, n, x_init, offspring_size=5, n_generations=10, 
            self_adapt_rule=default_rule, max_fitness=None ):
        
        # parameters
        p = min(1, float(offspring_size) / n)
        c = 1. / offspring_size        
        
        # initialization
        x, fit_x = x_init, self.fitness(x_init)

        # optimization
        for _ in range(n_generations):
            
            # mutation
            l = self.l_distribution(n, p)
            x_mut = [self.mutation(l, x) for _ in range(int(offspring_size))]
            x_tab_fitness = map(self.fitness, x_mut)
            x_arg_max = np.argmax(x_tab_fitness)
            xx = x_mut[ x_arg_max ] # x'
            
            # crossover, avoid the useless crossover phase when lambda = 1
            if offspring_size > 1:
                y_cross = [self.crossover(c, x, xx) for _ in range(int(offspring_size))]
                y_tab_fitness = map(self.fitness, y_cross)
                y_arg_max = np.argmax(y_tab_fitness)
                y = y_cross[y_arg_max]
                fit_y = y_tab_fitness[y_arg_max]
            else:
                y = xx
                fit_y = x_tab_fitness[x_arg_max]

            if fit_y > fit_x: 
                x = y
                fit_x = fit_y
                offspring_size = self_adapt_rule(offspring_size, True)
            else:
                offspring_size = self_adapt_rule(offspring_size, False)

            p = min(1, float(offspring_size) / n)   
            c = 1. / offspring_size 
            
            if max_fitness != None and fit_x >= max_fitness:
                    return x 
        return x
  


