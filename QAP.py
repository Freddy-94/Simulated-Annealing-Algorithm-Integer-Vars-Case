"""
Author: Alfredo Bernal Luna
Date: March 13th 2023
Theme: Quadratic assignment problem - Simulated annealing algorithm insight 

"""

import numpy as np
import numpy.random as rn
from itertools import permutations
import random
import copy
import math
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl 

"""
def matrices_gen():
    size = int(input().strip())
    input().strip() # empty string
    flux_matrix = []
    for line in range(size):
        flux_matrix.append(input().strip().split())
    flux_matrix = np.matrix(flux_matrix).astype('float64')
    input().strip() # empty string
    dist_matrix = []
    for line in range(size):
        dist_matrix.append(input().strip().split())
    dist_matrix = np.matrix(dist_matrix).astype('float64')
    print("===============================================")
    print("Execution is starting. Please wait few mins ;)")
    print("===============================================")
    print()
    print(f"Problem size = {size}")
    print()
    print(f"Input flux matrix is: ")
    print()
    print(flux_matrix)
    print()
    print(f"Input distances matrix is: ")
    print()
    print(dist_matrix)
    print()
    return size, flux_matrix, dist_matrix 
"""
  
def perms(size):
    """
    Given the size of the problem, generate the search space of solution
    """
    perm_sols = list()
    canonical_perm = range(1, size+1)
    for perm in permutations(canonical_perm):
        perm = list(perm)
        perm_sols.append(perm)
    perm_sols = np.array(perm_sols) - 1
    # print(perm_sols)
    return perm_sols

def swap_operator(perm):
    rand_index1 = random.randint(0, len(perm)-1)
    rand_index2 = random.randint(0, len(perm)-1)
    while rand_index2 == rand_index1:
        rand_index2 = random.randint(0, len(perm)-1)
    perm[rand_index1], perm[rand_index2] = perm[rand_index2], perm[rand_index1]
    # print(perm)
    return perm  

def reverse_operator(perm):
    rand_index1 = random.randint(0, len(perm)-1)
    rand_index2 = random.randint(0, len(perm)-1)
    while rand_index2 == rand_index1:
        rand_index2 = random.randint(0, len(perm)-1)
    lower_bound = min(rand_index1, rand_index2)
    upper_bound = max(rand_index1, rand_index2)
    # print(lower_bound, upper_bound)
    while lower_bound < upper_bound:
        perm[lower_bound], perm[upper_bound] = perm[upper_bound], perm[lower_bound]
        lower_bound += 1
        upper_bound -= 1
    # print(perm)
    return perm

def insersion_operator(perm):
    rand_index1 = random.randint(0, len(perm)-1)
    rand_element = perm[rand_index1]
    new_pos = random.randint(0, len(perm)-1)
    while new_pos == rand_element:
        new_pos = random.randint(0, len(perm)-1)
    # print(rand_element, new_pos)
    #np.delete(perm, rand_index1) 
    perm.pop(rand_index1)
    perm.insert(new_pos, rand_element)
    # print(perm)
    return perm 

def cost_function_perm(size, flux_matrix, dist_matrix, perm):
    cost_value = 0
    for i in range(size):
        for j in range(size):
            cost_value = cost_value + (flux_matrix.item((i, j)) * dist_matrix.item((perm[i], perm[j])))
            # print(f"flux matrix item {flux_matrix.item((i, j))}* dist matrix item {dist_matrix.item((perm[i], perm[j]))} = {cost_value}")   
    return cost_value            

def lineal_cooling(last_tmp, beta):
    last_tmp = last_tmp - beta
    return last_tmp     
            
def min_cost():
    size, flux_matrix, dist_matrix = matrices_gen()
    perm_sols = perms(size)
    costs = []
    for perm in perm_sols:
        cost = cost_function_perm(size, flux_matrix, dist_matrix, perm)
        costs.append(cost)
    # print(costs)
    global_min = min(costs)
    return global_min

def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        try:
            p = np.exp(- (new_cost - cost) / temperature)
            # print("    - Acceptance probabilty = {:.3g}...".format(p))            
        except ZeroDivisionError:
            p = 0
        return p            

def see_annealing(states, costs, temps):
    plt.figure()
    plt.suptitle(f"Evolution of states and costs of the simulated annealing algorithm")
    plt.subplot(121)
    plt.plot(states, 'r')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'b')
    plt.title("Costs")
    #plt.figure()
    #plt.suptitle(f"Temperatures plot")
    #plt.plot(temps, 'g')
    #plt.title("Temps")
    plt.show()

def annealing(size,
              flux_matrix,
              dist_matrix,
              mutation_operator,              # mutator operator functions defined above             
              maxsteps=500000,                # max number of iterations
              debug=True):
    #size, flux_matrix, dist_matrix  = matrices_gen()
    #solution_space = list(perms(size))
    #index_random_solution = random.randint(0, len(solution_space)-1)    
    for i in range(1, 21):
        state = list(range(0, size))
        cost = cost_function_perm(size, flux_matrix, dist_matrix, state) # First evaluation of the cost function, with a random solution
        T = 100    # Initial temperature fixed to 1000
        beta = 34 # Fixed beta parameter for the linear cooling function
        T0 = copy.copy(T) # Save this temp to start new experiment
        states, costs, temps = [state], [cost], [T] # store all of the found solutions, with their corresponding values, and the current temperature in the system   
        for step in range(1, maxsteps+1):               
            new_state = mutation_operator(state)
            new_cost = cost_function_perm(size, flux_matrix, dist_matrix, new_state)
            #while new_state not in states:
            with open(f'Output {size} sites with {mutation_operator.__name__}.txt', 'a+') as f:                    
                content = "Step #{:>2}/{:>2} : T = {:>4.3f}, state = {}, cost = {:>4.3f}, new_state = {}, new_cost = {:>4.3f} ...\n".format(step, maxsteps, T, state, cost, new_state, new_cost)
                #print("T value in iter " +  str(step) + " = " + str(T)) 
                if debug: f.write(content)
                if acceptance_probability(cost, new_cost, T) > rn.random():
                    state, cost = new_state, new_cost
                    states.append(state)
                    costs.append(cost)
                    # print("  ==> Accept it!")
                # else:
                #    print("  ==> Reject it...")
                T = lineal_cooling(T, beta)  # Decrease temperature
                f.write("============================================\n")
                f.write("                                            \n")
                f.write("============================================\n")
        states = np.array(states)
        costs = np.array(costs)
        avg_solution = np.sum(costs)/costs.size # Solution evaluated in the objective function
        std_dev = math.sqrt((1/states.size)*(np.sum((states - avg_solution)**2)))
        min_sol = np.amin(costs)
        max_sol = np.amax(costs)
        with open(f'Statistical Results for {size} sites {mutation_operator.__name__}.txt', 'a') as stats:
            stats.write("===========================================================================\n")
            # -stats.write(f"Current temperature for iteration {i}, for {coolFun.name} is: {temps[i-1]}\n")
            stats.write(f"Average solution for experiment {i}, is: {avg_solution}\n")
            stats.write(f"Standard deviation for experiment {i}, is: {std_dev}\n")
            stats.write(f"Min solution for experiment {i}, for is: {min_sol}\n")
            stats.write(f"Max solution for experiment {i}, for is: {max_sol}\n")
            stats.write("===========================================================================\n")
        T = T0 # Start new experiment
        if i == 20:
            see_annealing(states, costs, temps)
    print("======================================================")
    print("Success! Watch the output files in your directory :D")
    print("======================================================")    
    return states, costs, temps # state, cost_function(n, state), states, costs, temps


def main():
    try:
        size = int(input().strip())
        input().strip() # empty string
        flux_matrix = []
        for line in range(size):
            flux_matrix.append(input().strip().split())
        flux_matrix = np.matrix(flux_matrix).astype('float64')
        input().strip() # empty string
        dist_matrix = []
        for line in range(size):
            dist_matrix.append(input().strip().split())
        dist_matrix = np.matrix(dist_matrix).astype('float64')
        print("===============================================")
        print("Execution is starting. Please wait few mins ;)")
        print("===============================================")
        print()
        print(f"Problem size = {size}!")
        print()
        print(f"Input flux matrix is: ")
        print()
        print(flux_matrix)
        print()
        print(f"Input distances matrix is: ")
        print()
        print(dist_matrix)
        print()
    except EOFError as e:
        print(e)
    operators = [swap_operator, reverse_operator, insersion_operator]
    for operator in operators:
        annealing(size,
                  flux_matrix,
                  dist_matrix,
                  operator,              
                  maxsteps=1000,
                  debug = True)
    
if __name__ == '__main__':
    main()