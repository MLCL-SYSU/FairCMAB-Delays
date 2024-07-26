import numpy as np
import math
from scipy.optimize import minimize
import random
from random import choice


def DepRound(M, weights_p):
    p = np.array(weights_p)
    K = len(p)
    assert np.isclose(np.sum(p),M), "Error: the sum of weights p_1 + ... + p_K should be = {} (= {}).".format(M,np.sum(p)) 

    # Main loop

    possible_ij = [a for a in range(K) if 0 < p[a] < 1]
    while possible_ij:
        # Choose distinct i, j with 0 < p_i, p_j < 1
        if len(possible_ij) == 1:
            i = np.random.choice(possible_ij, size=1)
            j = i
        else:
            i, j = np.random.choice(possible_ij, size=2, replace=False)
        pi, pj = p[i], p[j]

        # Set alpha, beta
        alpha, beta = min(1 - pi, pj), min(pi, 1 - pj)
        proba = beta / (alpha + beta)
        if np.random.uniform(0,1) < proba:  # with probability = alpha/(alpha+beta)
            pi, pj = pi + alpha, pj - alpha
        else:            # with probability = beta/(alpha+beta)
            pi, pj = pi - beta, pj + beta

        # Store
        p[i], p[j] = pi, pj
        # And update
        possible_ij = [a for a in range(K) if 0 < p[a] < 1]
        if len([a for a in range(K) if np.isclose(p[a], 0)]) == K - M:
            break
    # Final step
    subset = [a for a in range(K) if np.isclose(p[a], 1)]
    if len(subset) < M:
        subset = [a for a in range(K) if not np.isclose(p[a], 0)]
    assert len(subset) == M, "Error: DepRound({}, {}) is supposed to return a set of size {}, but {} has size {}...".format(weights_p, M, M, subset, len(subset))  # DEBUG
    return subset


def Mean_Reward_Fun(mean,K,L):
    merit_sum = sum([Merit_Fun(mean[i],K,L) for i in range(len(mean))])
    return -sum([(Merit_Fun(mean[i],K,L) * mean[i]) / merit_sum for i in range(len(mean))])

def Mean_Reward_Max(fun, ucb, lcb):
    bounds = []
    K = len(ucb)
    for i in range(K):
        bounds.append([lcb[i], ucb[i]])
    initial_value = [(lcb[i]+ucb[i]) / 2 for i in range(K)]
    result = minimize(fun, initial_value, bounds = bounds)#, method='Nelder-Mead', options={'xatol':1e-6,'disp':False})
    return result.x
    



def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+" ")
    if i == len(content)-1:
        file.write('\n')
    file.close()
    

def Merit_Fun(x,K,L,c=4):
    return 1+3.5*(x**c)

    
def Mean_Reward_Fun(mean,K,L):
    merit_sum = sum([Merit_Fun(mean[i],K,L) for i in range(len(mean))])
    return -sum([(Merit_Fun(mean[i],K,L) * mean[i]) / merit_sum for i in range(len(mean))])



def solve_optimized_problem(lcb,ucb,K,L):
    initial_guess = np.random.uniform(lcb, ucb)
    bnds=[]
    for i in range(K):
        bnds.append((lcb[i], ucb[i]))
    res = minimize(Mean_Reward_Fun, initial_guess, args=(K,L),method='SLSQP',bounds=bnds)
    return res.x



def get_arm_set(input_file='Top_10_products.txt', delimiter='\t'):
    arm_set= []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        columns = line.strip().split(delimiter)
        if len(columns) >= 5:
            value = columns[4] 
            if value not in arm_set:
                arm_set.append(value)

    return arm_set


def get_mean_reward(arm_set, click_rate, input_file='Top_10_products.txt',delimiter='\t'):
    mean_reward = []
    for arm in arm_set:
        total_sum = 0
        count = 0

        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            columns = line.strip().split(delimiter)
            if len(columns) >= 5 and columns[4] == arm:  
                try:
                    first_column_value = float(columns[0])  
                    total_sum += first_column_value
                    count += 1
                except ValueError:
                    pass 
        mean_reward.append(click_rate*(total_sum / count))
    mean_reward_max = max(mean_reward)
    mean_reward_min = min(mean_reward)
    mean_reward = (np.array(mean_reward)-mean_reward_min)/(mean_reward_max-mean_reward_min) 
    return mean_reward, mean_reward_max, mean_reward_min



def random_sample_dataset(input_file, value_to_select, delimiter='\t'):
    selected_rows = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        columns = line.strip().split(delimiter)
        if len(columns) >= 5 and columns[4] == value_to_select:
            selected_rows.append(line)

    if selected_rows:
        selected_row = random.choice(selected_rows)
        selected_row_list = [value.strip() for value in selected_row.split(delimiter)]
        return selected_row_list
    else:
        return None


def reward_regret_each_slot(pro,K,L,mean_reward):
    reward_regret = 0
    merit_sum = sum([Merit_Fun(mean_reward[i],K,L) for i in range(K)])
    opt_pro = [L * Merit_Fun(mean_reward[i],K,L) / merit_sum for i in range(K)]
    for a in range(K):
        reward_regret += ((opt_pro[a] - pro[a]) * mean_reward[a])

    return max(reward_regret,0)

def fairness_regret_each_slot(pro,K,L,mean_reward):
    fairness_regret = 0
    merit_sum = sum([Merit_Fun(mean_reward[i],K,L) for i in range(K)])
    opt_pro = [L * Merit_Fun(mean_reward[i],K,L) / merit_sum for i in range(K)]
    for a in range(K):
        fairness_regret += abs(opt_pro[a] - pro[a])

    return fairness_regret

def average_selection_each_slot(number_of_pull,K,L):
    average_selection = [L*number_of_pull[i] / sum(number_of_pull) for i in range(K)]
    return np.array(average_selection)