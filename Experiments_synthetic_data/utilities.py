import numpy as np
import math
from scipy.optimize import minimize

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
        if np.random.uniform(0,1) < proba:  
            pi, pj = pi + alpha, pj - alpha
        else:            
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

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+" ")
    if i == len(content)-1:
        file.write('\n')
    file.close()
    
def Merit_Fun(x,K,L,c):
    return 1+2*(x**c)


def Mean_Reward_Fun(mean,K,L,c=4):
    merit_sum = sum([Merit_Fun(mean[i],K,L,c) for i in range(len(mean))])
    return -sum([(Merit_Fun(mean[i],K,L,c) * mean[i]) / merit_sum for i in range(len(mean))])

def solve_optimized_problem(lcb,ucb,K,L,c):
    initial_guess = np.random.uniform(lcb, ucb)
    bnds=[]
    for i in range(K):
        bnds.append((lcb[i], ucb[i]))
    res = minimize(Mean_Reward_Fun, initial_guess, args=(K,L,c),method='SLSQP',bounds=bnds)
    return res.x

def reward_regret_each_slot(pro,mean_reward,K,L,c):
    reward_regret = 0
    merit_sum = sum([Merit_Fun(mean_reward[i],K,L,c) for i in range(K)])
    opt_pro = [L * Merit_Fun(mean_reward[i],K,L,c) / merit_sum for i in range(K)]
    for a in range(K):
        reward_regret += ((opt_pro[a] - pro[a]) * mean_reward[a])
    return max(reward_regret,0)

def fairness_regret_each_slot(pro,mean_reward,K,L,c):
    fairness_regret = 0
    merit_sum = sum([Merit_Fun(mean_reward[i],K,L,c) for i in range(K)])
    opt_pro = [L * Merit_Fun(mean_reward[i],K,L,c) / merit_sum for i in range(K)]
    for a in range(K):
        fairness_regret += abs(opt_pro[a] - pro[a])
    return fairness_regret

def average_selection_each_slot(number_of_pull,K,L):
    average_selection = [L*number_of_pull[i] / sum(number_of_pull) for i in range(K)]
    return np.array(average_selection)