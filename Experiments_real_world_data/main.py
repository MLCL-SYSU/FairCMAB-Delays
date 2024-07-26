import numpy as np
from utilities import *
from algorithms import *
from plot import *
L = 3   # the number of selected arm at each round 
K = 10   # the number of arm
T = 50000  # time horizon
total_simulations = 120
click_rate = 0.05
arm_set_title = get_arm_set() # the title of products
mean_reward, mean_reward_max, mean_reward_min = get_mean_reward(arm_set_title, click_rate)

reward_regret_list = []
fairness_regret_list = []
reward_regret_matrix = []
fairness_regret_matrix = []
average_selection_list = np.zeros(K)


policies = []
policies.append(FCUCB_D(L, K, click_rate,mean_reward,arm_set_title))
policies.append(FCTS_D(L, K, click_rate, mean_reward,arm_set_title))
policies.append(OP_FCUCB_D(L, K, click_rate,mean_reward,arm_set_title))
policies.append(OP_FCTS_D(L, K, click_rate, mean_reward,arm_set_title))
policies.append(CUCB(L, K, click_rate,mean_reward,arm_set_title))
policies.append(MP_TS(L, K, click_rate, mean_reward,arm_set_title))
policies.append(FGreedy_D(L, K, click_rate, mean_reward,arm_set_title))

for policy in policies:

    for _ in range(total_simulations):

        for round in range(1, T+1):

            pro, arm_selected = policy.select_arms(round)
            
            reward_of_selected_arms,delay_of_selected_arms = policy.pull_arms(arm_selected)
            policy.update_state(arm_selected, reward_of_selected_arms, delay_of_selected_arms, round)

            reward_regret_list.append(policy.get_reward_regret(pro))
            fairness_regret_list.append(policy.get_fairness_regret(pro))

        average_selection_list += policy.get_average_selection()
        reward_regret_matrix.append(reward_regret_list)
        fairness_regret_matrix.append(fairness_regret_list)
        reward_regret_list = []
        fairness_regret_list = []
        policy.reset()

    reward_regret_array = np.array(reward_regret_matrix)
    fairness_regret_array = np.array(fairness_regret_matrix)

    cumulative_reward_regret_each_sim = np.cumsum(reward_regret_array,axis=1)
    cumulative_fairness_regret_each_sim = np.cumsum(fairness_regret_array,axis=1)
    
    reward_regret_average = np.sum(reward_regret_array, axis=0) / total_simulations
    fairness_regret_average = np.sum(fairness_regret_array, axis=0) / total_simulations
    
    cumulative_reward_regret = np.cumsum(reward_regret_average, axis=0)
    cumulative_fairness_regret = np.cumsum(fairness_regret_average, axis=0)

    
    average_selection = average_selection_list / total_simulations

    filename1='./data/Cum_RR.txt'
    filename2='./data/Cum_FR.txt'

    text_save(cumulative_reward_regret,filename1,mode='a')
    text_save(cumulative_fairness_regret,filename2,mode='a')


    reward_regret_matrix = []
    fairness_regret_matrix = []
    average_selection_list = np.zeros(K)

# plot
Simulation_results()
