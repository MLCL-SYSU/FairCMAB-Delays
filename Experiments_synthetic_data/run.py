import numpy as np
from utilities import *
from algorithms import *
def Experiments(policies,mean_reward,total_simulations,L,K,T,f,c=4):
    
    reward_regret_list = []
    fairness_regret_list = []
    reward_regret_matrix = []
    fairness_regret_matrix = []
    average_selection_list = np.zeros(K)

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

        reward_regret_average = np.sum(reward_regret_array, axis=0) / total_simulations
        fairness_regret_average = np.sum(fairness_regret_array, axis=0) / total_simulations
        
        cumulative_reward_regret = np.cumsum(reward_regret_average, axis=0)
        cumulative_fairness_regret = np.cumsum(fairness_regret_average, axis=0)
        if f == 'Fixed_delays':
            filename1 = './data/Cum_RR_last_round_'+f+'.txt'
            filename2 = './data/Cum_FR_last_round_'+f+'.txt'
            text_save([cumulative_reward_regret[-1]],filename1,mode='a')
            text_save([cumulative_fairness_regret[-1]],filename2,mode='a')
        
        average_selection = average_selection_list / total_simulations
        
        filename3='./data/Cum_RR_'+f+'.txt'
        filename4='./data/Cum_FR_'+f+'.txt'
        text_save(cumulative_reward_regret,filename3,mode='a')
        text_save(cumulative_fairness_regret,filename4,mode='a')
        if f == 'Geometric_delays':
            filename5='./data/Selection_pro_'+f+'.txt'
            text_save(average_selection,filename5,mode='a')

        reward_regret_matrix = []
        fairness_regret_matrix = []
        average_selection_list = np.zeros(K)

    # optimal fair policy
    if f == 'Geometric_delays':
        merit_sum = sum([Merit_Fun(mean_reward[i],K,L,c) for i in range(K)])
        opt_pro = [L * Merit_Fun(mean_reward[i],K,L,c) / merit_sum for i in range(K)]
        text_save(opt_pro,filename5,mode='a')
    

    