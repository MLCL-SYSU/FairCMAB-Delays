from utilities import *
import numpy as np
import scipy.stats as st
from arm import Arm


class FCUCB_D():
    def __init__(self, L_, K_,click_rate_,mean_reward_,arm_set_title_):
        self.L = L_
        self.K = K_
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.click_rate = click_rate_
        self.arm_set_title = arm_set_title_
        self.mean_reward = mean_reward_
        self.arms=[Arm(self.arm_set_title,self.click_rate,self.mean_reward) for a in range(self.K)]
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]

    def select_arms(self, round):
        delta = 0.01
        empirical_average = self.sum_of_received_reward / np.maximum(self.number_of_received_feedback,1)
        confidence_radius = np.power(np.log(4 * self.K * round / delta) / (2*np.maximum(self.number_of_received_feedback,1)), 1/2)
        ucb = (empirical_average + confidence_radius) 
        lcb = (empirical_average - confidence_radius) 
        mean_reward_est = solve_optimized_problem(lcb,ucb,self.K,self.L)
        merit_sum_est = sum([Merit_Fun(mean_reward_est[i],self.K,self.L) for i in range(self.K)])
        pro = [self.L * Merit_Fun(mean_reward_est[i],self.K,self.L) / merit_sum_est for i in range(self.K)]
        arm_selected = DepRound(self.L, pro)

        return pro, arm_selected
    
    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull(a)
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()


    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward )
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.number_of_pull = np.ones(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.number_of_received_feedback = np.ones(self.K)
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]
            
            
class FCTS_D():
    def __init__(self, L_, K_,click_rate_,mean_reward_,arm_set_title_):
        self.L = L_
        self.K = K_
        self.click_rate = click_rate_
        self.arm_set_title = arm_set_title_
        self.mean_reward = mean_reward_
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.arms=[Arm(self.arm_set_title,self.click_rate,self.mean_reward) for a in range(self.K)]
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]


    def select_arms(self, round):
        mean_reward_est = [st.beta.rvs(a=self.a[i],b=self.b[i],size=1)[0] for i in range(self.K)]
        merit_sum_est = sum([Merit_Fun(mean_reward_est[i],self.K,self.L) for i in range(self.K)])
        pro = [self.L * Merit_Fun(mean_reward_est[i],self.K,self.L) / merit_sum_est for i in range(self.K)]
        arm_selected = DepRound(self.L, pro)

        return pro, arm_selected

    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull(a)
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])
            for s in reward_index:
                if self.outstanding_reward[i][s] == 1:
                    self.a[i] += 1
                else:
                    self.b[i] += 1

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()

    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]
            

            
class OP_FCUCB_D():
    def __init__(self, L_, K_,click_rate_,mean_reward_,arm_set_title_):
        self.L = L_
        self.K = K_
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.click_rate = click_rate_
        self.arm_set_title = arm_set_title_
        self.mean_reward = mean_reward_
        self.arms=[Arm(self.arm_set_title,self.click_rate,self.mean_reward) for a in range(self.K)]
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]
            
    def select_arms(self, round):
        delta = 0.01
        empirical_average_pessimistic = self.sum_of_received_reward / self.number_of_pull
        empirical_average_optimistic = (self.number_of_pull-self.number_of_received_feedback)/self.number_of_pull + self.sum_of_received_reward / self.number_of_pull
        confidence_radius = np.power(np.log(6 * self.K * round / delta) / (2*self.number_of_pull), 1/2)
        optimistic_ucb = empirical_average_optimistic + confidence_radius
        pessimistic_lcb = empirical_average_pessimistic - confidence_radius
        ones = np.ones(self.K)
        zeros = np.zeros(self.K)
        optimistic_ucb = np.minimum(optimistic_ucb, ones)
        pessimistic_lcb = np.maximum(pessimistic_lcb, zeros)
        mean_reward_est = solve_optimized_problem(pessimistic_lcb,optimistic_ucb,self.K,self.L)
        merit_sum_est = sum([Merit_Fun(mean_reward_est[i],self.K,self.L) for i in range(self.K)])
        pro = [self.L * Merit_Fun(mean_reward_est[i],self.K,self.L) / merit_sum_est for i in range(self.K)]
        arm_selected = DepRound(self.L, pro)

        return pro, arm_selected
    
    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull(a)
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()


    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.number_of_pull = np.ones(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.number_of_received_feedback = np.ones(self.K)
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]
            
            
            
class OP_FCTS_D():
    def __init__(self, L_, K_,click_rate_,mean_reward_,arm_set_title_):
        self.L = L_
        self.K = K_
        self.click_rate = click_rate_
        self.arm_set_title = arm_set_title_
        self.mean_reward = mean_reward_
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.arms=[Arm(self.arm_set_title,self.click_rate,self.mean_reward) for a in range(self.K)]
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]

    def select_arms(self, round):
        mean_reward_est_optimistic = [st.beta.rvs(a=self.a[i]+self.number_of_pull[i]-self.number_of_received_feedback[i],b=self.b[i],size=1)[0] for i in range(self.K)]
        mean_reward_est_pessimistic = [st.beta.rvs(a=self.a[i],b=self.b[i]+self.number_of_pull[i]-self.number_of_received_feedback[i],size=1)[0] for i in range(self.K)]
        merit_sum_est = sum([Merit_Fun((mean_reward_est_optimistic[i]+mean_reward_est_pessimistic[i])/2,self.K,self.L) for i in range(self.K)])
        pro = [self.L * Merit_Fun((mean_reward_est_optimistic[i]+mean_reward_est_pessimistic[i])/2,self.K,self.L) / merit_sum_est for i in range(self.K)]
        arm_selected = DepRound(self.L, pro)

        return pro, arm_selected

    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull(a)
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])
            for s in reward_index:
                if self.outstanding_reward[i][s]==1:
                    self.a[i] += 1
                else: 
                    self.b[i] += 1

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()

    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.initial_reward_delay = [list(self.arms[i].pull(i)) for i in range(self.K)]
        self.arriving_round = [[self.initial_reward_delay[i][1]] for i in range(self.K)] 
        self.outstanding_reward = [[self.initial_reward_delay[i][0]] for i in range(self.K)]



class CUCB():
    def __init__(self, L_, K_, mean_reward_,delay_type, reward_type, p_=0, c=4):
        self.L = L_
        self.K = K_
        self.p = p_
        self.c = c        
        self.delay_type = delay_type
        self.reward_type = reward_type
        self.mean_reward = mean_reward_        
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])
        
    def select_arms(self, round):
        delta = 0.01
        empirical_average = self.sum_of_received_reward / np.maximum(self.number_of_received_feedback,1)
        confidence_radius = np.power((self.L+1)*np.log(round) / (np.maximum(self.number_of_received_feedback,1)), 1/2)
        ucb = empirical_average + confidence_radius
        arm_selected = list(np.argsort(-ucb))[0:self.L]
        pro = [0]*self.K
        for a in arm_selected:
            pro[a] = 1

        return pro, arm_selected

    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull()
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()

    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.mean_reward,self.K,self.L)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.number_of_pull = np.ones(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])


class MP_TS():
    def __init__(self, L_, K_, mean_reward_,delay_type, reward_type, p_=0, c=4):
        self.L = L_
        self.K = K_
        self.p = p_
        self.c = c
        self.delay_type = delay_type
        self.reward_type = reward_type
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.mean_reward = mean_reward_        
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])


    def select_arms(self, round):
        mean_reward_est = [st.beta.rvs(a=self.a[i],b=self.b[i],size=1)[0] for i in range(self.K)]
        arm_selected = list(np.argsort(-np.array(mean_reward_est)))[0:self.L]
        pro = [0]*self.K
        for a in arm_selected:
            pro[a] = 1
        return pro, arm_selected

    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull()
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])
            for s in reward_index:
                if self.outstanding_reward[i][s]==1:
                    self.a[i] += 1
                else: 
                    self.b[i] += 1

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()

    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.K,self.L,self.mean_reward)
        return reward_regret

    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.mean_reward,self.K,self.L)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.number_of_pull = np.ones(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.a = [1]*self.K
        self.b = [1]*self.K
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])
            
            
class FGreedy_D():
    def __init__(self, L_, K_, mean_reward_, delay_type, reward_type, p_=0,c=4):
        self.L = L_
        self.K = K_
        self.p = p_
        self.c = c
        self.mean_reward = mean_reward_        
        self.number_of_pull = np.ones(self.K)
        self.number_of_received_feedback = np.zeros(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.delay_type = delay_type
        self.reward_type = reward_type
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
            
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])
            
    def select_arms(self, round):
        epsilon = 0.15
        
        if np.random.uniform(0,1) < epsilon:
            arm_selected = random.sample(list(range(0,self.K)), self.L)
            pro = [self.L/self.K]*self.K
        else:
            empirical_average = self.sum_of_received_reward / np.maximum(self.number_of_received_feedback,1)        
            merit_sum_est = sum([Merit_Fun(empirical_average[i],self.K,self.L,self.c) for i in range(self.K)])
            pro = [self.L * Merit_Fun(empirical_average[i],self.K,self.L,self.c) / merit_sum_est for i in range(self.K)]
            arm_selected = DepRound(self.L, pro)

        return pro, arm_selected
    
    def pull_arms(self,arm_selected):
        reward_of_selected_arms = []
        delay_of_selected_arms = []
        for a in arm_selected:
            reward, delay = self.arms[a].pull(a)
            reward_of_selected_arms.append(reward)
            delay_of_selected_arms.append(delay)
        return reward_of_selected_arms,delay_of_selected_arms

    def update_state(self, arm_selected, reward_of_selected_arms, delay_of_selected_arms, round):
        for a in arm_selected:
            self.number_of_pull[a] += 1
            self.arriving_round[a].append(delay_of_selected_arms[arm_selected.index(a)] + round)
            self.outstanding_reward[a].append(reward_of_selected_arms[arm_selected.index(a)])

        for i in range(self.K):   
            arriving_round_array = np.array(self.arriving_round[i])
            reward_index = np.where(arriving_round_array <= round + 1)[0]
            self.number_of_received_feedback[i] += len(reward_index) 
            self.sum_of_received_reward[i] += sum([self.outstanding_reward[i][s] for s in reward_index])

            self.arriving_round[i] = np.delete(self.arriving_round[i], reward_index).tolist()
            self.outstanding_reward[i] = np.delete(self.outstanding_reward[i], reward_index).tolist()


    def get_reward_regret(self, pro):
        reward_regret = reward_regret_each_slot(pro,self.mean_reward,self.K,self.L)
        
    def get_fairness_regret(self, pro):
        fairness_regret = fairness_regret_each_slot(pro,self.mean_reward,self.K,self.L)
        return fairness_regret

    def get_average_selection(self):
        average_selection = average_selection_each_slot(self.number_of_pull,self.K,self.L)
        return average_selection

    def reset(self):
        self.number_of_pull = np.ones(self.K)
        self.sum_of_received_reward = np.zeros(self.K)
        self.number_of_received_feedback = np.ones(self.K)
        self.arms=[Arm(self.reward_type, self.delay_type,self.mean_reward[a],self.p[a]) for a in range(self.K)]
        self.outstanding_reward = []
        self.arriving_round = []
        for i in range(self.K):
            reward, delay = self.arms[i].pull(i)
            self.outstanding_reward.append([reward])
            self.arriving_round.append([delay])