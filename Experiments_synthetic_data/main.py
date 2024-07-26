import numpy as np
from utilities import *
from algorithms import *
from run import *
import plots
L = 3   # the number of selected arm at each round 
K = 7   # the number of arm
mean_reward = [0.3,0.5,0.7,0.9,0.8,0.6,0.4]
fixed_delays_set = [0,50,100,150,200,250,300]
p = [0.05]*K # success pro of Geometric delay distributions
eta = np.random.uniform(0,1,K) # tail index of alpha-Pareto delay distributions
packet_loss_pro = np.random.uniform(0.3,0.8,K) # pro of observing reward of packet-loss delay distributions
c_set=[2,4,6,8] # the gradient of the merit function

total_simulations = 120

reward_type = 'Bernoulli_reward'

#Experiments under Geometric feedback delays
T = 20000 # time horizon
policies = []
delay_type = 'Geometric_delay'
parameter = p
policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Geometric_delays',c=4) # save data
plots.Simulation_Geometric_delays()

# Experiments under fixed feedback delays
T = 100000 # time horizon
policies = []
delay_type = 'Fixed_delay' 
for delay in fixed_delays_set:
    parameter = [delay]*K
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Fixed_delays',c=4)
plots.Simulation_Fixed_delays()

# Experiments under alpha-Pareto feedback delays
T = 40000   # time horizon
policies = []
delay_type = 'Pareto_delay'
parameter = eta
policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Alpha_pareto_delays',c=4)
plots.Simulation_Alpha_pareto_delays()

# Experiments with packet-loss delays
T = 40000   # time horizon
policies = []
reward_type = 'Bernoulli_reward'
delay_type = 'Packetloss_delay'
parameter = packet_loss_pro
policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Packet-loss_delays',c=4)
plots.Simulation_Packet_loss_delays()

# Experiments under biased feedback delays
T = 40000   # time horizon
policies = []
delay_type = 'Fixed_delay_RD'
biased_delays = 6000
policies.append(OP_FCUCB_D(L, K,mean_reward,delay_type,reward_type,biased_delays,c=4))
policies.append(OP_FCTS_D(L, K,mean_reward,delay_type,reward_type,biased_delays,c=4))
policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,biased_delays,c=4))
policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,biased_delays,c=4))
policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Biased_delays',c=4)
plots.Simulation_Biased_delays()

# Experiments on merit functions with different c under different types of delays

# Fixed delays
T = 40000   # time horizon
policies = []
delay_type = 'Fixed_delay'
parameter = [200]*K
for c in c_set:
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Fixed_delays_Different_c',c)
plots.Simulation_Fixed_delays_Different_c()

# Geometric delays
T = 40000   # time horizon
policies = []
delay_type = 'Geometric_delay'
parameter = p
for c in c_set:
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Geometric_delays_Different_c',c)
plots.Simulation_Geometric_delays_Different_c()

# Pareto delays
T = 40000   # time horizon
policies = []
delay_type = 'Pareto_delay'
parameter = eta
for c in c_set:
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Pareto_delays_Different_c',c)
plots.Simulation_Alpha_Pareto_delays_Different_c()

# Packet-loss delays
T = 40000   # time horizon
policies = []
delay_type = 'Packetloss_delay'
parameter = packet_loss_pro
for c in c_set:
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Packet_loss_delays_Different_c',c)
plots.Simulation_Packet_loss_delays_Different_c()

# Biased delays
T = 40000   # time horizon
policies = []
delay_type = 'Fixed_delay_RD'
for c in c_set:
    policies.append(OP_FCUCB_D(L, K,mean_reward,delay_type,reward_type,c))
    policies.append(OP_FCTS_D(L, K,mean_reward,delay_type,reward_type,c))
    policies.append(FCUCB_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(FCTS_D(L, K,mean_reward,delay_type,reward_type,parameter,c))
    policies.append(CUCB_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(MP_TS_D(L, K, mean_reward,delay_type,reward_type,parameter,c=4))
    policies.append(FGreedy_D(L, K,mean_reward,delay_type,reward_type,parameter,c=4))
Experiments(policies,mean_reward,total_simulations,L,K,T,'Biased_delays_Different_c',c)
plots.Simulation_Biased_delays_Different_c()




