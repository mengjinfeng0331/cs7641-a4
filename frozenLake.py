import gym,os
import numpy as np
import pandas as pd
import mdptoolbox.example
import algorithms
import mdptoolbox
from gym.envs.toy_text.frozen_lake import generate_random_map
import plot
################
## forest definition
SIZE = [ 4, 8, 10, 20, 30]
discount=0.9
gamma_list = [0.6,0.7,0.8,0.9,0.95, 0.98, 0.999]

SEED = 10
RESULT_DIR = 'results'
df = pd.DataFrame()
np.random.seed(SEED)

################
if not os.path.isdir(RESULT_DIR): os.mkdir(RESULT_DIR)
def compose_TR(env):
    A = env.env.nA
    S = env.env.nS
    rawP = env.env.P
    P = np.zeros((A,S,S))
    R = np.zeros((A,S))
    
    for aa in range(A):
        for ss in range(S):
            for item in rawP[ss][aa]:
                prob = item[0]
                end_state = item[1]
                reward = item[2]
#                done = item[3]
                P[aa,ss, end_state] = prob
                if R[aa,end_state] !=1:
                    R[aa,end_state] = reward
    return P,R

## value iteration
vi_history_dict = {}
vi_history_dict_epsilon = {}

for s in SIZE:
    for gamma in gamma_list:
        random_map = generate_random_map(size=s, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map)
        env.reset()
        P,R= compose_TR(env)
        print('size :',s)
        vi = algorithms.valueIteration(P,R,gamma ,verbose=False)
        vi.run()
#        vi_history_dict[s**2] = vi.history['diff']
        if gamma == 0.9: vi_history_dict[s**2] = vi.history['diff']
        if s == 20: vi_history_dict_epsilon[gamma] = vi.history['diff']
        
        index = 'frozen_VI_s{}_e{}'.format(s,gamma)
        df.loc[index,'alg'] = 'VI'
        df.loc[index,'n_state'] = s**2
        df.loc[index,'gamma'] = gamma
        df.loc[index,'iter'] = vi.iterations
        df.loc[index,'time'] =vi.total_time 
        string = [str(i) for i in vi.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))
plot.plot_vipi_history(vi_history_dict,'VI-training-Error', output_file='frozen_VI_vdiff.png') 
plot.plot_vipi_history(vi_history_dict_epsilon,'VI-training-Error',cat ='gamma', output_file='frozen_VI_vdiff_epsilon.png') 
    
    
## Policy iteration
pi_vdiff_dict = {}
pi_pdiff_dict = {}
for s in SIZE:
    for gamma in gamma_list:
    
        random_map = generate_random_map(size=s, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map)
        env.reset()
        P,R= compose_TR(env)
        print('size :',s)
#        s = s**2
        pi = algorithms.policy_Iteration(P,R, gamma)
        pi.run()
        file_name = RESULT_DIR + os.sep + 'frozen_PI_s{}.json'.format(s)
        pi.history2file(file_name)
        if gamma == 0.9:
            pi_vdiff_dict[s**2] = pi.history['v_diff']
            pi_pdiff_dict[s**2] = pi.history['p_diff']
        index = 'frozen_PI_s{}_e{}'.format(s,gamma)
        df.loc[index,'alg'] = 'PI'
        df.loc[index,'n_state'] = s**2
        df.loc[index,'gamma'] = gamma
        df.loc[index,'iter'] = pi.iterations
        df.loc[index,'time'] =pi.total_time 
        string = [str(i) for i in pi.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))

plot.plot_vipi_history(pi_vdiff_dict,'PI-Value-Error', output_file='frozen_PI_vdiff.png') 
plot.plot_vipi_history(pi_pdiff_dict,'PI-Policy-Error', output_file='frozen_PI_pdiff.png') 


#########################################################
## Q-learning
ql_vdiff_dict={}

for s in SIZE:
    for gamma in gamma_list:
        print('size :{}, gamma:{}'.format(s,gamma))
        
        random_map = generate_random_map(size=s, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map)
        env.reset()
        P,R= compose_TR(env)
#        ql = algorithms.QLearning(P, R.T, gamma, env=env)
#        ql.run_frozen()
        ql = algorithms.QLearning(P, R.T, gamma)
        ql.run_forest()
        
        if gamma == 0.9:
            ql_vdiff_dict[s**2] = ql.mean_discrepancy
        index = 'QL_s{}_e{}'.format(s**2,gamma)
        df.loc[index,'alg'] = 'QL'
        df.loc[index,'n_state'] = s**2
        df.loc[index,'gamma'] = gamma
        df.loc[index,'iter'] = ql.iter
        df.loc[index,'time'] =ql.time 
        string = [str(i) for i in ql.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))
        
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

avg = movingaverage(ql.discrepancy,20)
ql_vdiff_dict_20 = { 64 : ql.discrepancy, '64-avg':avg}

plot.plot_vipi_history(ql_vdiff_dict_20,'QL-Value-Error', output_file='frozen_QL_vdiff.png') 


df.to_csv('frozenlake_df.csv')    
df.n_state = df.n_state.astype(int) 
df.iter = df.iter.astype(int)  
df.to_csv('frozen_df.csv')

plot.plot_part1_df(df)
plot.plot_data_heatmap(df[df.alg=='VI'], 'frozen_VI-df.PNG')

