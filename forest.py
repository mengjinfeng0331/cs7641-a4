import numpy as np
import mdptoolbox.example
import algorithms
import mdptoolbox
import os
import pandas as pd
import plot

################
## forest definition
FORESET_SIZE = [ 3, 10, 20, 50, 100]
gamma_list = [0.6,0.7,0.8,0.9,0.95, 0.98, 0.999]
discount=0.9
RESULT_DIR = 'results'
df = pd.DataFrame()
np.random.seed(10)
################
if not os.path.isdir(RESULT_DIR): os.mkdir(RESULT_DIR)
##################################################################
## VI
vi_history_dict = {}
vi_history_dict_epsilon = {}
for s in FORESET_SIZE:
    for epsilon in gamma_list:
        P, R = mdptoolbox.example.forest(S=s)
        vi = algorithms.valueIteration(P, R.T, epsilon, verbose=False)
        vi.run()
        if epsilon == 0.9: vi_history_dict[s] = vi.history['diff']
        if s == 20: vi_history_dict_epsilon[epsilon] = vi.history['diff']
        index = 'VI_s{}_e{}'.format(s,epsilon)
        df.loc[index,'alg'] = 'VI'
        df.loc[index,'n_state'] = s
        df.loc[index,'gamma'] = epsilon
        df.loc[index,'iter'] = vi.iterations
        df.loc[index,'time'] =vi.total_time 
        string = [str(i) for i in vi.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))
        
plot.plot_vipi_history(vi_history_dict,'VI-training-Error', output_file='part1_VI_vdiff.png') 
plot.plot_vipi_history(vi_history_dict_epsilon,'VI-training-Error',cat ='gamma', output_file='part1_VI_vdiff_epsilon.png') 


#################################################    
## PI
pi_vdiff_dict = {}
pi_pdiff_dict = {}
for s in FORESET_SIZE:
    for epsilon in gamma_list:
        P, R = mdptoolbox.example.forest(S=s)
        pi = algorithms.policy_Iteration(P,R.T, epsilon)
        pi.run()
        if epsilon == 0.9:
            pi_vdiff_dict[s] = pi.history['v_diff']
            pi_pdiff_dict[s] = pi.history['p_diff']
        index = 'PI_s{}_e{}'.format(s,epsilon)
        df.loc[index,'alg'] = 'PI'
        df.loc[index,'n_state'] = s
        df.loc[index,'gamma'] = epsilon
        df.loc[index,'iter'] = pi.iterations
        df.loc[index,'time'] =pi.total_time 
        string = [str(i) for i in pi.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))

plot.plot_vipi_history(pi_vdiff_dict,'PI-Value-Error', output_file='part1_PI_vdiff.png') 
plot.plot_vipi_history(pi_pdiff_dict,'PI-Policy-Error', output_file='part1_PI_pdiff.png') 

#########################################################
## Q-learning
ql_vdiff_dict = {}
for s in FORESET_SIZE:
    for gamma in gamma_list:
        P, R = mdptoolbox.example.forest(S=s)
        ql = algorithms.QLearning(P, R, gamma)
        ql.run_forest()
        if gamma == 0.9:
            ql_vdiff_dict[s] = ql.mean_discrepancy
        index = 'QL_s{}_e{}'.format(s,gamma)
        df.loc[index,'alg'] = 'QL'
        df.loc[index,'n_state'] = s
        df.loc[index,'gamma'] = gamma
        df.loc[index,'iter'] = ql.iter
        df.loc[index,'time'] =ql.time 
        string = [str(i) for i in ql.policy.tolist()]
        df.at[index,'policy'] = str(','.join(string))
        
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
        
avg = movingaverage(ql_vdiff_dict[20],20)
ql_vdiff_dict_20 = { 20 : ql_vdiff_dict[20], '20_avg':avg}

plot.plot_vipi_history(ql_vdiff_dict_20,'QL-Value-Error', output_file='forest_QL_vdiff.png') 

df.n_state = df.n_state.astype(int) 
df.iter = df.iter.astype(int)  
df.to_csv('forest_df.csv')

plot.plot_part1_df(df)
plot.plot_data_heatmap(df[df.alg=='VI'], 'VI-df.PNG')