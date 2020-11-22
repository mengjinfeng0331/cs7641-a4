import numpy as np
import time
import utils
import math

class valueIteration:
    def __init__(self, transition, reward, gamma=0.96, n_episodes=1000, threshold=0.0001,verbose=True):
        self.transition = transition
        assert len(transition.shape) == 3
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.A = transition.shape[0]
        self.S = transition.shape[1]
#        print('Shape S:{}, A:{}'.format(self.S,self.A))
        assert reward.shape ==(self.A, self.S)
        self.reward = reward
        self.Q = np.zeros_like(reward)
        self.V = np.zeros((self.S,1))
        self.policy = np.zeros((self.S))
        self.threshold = threshold
        self.history = { 'diff':[]}
        self.verbose= verbose
        self.total_time = 0
        self.iterations = 0
        
    def diff(self, old_Q, new_Q):
        return  np.linalg.norm(old_Q- new_Q)
    
    def getSpan(self,array):
        return array.max() - array.min()

    def history2file(self, output_file):
        utils.write2json(output_file, self.history)
        
    def run(self):
        total_start = time.time()
        for i in range(self.n_episodes):
            Q = np.zeros((self.A, self.S))
            prev_V = self.V.copy()
            for aa in range(self.A):
                Q[aa] = self.reward[aa] + self.gamma * self.transition[aa].dot(self.V).flatten()
                
            self.V = np.max(Q,axis=0).reshape(-1,1)
            
            ## check convergence
            diff =self.diff(prev_V, self.V)
            self.history['diff'].append(diff)
            
            if self.verbose:
                print('#{}, diff:{}, V:{}'.format(i,diff,self.V.flatten()))
            if diff < self.threshold:
#                print('convergence due to meet threshold')
                break 
#            print("Q:",Q)
#            time.sleep(1)
        self.policy = np.argmax(Q,axis=0)    
        self.iterations = i
        total_end = time.time()
        self.total_time = total_end - total_start
    
class policy_Iteration:
    def __init__(self, transition, reward, gamma=0.96, n_episodes=1000, threshold=0.0001):
        self.transition = transition
        assert len(transition.shape) == 3
        self.reward = reward
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.A = transition.shape[0]
        self.S = transition.shape[1]
#        print('Shape S:{}, A:{}'.format(self.S,self.A))
        assert reward.shape ==(self.A, self.S)
        self.Q = np.zeros_like(reward)
        self.V = np.zeros((self.S))
        self.policy = np.zeros((self.S))
        self.threshold = threshold
        self.history = {  'v_diff':[],'p_diff':[]}
        self.iterations = 0
        self.total_time = 0
        
    def diff(self, old_Q, new_Q):
        return  np.linalg.norm(old_Q- new_Q)
    
    def history2file(self, output_file):
        utils.write2json(output_file, self.history)   
        
    def gen_policyPR(self, policy):
        P_policy = np.zeros((self.S,self.S))
        R_policy = np.zeros((self.S))
        
        for aa in range(self.A):
            index = (policy==aa).nonzero()[0]
            P_policy[index,:] = self.transition[aa][index,:]
            R_policy[index] = self.reward[aa][index]
        return P_policy, R_policy
    
    def _bellmanOperator(self):
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.reward[aa] + self.gamma * self.transition[aa].dot(self.V)
        return (Q.argmax(axis=0), Q.max(axis=0))
        
    def policy_evaluation(self, policy, max_iter= 1000):
        old_V = np.zeros(self.S)
        P_policy, R_policy = self.gen_policyPR(policy)
        
        iterations=0
        converged = False
        while not converged:
            new_V = R_policy + self.gamma * P_policy.dot(old_V)
            
            variation = np.absolute(old_V - new_V).max()
            if variation < self.threshold:
                converged = True
            
            if iterations > max_iter:
                converged = True
            iterations +=1
            old_V = new_V
        self.V = new_V
    
    def policy_generation(self, Q):
        new_policy = np.argmax(Q,axis=0)
        return new_policy
    
    def policy_diff(self, old_policy, new_policy):
        return (old_policy != new_policy).sum()
        
    def run(self):
        old_policy = np.random.randint(self.A,size=self.S)
        total_start = time.time()
        
        for i in range(self.n_episodes):
            old_v = self.V.copy()
            self.policy_evaluation(old_policy)
            new_policy,_ = self._bellmanOperator()
            
            v_diff = self.diff(old_v, self.V)
            p_diff = self.policy_diff(old_policy, new_policy)
#            if i% 40 == 0: print('#{}, diff:{}'.format(i, diff))
            old_policy = new_policy
            
            self.history['v_diff'].append(v_diff)
            self.history['p_diff'].append(float(p_diff))
            if p_diff == 0:
#                print('policy converge at {}, policy:{}'.format(i,new_policy))
                break 
        self.iterations  =i
        total_end = time.time()
        self.total_time = total_end - total_start
            
        self.policy = new_policy

class QLearning:
    def __init__(self, transition, reward, gamma, env=None, n_iter=100000,
                 skip_check=False):
        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."
        self.env = env
        self.P = transition
        assert len(transition.shape) == 3
        self.reward = reward
        self.gamma = gamma
        self.A = transition.shape[0]
        self.S = transition.shape[1]
#        print('Shape S:{}, A:{}'.format(self.S,self.A))
        assert reward.shape ==(self.S, self.A)
        self.iter = 0
        self.R = reward
        self.threshold = 0.0005
        # Initialisations
        self.Q = np.zeros((self.S, self.A))
        self.V = np.zeros((self.S))
        self.diff_list  = []
        self.time = 0
        self.epsilon = 0.9
        self.epsilon_decay = 0.98
        self.mean_discrepancy = []
        self.discrepancy = []
    def simulator():
        pass
    
    def get_action(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.A)
        action = np.argmax(self.Q[state,:])
        return action
    
    def diff(self, old_Q, new_Q):
        return  np.linalg.norm(old_Q- new_Q) 
    
    def run_forest(self):
            # Run the Q-learning algoritm.
            start = time.time()
            # initial state choice
            s = np.random.randint(0, self.S)
            discrepancy =[]
            
            for n in range(1, self.max_iter + 1):
                old_v = self.V.copy()
                # Reinitialisation of trajectories every 100 transitions
                if (n % 10) == 0:
                    s = np.random.randint(0, self.S)
    
                # Action choice : greedy with increasing probability
                # probability 1-(1/log(n+2)) can be changed  
                a = self.get_action(s)
                # Simulating next state s_new and reward associated to <s,s_new,a>
                p_s_new = np.random.random()
                p = 0
                s_new = -1
                while (p < p_s_new) and (s_new < (self.S - 1)):
                    s_new = s_new + 1
                    p = p + self.P[a][s, s_new]
    
                try:
                    r = self.R[a][s, s_new]
                except IndexError:
                    try:
                        r = self.R[s, a]
                    except IndexError:
                        r = self.R[s]

                # Updating the value of Q
                # Decaying update coefficient (1/sqrt(n+2)) can be changed
                delta = r + self.gamma * self.Q[s_new, :].max() - self.Q[s, a]
                dQ = (1 / math.sqrt(n + 2)) * delta
                self.Q[s, a] = self.Q[s, a] + dQ
    
                # current state is updated
                s = s_new
    
                # Computing and saving maximal values of the Q variation
                discrepancy.append(np.absolute(dQ))
                self.discrepancy.append(dQ)
                # Computing means all over maximal Q variations values
                if len(discrepancy) == 100:
                    self.mean_discrepancy.append(np.mean(discrepancy))
                    if np.mean(discrepancy) < self.threshold and self.iter > 10000:
                        discrepancy = []
#                        print('Converge')
                        break
                    else:
                        discrepancy = []
#                self.mean_discrepancy.append(np.absolute(dQ))
                # compute the value function and the policy
                self.V = self.Q.max(axis=1)
                self.policy = self.Q.argmax(axis=1)
                self.iter +=1
            end =time.time()
            self.time = end-start       
            
    def update(self, state1, action1, reward, state2):
        error = reward + self.gamma * np.max(self.Q[state2,:]) - self.Q[state1,action1]
        self.Q[state1,action1] = self.Q[state1,action1] + self.alpha * (error)
        
    def next_(self, state, action):
        prob = self.P[action, state]
        prob = prob/sum(prob)
        new_state = np.random.choice(self.S, p=prob)
        reward = self.R[new_state, action]
        if max(prob) ==1: 
            done =True
        else:
            done =False
        return new_state, reward ,done
    
    def run_frozen(self):
        # Run the Q-learning algoritm.
        start = time.time()
        self.iter = 0
        discrepancy = []
        for n in range(1, self.max_iter + 1):
            old_v = self.V.copy()
            
            # Reinitialisation of trajectories every 100 transitions
            s = self.env.reset()
            
            while True:
                # Action choice : greedy with increasing probability
                # probability 1-(1/log(n+2)) can be changed  
                a = self.get_action(s)
                # Simulating next state s_new and reward associated to <s,s_new,a>
                s_new, r, done = self.next_(s, a)
                
                # Updating the value of Q
                # Decaying update coefficient (1/sqrt(n+2)) can be changed
                delta = r + self.gamma * self.Q[s_new, :].max() - self.Q[s, a]
                dQ = (1 / math.sqrt(n + 2)) * delta
                self.Q[s, a] = self.Q[s, a] + dQ
    
                # current state is updated
                s = s_new
                
#                print('state:{}, action:{},reward:{}, s_new:{}'.format(s,a,r, s_new))
                ## end of episode
                if r == 1 or done: 
#                        print('{} converge'.format(self.iter))
                    break
                
            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.iter +=1
            
            diff = self.diff(self.V, old_v)
            discrepancy.append(diff)
            
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(np.mean(discrepancy))
                if np.mean(discrepancy) < self.threshold:
                    print('converge break', discrepancy)
                    break
                    discrepancy = []
                else:
                    discrepancy = []
                
            self.diff_list.append(diff)
            print('{},  diff:{}'.format(self.iter, diff))
#            if np.mean(discrepancy) < self.threshold:
#                break
            
        self.policy = self.Q.argmax(axis=1)

        end =time.time()
        self.time = end-start 
            