import random

import numpy as np

class WoLFAgent():
    """
        Policy hill-climbing algorithm(PHC)
        http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
    """
    def __init__(self, gamma=0.999, alpha=0.1, delta=0.0001, actions=None,  nb_states = 1, high_delta=0.004, low_delta=0.002):
        random.seed(25)
        self.alpha = alpha
        self.nb_states=nb_states
        self.gamma = gamma
        self.actions = actions  
        self.last_action_id = None
        self.q_values = self._init_q_values()
        self.pi = [[(1.0/len(actions)) for idx in range(len(actions))] for _ in range(self.nb_states)]
        self.pi_average = [[(1.0/len(actions)) for idx in range(len(actions))] for _ in range(self.nb_states)]
        self.high_delta = high_delta
        self.row_delta = low_delta 

        self.reward_history = []
        self.conter = 0

    def _init_q_values(self):
        q_values = {}
        q_values = np.zeros(shape=(self.nb_states, len(self.actions)))
        return q_values

    def act(self, s, q_values=None):
        #action_id = np.random.choice(np.arange(len(self.actions)), p=self.pi[s])
        action_id = random.choices(np.arange(len(self.actions)), weights=self.pi[s])[0]
        self.last_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, reward, obs, nextobs):
        self.reward_history.append(reward)
        self.q_values[obs][self.last_action_id] = ((1.0 - self.alpha) * self.q_values[obs][self.last_action_id]) \
                                             + (self.alpha * (reward + self.gamma * (self.q_values[nextobs].max())))
        self._update_pi_average(obs)
        self._update_pi(obs)

    def _update_pi_average(self, s):
       self.conter += 1
       for aidx, _ in enumerate(self.pi[s]):
           self.pi_average[s][aidx] = self.pi_average[s][aidx] + (1/self.conter)*(self.pi[s][aidx]-self.pi_average[s][aidx])
           if self.pi_average[s][aidx] > 1: self.pi_average[s][aidx] = 1
           if self.pi_average[s][aidx] < 0: self.pi_average[s][aidx] = 0

    def _update_pi(self, s):
       delta = self.decide_delta(s)
       max_action_id = np.argmax(self.q_values[s])
       for aidx, _ in enumerate(self.pi[s]):
           if aidx == max_action_id:
               update_amount = delta
           else:
               update_amount = ((-delta)/(len(self.actions)-1))
           self.pi[s][aidx] = self.pi[s][aidx] + update_amount
           if self.pi[s][aidx] > 1: self.pi[s][aidx] = 1
           if self.pi[s][aidx] < 0: self.pi[s][aidx] = 0

    def decide_delta(self, s):
        """
            comfirm win or lose 
        """
        expected_value = 0
        expected_value_average = 0
        for aidx, _ in enumerate(self.pi[s]):
            expected_value += self.pi[s][aidx]*self.q_values[s][aidx]
            expected_value_average += self.pi_average[s][aidx]*self.q_values[s][aidx]

        if expected_value > expected_value_average: # win
            return self.row_delta
        else:   # lose
            return self.high_delta
