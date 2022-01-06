import random
from copy import deepcopy

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
        if sum(self.pi[s]) > 1.001:
            x=2

        if self.pi[s][max_action_id] + delta > 1:
            self.pi[s][max_action_id] = 1
            for i in range(len(self.pi[s])):
                if i != max_action_id:
                    self.pi[s][i] = 0
        else:

            self.pi[s][max_action_id] = self.pi[s][max_action_id] + delta
            t = [self.pi[s][i] for i in range(len(self.pi[s])) if i != max_action_id]
            sumt = sum(t)
            norm = [t[i] / sumt for i in range(len(t))]
            notx = sumt
            notx = notx - delta
            newProba = [(t[i] * notx)/sumt for i in range(len(norm))]
            j = 0
            for i in range(len(self.pi[s])):
                if i != max_action_id:
                    self.pi[s][i] = newProba[j]
                    j = j + 1




        """for aidx, _ in enumerate(self.pi[s]):
            if aidx == max_action_id:
                update_amount = delta
            else:
                update_amount = ((-delta)/(len(self.actions)-1))
            self.pi[s][aidx] = self.pi[s][aidx] + update_amount"""

        '''picopy = deepcopy(self.pi[s])
        for aidx, _ in enumerate(self.pi[s]):
            if aidx == max_action_id:
                if picopy[aidx] > 1:
                    offset = picopy[aidx]-1
                    for i, _ in enumerate(self.pi[s]):
                        if i != aidx:
                            self.pi[s][i] -= offset/(len(self.actions)-1)
            else:
                if picopy[aidx] < 0:
                    offset = picopy[aidx]
                    for i, _ in enumerate(self.pi[s]):
                        if i != aidx:
                            self.pi[s][i] += offset/(len(self.actions)-1)
        for aidx, _ in enumerate(self.pi[s]):
            if aidx == max_action_id:
                self.pi[s][aidx] = min(1.0, self.pi[s][aidx])
            else:
                self.pi[s][aidx] = max(0.0, self.pi[s][aidx])'''



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
