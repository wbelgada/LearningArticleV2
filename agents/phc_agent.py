import numpy as np
def generateRandomFromDistribution (distribution):
    randomIndex = 0
    randomSum = distribution[randomIndex]
    randomFlag = np.random.random_sample()
    while randomFlag > randomSum:
        randomIndex += 1
        randomSum += distribution[randomIndex]
    return randomIndex

class PHCAgent:
    def __init__(self, initialStrategy = (0.5, 0.5), nb_actions=2, nb_states=1, gammma = 0.9, delta = 0.0001):
        self.timeStep = 0
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)
        self.gamma = gammma
        self.nb_states = nb_states
        self.strategy = [list(initialStrategy) for _ in range(nb_states)]
        self.actions = np.arange(nb_actions)
        self.lengthOfAction = nb_actions

        self.reward = 0.0

        self.q_values = np.zeros(shape=(self.nb_states,self.lengthOfAction))
        # self.actionRewards = np.zeros((2))
        # self.currentActionIndex = 0
        # self.nextActionIndex = 0
        self.currentAction = np.random.choice(self.actions)
        self.currentReward = 0
        self.maxAction = np.random.choice(self.actions)
        self.EPSILON = 1 / (1 + 0.0001 * self.timeStep)
        self.deltaAction = np.zeros((self.lengthOfAction))
        self.deltaActionTop = np.zeros((self.lengthOfAction))
        self.delta = delta

    def initialSelfStrategy (self,s):
        for i in range(self.lengthOfAction):
            self.strategy[s][i] = 1 / self.lengthOfAction

    def initialActionValues (self,s):
        for i in range(self.lengthOfAction):
            self.q_values[s][i] = 0

    def act (self, s):
        if np.random.random() < self.EPSILON:
            self.currentAction = np.random.choice(self.actions)
        else:
            self.currentAction = np.random.choice(self.actions, p = self.strategy[s])

    def chooseActionWithFxiedStrategy (self,s):
        self.currentAction = self.actions[generateRandomFromDistribution(self.strategy[s])]

    def getCurrentAction (self):
        return self.currentAction

    def setReward (self, agentReward):
        self.currentReward = agentReward

    def updateActionValues (self,obs, nextObs):
        self.q_values[obs][self.currentAction] = (1 - self.alpha) * self.q_values[obs][self.currentAction] \
                                            + self.alpha * (self.currentReward + self.gamma * np.max(self.q_values[nextObs][:]))
    def updateStrategy (self, s):
        self.maxAction = np.argmax(self.q_values[s])
        for i in range(self.lengthOfAction):
            self.deltaAction[i] = np.min([self.strategy[s][i], self.delta / (self.lengthOfAction - 1)])
        self.sumDeltaAction = 0
        for action_i in [action_j for action_j in self.actions if action_j != self.maxAction]:
            self.deltaActionTop[action_i] = -self.deltaAction[action_i]
            self.sumDeltaAction += self.deltaAction[action_i]
        self.deltaActionTop[self.maxAction] = self.sumDeltaAction
        for i in range(self.lengthOfAction):
            self.strategy[s][i] += self.deltaActionTop[i]

        # if self.currentAction != self.maxAction:
        #     self.deltaActionTop[self.currentAction] = -self.deltaAction[self.currentAction]
        # else:
        #     self.sumDeltaAction = 0
        #     for action_i in [action_j for action_j in self.actions if action_j != self.currentAction]:
        #         self.sumDeltaAction += self.deltaAction[action_i]
        #     self.deltaActionTop[self.currentAction] = self.sumDeltaAction
        # self.strategy[self.currentAction] += self.deltaActionTop[self.currentAction]

    def updateTimeStep (self):
        self.timeStep += 1

    def updateEpsilon (self):
        self.EPSILON = 1 / (1 + 0.00001 * self.timeStep)

    def updateAlpha (self):
        pass
        #self.alpha = 1 / (10 + 0.00001 * self.timeStep)