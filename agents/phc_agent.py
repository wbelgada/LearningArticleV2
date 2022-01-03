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
    def __init__(self, initialStrategy = (0.5, 0.5), gammma = 0.9, delta = 0.0001):
        self.timeStep = 0
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)
        self.gamma = gammma
        self.strategy = list(initialStrategy)
        self.actions = np.arange(len(self.strategy))
        self.lengthOfAction = len(self.actions)
        self.reward = 0.0

        self.actionValues = np.zeros((self.lengthOfAction))
        # self.actionRewards = np.zeros((2))
        # self.currentActionIndex = 0
        # self.nextActionIndex = 0
        self.currentAction = np.random.choice(self.actions)
        self.currentReward = 0
        self.maxAction = np.random.choice(self.actions)
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)
        self.deltaAction = np.zeros((self.lengthOfAction))
        self.deltaActionTop = np.zeros((self.lengthOfAction))
        self.delta = delta
        self.pi_history = [1/len(self.strategy)]
        self.pi_history2 = [1/len(self.strategy)]

    def initialSelfStrategy (self):
        for i in range(self.lengthOfAction):
            self.strategy[i] = 1 / self.lengthOfAction

    def initialActionValues (self):
        for i in range(self.lengthOfAction):
            self.actionValues[i] = 0

    def act (self):
        if np.random.random() < self.EPSILON:
            self.currentAction = np.random.choice(self.actions)
        else:
            self.currentAction = np.random.choice(self.actions, p = self.strategy)

    def chooseActionWithFxiedStrategy (self):
        self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def getCurrentAction (self):
        return self.currentAction

    def setReward (self, agentReward):
        self.currentReward = agentReward

    def updateActionValues (self):
        self.actionValues[self.currentAction] = (1 - self.alpha) * self.actionValues[self.currentAction] \
                                                 + self.alpha * (self.currentReward + self.gamma * np.amax(self.actionValues[:]))
    def updateStrategy (self):
        self.maxAction = np.argmax(self.actionValues)
        for i in range(self.lengthOfAction):
            self.deltaAction[i] = np.min([self.strategy[i], self.delta / (self.lengthOfAction - 1)])
        self.sumDeltaAction = 0
        for action_i in [action_j for action_j in self.actions if action_j != self.maxAction]:
            self.deltaActionTop[action_i] = -self.deltaAction[action_i]
            self.sumDeltaAction += self.deltaAction[action_i]
        self.deltaActionTop[self.maxAction] = self.sumDeltaAction
        for i in range(self.lengthOfAction):
            self.strategy[i] += self.deltaActionTop[i]

        # if self.currentAction != self.maxAction:
        #     self.deltaActionTop[self.currentAction] = -self.deltaAction[self.currentAction]
        # else:
        #     self.sumDeltaAction = 0
        #     for action_i in [action_j for action_j in self.actions if action_j != self.currentAction]:
        #         self.sumDeltaAction += self.deltaAction[action_i]
        #     self.deltaActionTop[self.currentAction] = self.sumDeltaAction
        # self.strategy[self.currentAction] += self.deltaActionTop[self.currentAction]
        self.pi_history.append(self.strategy[0])
        self.pi_history2.append(self.strategy[1])

    def updateTimeStep (self):
        self.timeStep += 1

    def updateEpsilon (self):
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)

    def updateAlpha (self):
        pass
        #self.alpha = 1 / (10 + 0.00001 * self.timeStep)