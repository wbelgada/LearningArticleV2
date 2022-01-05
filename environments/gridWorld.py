import numpy as np

class GridWorldEnv:

    def __init__(self):
        self.agent1 = 6
        self.agent2 = 8

    def step(self, action1, action2):
        reward1 = -1
        reward2 = -1
        done = False
        new1, north1 = self.actOne(action1, 1)
        new2, north2 = self.actOne(action2, 2)
        moved1 = True
        moved2 = True
        if self.agent1 == new1:
            moved1 = False
        if self.agent2 == new2:
            moved2 = False

        if new1 != new2:
            if moved1:
                reward1 = self.rew(1, new1, north1)
            if moved2:
                reward2 = self.rew(2, new2, north2)

            self.agent1 = new1
            self.agent2 = new2
        if self.agent1 == 1 or self.agent2 == 1:
            done=True
        return self.agent1, self.agent2, reward1, reward2, done




    def actOne(self, action, agent):
        north=False
        if(agent == 1):
            currentAgent = self.agent1
        else:
            currentAgent = self.agent2
        if (currentAgent == 6 or currentAgent == 8) and action == 0: #north
            north = True
            if np.random.random() < 0.5:
                currentAgent -= 3
        elif action == 0: #north
            if currentAgent!=0 and currentAgent != 1 and currentAgent!=2:
                currentAgent-=3
        elif action == 1: #south
            if currentAgent != 6 and currentAgent != 7 and currentAgent != 8:
                currentAgent+=3
        elif action == 2: #east
            if currentAgent != 2 and currentAgent != 5 and currentAgent != 8:
                currentAgent+=1
        elif action == 3: #west
            if currentAgent != 0 and currentAgent != 3 and currentAgent != 6:
                currentAgent -=1

        return currentAgent, north

    def reset(self):
        self.__init__()
        return self.agent1, self.agent2

    def rew(self,cur, pos, north):
        if cur == 1:
            return self.getRew(self.agent1, pos, north)
        else :
            return self.getRew(self.agent2, pos, north)


    def getRew(self, old, new, north):
        if old == 6 or old == 8:
            if new == 3 or new == 5 or new == 7:
                return 0.3
        elif old == 3 or old == 5 or old == 7:
            if new == 6 or new == 8:
                return -0.3
            elif new == 4 or new == 0 or new == 2:
                return 0.6
        elif old == 4 or old == 0 or old == 2:
            if new == 1:
                return 1
            elif new == 3 or new == 5 or new == 7:
                return -0.8

        if old == 6 or old == 8 and north:
            if new == 6 or new == 8:
                return 0.5
        if old == new:
            return -1