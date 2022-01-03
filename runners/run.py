import numpy as np
import matplotlib.pyplot as plt

from agents.phc_agent import PHCAgent
from agents.wolf_agent import WoLFAgent
from matrix_game import MatrixGame

if __name__ == '__main__':
    nb_episode = 1000000
    evaluate_every = 100000

    actions = np.arange(2)
    agent1 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002)
    agent2 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002)

    #agent1 = PHCAgent(delta = 0.0004,initialStrategy=(1/len(actions) for i in range(len(actions))))
    #agent2 = PHCAgent(delta = 0.0004,initialStrategy=(1/len(actions) for i in range(len(actions))))

    #agent1 = PHCAgent(delta = 0.0004,initialStrategy=(1.0, 0.0))
    #agent2 = PHCAgent(delta = 0.0004,initialStrategy=(1.0, 0.0))

    game = MatrixGame("MP")
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        """agent1.act()
        action1 = agent1.currentAction
        agent2.act()
        action2 = agent2.currentAction
        """
        _, r1, r2 = game.step(action1, action2)

        agent1.observe(reward=r1)
        agent2.observe(reward=r2)
        """agent1.setReward(r1)
        agent1.updateActionValues()
        agent1.updateStrategy()
        agent1.updateTimeStep()
        agent1.updateEpsilon()
        agent1.updateAlpha()

        agent2.setReward(r2)
        agent2.updateActionValues()
        agent2.updateStrategy()
        agent2.updateTimeStep()
        agent2.updateEpsilon()
        agent2.updateAlpha()"""

    #print(agent1.pi)
    #print(agent2.pi)
    plt.plot(np.arange(len(agent1.pi_history)),agent1.pi_history, label="agent1's pi(0)")
    plt.plot(np.arange(len(agent2.pi_history)),agent2.pi_history, label="agent2's pi(0)")

    plt.ylim(0, 1)
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.legend()
    #plt.savefig("result.jpg")
    plt.show()
