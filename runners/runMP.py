import numpy as np
import matplotlib.pyplot as plt
from time import time

from agents.phc_agent import PHCAgent
from agents.wolf_agent import WoLFAgent
from agents.Q_learner_agent import QLearnerAgent
from environments.matrix_game import MatrixGame

if __name__ == '__main__':

    nb_episode = 100000
    evaluate_every = 1000
    nb_runs = 1
    total_pi_historyHeads = np.zeros(int(nb_episode / evaluate_every))
    total_pi_historyHeads2 = np.zeros(int(nb_episode / evaluate_every))
    game = MatrixGame("MP")
    t1 = time()
    for i in range(nb_runs):
        actions = np.arange(2)
        agent1 = WoLFAgent(alpha=0.1, actions=actions,nb_states=1, high_delta=0.0004, low_delta=0.0002)
        #agent2 = WoLFAgent(alpha=0.1, actions=actions,nb_states=1, high_delta=0.0004, low_delta=0.0002)
        #agent1 = PHCAgent(delta = 0.0004,initialStrategy=(1/len(actions) for i in range(len(actions))))
        #agent2 = PHCAgent(delta = 0.0004,initialStrategy=(1/len(actions) for i in range(len(actions))))

        #agent1 = PHCAgent(delta = 0.0004,initialStrategy=(1.0, 0.0))
        agent2 = PHCAgent(delta = 0.0004,initialStrategy=(1.0, 0.0))

        for episode in range(nb_episode):
            action1 = agent1.act(0)
            #action2 = agent2.act(0)

            """agent1.act()
            action1 = agent1.currentAction
            """
            agent2.act(0)
            action2 = agent2.currentAction

            _, r1, r2 = game.step(action1, action2)

            agent1.observe(reward=r1, obs=0, nextobs=0)
            #agent2.observe(reward=r2, obs=0, nextObs=0)
            agent2.setReward(r2)
            agent2.updateActionValues(0,0)
            agent2.updateStrategy(0)
            agent2.updateTimeStep()
            agent2.updateEpsilon()
            agent2.updateAlpha()

            if episode % evaluate_every == 0:
                index = int(episode / evaluate_every)
                total_pi_historyHeads[index] += agent1.pi[0][0]
                total_pi_historyHeads2[index] += agent2.strategy[0][0]

            if episode%50000 == 0:
                print(episode)
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
    t2 = time()
    print("time : " + str(t2-t1))
    total_pi_historyHeads/=nb_runs
    total_pi_historyHeads2/=nb_runs
    plt.plot(np.arange(0,len(total_pi_historyHeads)*evaluate_every, evaluate_every),total_pi_historyHeads, label="Wolf-PHC: Pr(Heads)",  linewidth=0.5)
    plt.plot(np.arange(0,len(total_pi_historyHeads2)*evaluate_every, evaluate_every),total_pi_historyHeads2, label="PHC: Pr(Heads)", linewidth=0.5)

    plt.ylim(0, 1)
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.legend()
    plt.savefig("resultMP.jpg")
    plt.show()
