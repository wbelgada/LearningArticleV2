import numpy as np
import matplotlib.pyplot as plt

from agents.phc_agent import PHCAgent
from agents.wolf_agent import WoLFAgent
from agents.Q_learner_agent import QLearnerAgent
from environments.matrix_game import MatrixGame

if __name__ == '__main__':
    nb_episode = 1000000
    evaluate_every = 10000
    nb_runs = 1

    actions = np.arange(3)

    total_pi_historyRock = np.zeros(int(nb_episode/evaluate_every))
    total_pi_historyRock2 = np.zeros(int(nb_episode/evaluate_every))
    total_pi_historyPaper = np.zeros(int(nb_episode/evaluate_every))
    total_pi_historyPaper2 = np.zeros(int(nb_episode/evaluate_every))



    game = MatrixGame("RPS")
    for i in range(nb_runs):
        printu=False
        agent1 = WoLFAgent(alpha=0.1, actions=actions, nb_states=1, high_delta=0.0004, low_delta=0.0002)
        #agent2 = WoLFAgent(alpha=0.1, actions=actions, nb_states=1, high_delta=0.0004, low_delta=0.0002)

        agent2 = PHCAgent(delta=0.0004, nb_actions=3, initialStrategy=(1 / len(actions) for i in range(len(actions))))
        for episode in range(nb_episode):
            """if episode >= 670000 and not printu:
                printu = True"""
            if episode%10000 == 0:
                print(episode)
            action1 = agent1.act(0)
            agent2.act(0)
            action2 = agent2.currentAction

            _, r1, r2 = game.step(action1, action2)

            agent1.observe(reward=r1, obs=0, nextobs=0)
            agent2.setReward(r2)
            agent2.updateActionValues(0,0)
            agent2.updateStrategy(0)
            agent2.updateTimeStep()
            agent2.updateEpsilon()
            agent2.updateAlpha()
            if episode % evaluate_every == 0:
                index = int(episode/evaluate_every)
                total_pi_historyRock[index] += agent1.pi[0][0]
                total_pi_historyRock2[index] += agent2.strategy[0][0]
                total_pi_historyPaper[index] += agent1.pi[0][1]
                total_pi_historyPaper2[index] += agent2.strategy[0][1]

    total_pi_historyRock /= nb_runs
    total_pi_historyRock2 /= nb_runs
    total_pi_historyPaper /= nb_runs
    total_pi_historyPaper2 /= nb_runs

    print(total_pi_historyPaper[len(total_pi_historyPaper2)-20:len(total_pi_historyPaper)])
    print(total_pi_historyRock[len(total_pi_historyPaper2)-20:len(total_pi_historyPaper)])

    #print(agent1.pi)
    #print(agent2.pi)
    plt.plot(total_pi_historyRock, total_pi_historyPaper, label="agent1's pi(0)")
    #plt.plot(total_pi_historyRock2, total_pi_historyPaper2, label="agent1's pi(0)")

    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.legend()
    #plt.savefig("result.jpg")
    plt.show()

