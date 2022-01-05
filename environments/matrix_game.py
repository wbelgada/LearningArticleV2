import random

import numpy as np

class MatrixGame():
    def __init__(self, game):
        self.game= game
        self.reward_matrix = self._create_reward_table()

    def step(self, action1, action2):

        r1, r2 = self.reward_matrix[action1][action2]

        return None, r1, r2

    def _create_reward_table(self):
        if self.game == "MP":

            reward_matrix = [
                                [[1, -1], [-1, 1]],
                                [[-1, 1], [1, -1]]
                            ]
        else:
            reward_matrix = [
                                [[0,0],[-1,1],[1,-1]],
                                [[1,-1],[0,0],[-1,1]],
                                [[-1,1],[1,-1],[0,0]]
                            ]

        return reward_matrix
