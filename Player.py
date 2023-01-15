import DQNAgent
import math
import numpy as np

from DQNAgent import DQNAgent
from copy import copy, deepcopy

class Player():

    def __init__(self):
        self.agent = DQNAgent(
            gamma=0.99,
            epsilon=1.0,
            learning_rate=0.0001,
            input_dims=[20, 10],
            n_actions=3,
            memory_size=10000,
            batch_size=10,
            epsilon_end=0.01
        )



    def hole_penalty(self, board, visited):
        holes = 0
        # iterate through all blocks of the board
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 0:
                    # check if the block has not been visited
                    if not visited[i][j]:
                        # mark it as visited
                        visited[i][j] = True
                        # increment hole count
                        holes += 1
                        # apply DFS to all unvisited neighbours
                        self.dfs(i, j, visited, board)
        return holes


    def dfs(self, i, j, visited, board):
        # check all neighbours
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if x >= 0 and x < len(board) and y >= 0 and y < len(board[x]) and board[x][y] == 0 and not visited[x][y]:
                visited[x][y] = True
                self.dfs(x, y, visited, board)


    def simulate(self, board):
        landed = False
        for _ in range(20):
            for i in range(19, 0, -1):
                for j in range(10):
                    if board[i][j] == 2:
                        if i == 19:
                            landed = True
                        elif i < 19 and board[i + 1][j] == 1:
                            landed = True
            for i in range(19, 0, -1):
                for j in range(10):
                    if board[i][j] == 2:
                        if i < 19 and board[i + 1][j] == 0 and not landed:
                            board[i][j] = 0
                            board[i + 1][j] = 2

        return board


    def reward(self, game, state, step):
        reward = 0
        done = False

        placement_score = 0
        fit = 1
        landing_columns = []

        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 2:
                    landing_columns.append(j)


        max_height = 1

        max_column = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(20):
            for j in range(10):
                if game.game.field[i][j] > 0:
                    max_column[j] = max(max_column[j], 20 - i)
                    max_height = max(max_height, 20 - i)

        visited = [[False for _ in range(10)] for _ in range(20)]
        holes_before = self.hole_penalty(state, visited)

        new_board = self.simulate(deepcopy(state))

        visited = [[False for _ in range(10)] for _ in range(20)]
        holes_after = self.hole_penalty(new_board, visited)

        fit = holes_after - holes_before + 1

        landing_height = max([max_column[i] for i in landing_columns])

        reward -= (landing_height * fit) / 10

        if game.done == True or max_height > 15:
            done = True
            reward -= 1000 / step

        reward += game.game.score * 10

        """
        max_height = 1

        max_column = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(20):
            for j in range(10):
                if game.game.field[i][j] > 0:
                    max_column[j] = max(max_column[j], 20 - i)
                    max_height = max(max_height, 20 - i)

        if game.done == True or max_height > 15:
            done = True
            reward -= 1000 / step



        unevenness = 0

        for i in range(9):
            unevenness += abs(max_column[i] - max_column[i + 1])

        reward -= unevenness / 50
        reward += step / 350
        reward += (1 / max_height)

        visited = [[False for _ in range(10)] for _ in range(20)]
        holes = self.hole_penalty(game.game.field, visited)

        reward -= holes / 10
        reward += game.game.score * 100

        #print(unevenness / 350, (1 / max_height), holes / 10, step / 200, reward)
        """

        return reward, done


    def apply_action(self, action, game, step):

        """
        ACTION = 1 -> LEFT
        ACTION = 2 -> RIGHT
        ACTION = 3 -> ROTATE
        """

        if action == 1:
            game.game.go_side(-1)
        elif action == 2:
            game.game.go_side(1)
        else:
            game.game.rotate()

        new_observation = deepcopy(game.game.field)

        for i in range(len(new_observation)):
            for j in range(len(new_observation[i])):
                new_observation[i][j] = min(new_observation[i][j], 1)

        image = deepcopy(game.game.figure.image())
        for i in range(4):
            for j in range(4):
                if i * 4 + j in image:
                    new_observation[i + game.game.figure.y][j + game.game.figure.x] = 2

        reward, done = self.reward(game, new_observation, step)
        return new_observation, reward, done
