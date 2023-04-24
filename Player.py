import DQNAgent
import math
import numpy as np

from DQNAgent import DQNAgent
from copy import copy, deepcopy


class Player():

    def __init__(self):
        self.agent = DQNAgent(
            gamma=0.99,
            epsilon=1,
            learning_rate=0.0001,
            input_dims=[13],
            n_actions=4,
            memory_size=1000000,
            batch_size=10,
            epsilon_end=0.01
        )



    def hole_count(self, board):
        holes = 0
        visited = [[False for _ in range(10)] for _ in range(20)]
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


    def reward(self, game, after, before):
        reward = 0
        done = False

        max_height = 1

        for i in range(20):
            for j in range(10):
                if game.game.field[i][j] > 0:
                    max_height = max(max_height, 20 - i)

        if game.done == True or max_height >= 7:
            done = True
            reward -= 1
        else:
            reward += 1

        if self.hole_count(before) < self.hole_count(after):
            reward -= 1

        reward -= max_height / 100
        reward += game.game.score

        return reward, done


    def apply_action(self, game, state, grid, current):

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 2:
                    game.game.color_map[i][j] = (game.game.figure.type + 1) % 7
                grid[i][j] = min(grid[i][j], 1)
        before = grid
        game.game.field = grid

        game.game.new_figure()

        reward, done = self.reward(game, grid, current)
        return reward, done
