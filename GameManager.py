import pygame
import pygame
import random
import Tetris
import Figure
import math

from Tetris import Tetris
from Figure import Figure
from copy import copy, deepcopy


colors = [
    ((0, 204, 204), (204, 0, 0), (0, 204, 0), (0, 0, 204), (204, 102, 0), (153, 0, 204), (204, 204, 0)),
]




class GameManager:

    def __init__(self):
        # Initialize the game engine
        pygame.init()
        # Define some colors
        self.TEXT = (217, 192, 163)
        self.BACKGROUND = (0, 0, 0)
        self.GRAY = (128, 128, 128)

        self.size = (400, 500)
        self.screen = pygame.display.set_mode(self.size)

        pygame.display.set_caption("Tetris AI")


        # Loop until the user clicks the close button.
        self.done = False
        self.clock = pygame.time.Clock()
        self.fps = 1
        self.game = Tetris(20, 10)
        self.counter = 0

        self.pressing_down = False

    def render_frame(self, render, mode):
        pygame.event.get()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True


        #self.game.go_down()
        self.game.break_lines()

        if render == True:

            self.screen.fill(self.BACKGROUND)
            pygame.Rect(30, 30, 60, 60)
            # [self.game.x * 2, self.game.y * 2, 20, 20]
            c = 3.8
            pygame.draw.rect(self.screen, self.GRAY, pygame.Rect(25 * c, 23 * c, 55 * c, 100 * c), 5)
            for i in range(self.game.height):
                for j in range(self.game.width):
                    #pygame.draw.rect(self.screen, self.GRAY, [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom, self.game.zoom], 1)
                    if self.game.field[i][j] > 0:
                        pygame.draw.rect(self.screen, colors[0][self.game.color_map[i][j] - 1],
                                         [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])
            """
            if self.game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in self.game.figure.image():
                            pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                             [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                              self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                              self.game.zoom - 2, self.game.zoom - 2])
            """

            pygame.display.flip()
            #self.clock.tick(self.fps)

            # If mode is play slow down rendering
            if mode == 'play':
                self.clock.tick(self.fps)

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

    def intersects(self, figure, field):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in figure.image():
                    if i + figure.y > 20 - 1 or \
                            j + figure.x > 10 - 1 or \
                            j + figure.x < 0 or \
                            field[i + figure.y][j + figure.x] > 0:
                        intersection = True
        return intersection



    def hole_count(self, board, visited):
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


    # Count the number of connected components in the grid
    def dfs(self, i, j, visited, board):
        # check all neighbours
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if x >= 0 and x < len(board) and y >= 0 and y < len(board[x]) and board[x][y] == 0 and not visited[x][y]:
                visited[x][y] = True
                self.dfs(x, y, visited, board)

    #
    def break_lines(self, board):
        lines = 0
        for i in range(1, 20):
            zeros = 0
            for j in range(10):
                if board[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(10):
                        board[i1][j] = board[i1 - 1][j]
        return lines, board

    def feature_map(self, state, type):
        features = []
        max_column = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        complete_lines, state = self.break_lines(state)

        for i in range(20):
            for j in range(10):
                if state[i][j] > 0:
                    max_column[j] = max(max_column[j], 20 - i)

        visited = [[False for _ in range(10)] for _ in range(20)]

        unevenness = 0
        total_height = max_column[0]
        holes = self.hole_count(state, visited)

        for i in range(1, len(max_column)):
            total_height += max_column[i]
            unevenness += abs(max_column[i] - max_column[i - 1])

        features.append(complete_lines)
        features.append(unevenness)
        features.append(total_height)
        features.append(holes)



        return features

    def current_state(self):
        grids = []
        observation = deepcopy(self.game.field)
        possible_states = []
        for i in range(len(observation)):
            for j in range(len(observation[i])):
                observation[i][j] = min(observation[i][j], 1)

        current_state = self.feature_map(observation, self.game.figure.type)
        piece = deepcopy(self.game.figure)
        for rotation in range(piece.rotation_states()):
            for translation in range(-10, 10, 1):
                state = deepcopy(observation)
                piece.x = translation
                image = deepcopy(piece.image())
                if not self.intersects(piece, state):
                    for i in range(4):
                        for j in range(4):
                            if i * 4 + j in image:
                                state[i + piece.y][j + piece.x] = 2
                    landing = self.simulate(state)
                    grids.append(landing)
                    vectorized_features = self.feature_map(landing, self.game.figure.type)
                    possible_states.append(vectorized_features)
            piece.rotate()
        return current_state, possible_states, grids, observation
