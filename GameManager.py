import pygame
import pygame
import random
import Tetris
import Figure

from Tetris import Tetris
from Figure import Figure
from copy import copy, deepcopy


colors = [
    (128, 229, 255), (164, 93, 77), (128, 229, 255),
]



class GameManager:

    def __init__(self):
        # Initialize the game engine
        pygame.init()
        # Define some colors
        self.TEXT = (217, 192, 163)
        self.BACKGROUND = (52, 55, 63)
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

    def render_frame(self, render):
        pygame.event.get()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        if self.game.figure is None and self.done == False:
            self.game.new_figure()


        self.game.go_down()

        if render == True:

            self.screen.fill(self.BACKGROUND)

            for i in range(self.game.height):
                for j in range(self.game.width):
                    pygame.draw.rect(self.screen, self.GRAY, [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom, self.game.zoom], 1)
                    if self.game.field[i][j] > 0:
                        pygame.draw.rect(self.screen, colors[self.game.field[i][j]],
                                         [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])

            if self.game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in self.game.figure.image():
                            pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                             [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                              self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                              self.game.zoom - 2, self.game.zoom - 2])


            pygame.display.flip()

            #self.clock.tick(self.fps)


    def current_state(self):
        observation = deepcopy(self.game.field)
        for i in range(len(observation)):
            for j in range(len(observation[i])):
                observation[i][j] = min(observation[i][j], 1)

        image = deepcopy(self.game.figure.image())
        for i in range(4):
            for j in range(4):
                if i * 4 + j in image:
                    observation[i + self.game.figure.y][j + self.game.figure.x] = 2
        return observation
