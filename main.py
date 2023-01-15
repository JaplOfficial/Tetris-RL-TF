import pygame
import random
import Tetris
import Figure
import GameManager
import Player
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from Tetris import Tetris
from Player import Player as Player
from Figure import Figure
from GameManager import GameManager
from copy import copy, deepcopy


tf.compat.v1.disable_eager_execution()


player = Player()

episodes = 5000000

eps = []
total_rewards = []

total_reward = 0

for ep in range(episodes):

    game = GameManager()

    ep_reward = 0

    render = False

    if ep % 50 == 0:
        render = True
    else:
        render = False

    if ep % 10 == 0:
        eps.append(ep)
        total_rewards.append(total_reward / 10)
        plt.plot(eps, total_rewards)
        total_reward = 0
        plt.savefig('reward.png')
        plt.clf()

    done = False

    for step in range(500):
        render = True
        game.render_frame(render)

        observation = game.current_state()

        action = player.agent.choose_action(observation)
        next_observation, reward, done = player.apply_action(action, game, step)

        total_reward += reward
        ep_reward += reward
        player.agent.store_transition(observation, action, reward, next_observation, done)
        observation = next_observation
        player.agent.learn()

        if done:
            break


    print('Episode : ', ep, 'Epsilon : ', player.agent.epsilon, 'Reward : ', ep_reward)



pygame.quit()
