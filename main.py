import pygame
import random
import Tetris
import Figure
import GameManager
import Player
import pickle
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from Tetris import Tetris
from Player import Player as Player
from Figure import Figure
from GameManager import GameManager
from copy import copy, deepcopy


tf.compat.v1.disable_eager_execution()

pygame.init()

player = Player()

episodes = 10000

eps = []
total_rewards = []

total_reward = 0
maximum = 0
highscore = 200

mode = 'train-unsupervised'


# Train the DQN from scratch
if mode == 'train-unsupervised':

    for ep in range(episodes):

        game = GameManager()

        game.game.new_figure()

        ep_reward = 0

        render = False

        if ep % 50 == 0:
            render = True
        else:
            render = False

        # Every 100 episodes save the better performing model and update reward graph
        if ep % 100 == 0 and ep != 0:
            if total_reward / 100 > maximum:
                maximum = total_reward / 100
                player.agent.save()
            eps.append(ep)
            total_rewards.append(total_reward / 100)
            plt.plot(eps, total_rewards)
            total_reward = 0
            plt.savefig('reward.png')
            plt.clf()

        done = False

        for step in range(500):
            render = True
            game.render_frame(render, mode)

            current_state, all_states_vectorized, all_states, current_grid = game.current_state()
            best_state, best_grid = player.agent.choose_action(all_states_vectorized, all_states)
            reward, done = player.apply_action(game, best_state, best_grid, current_grid)
            total_reward += reward
            ep_reward += reward
            player.agent.store_transition(current_state, best_state, reward, done)
            player.agent.learn()

            if done:
                break


        print('Episode:', ep, 'Epsilon:', player.agent.epsilon, 'Reward:', ep_reward, 'Memory:', len(player.agent.memory))

# Fine-tune previously trained model with smaller learning rate
elif mode == 'fine-tune':

        player.agent.fine_tune(0.0001)

        for ep in range(episodes):

            if ep % 50 == 0 and ep != 0:
                eps.append(ep)
                total_rewards.append(total_reward / 50)
                plt.plot(eps, total_rewards)
                total_reward = 0
                plt.savefig('reward.png')
                plt.clf()

            ep_reward = 0

            game = GameManager()

            game.game.new_figure()

            done = False

            for step in range(500):
                render = True
                game.render_frame(render, mode)

                current_state, all_states_vectorized, all_states = game.current_state()
                best_state, best_grid = player.agent.choose_action(all_states_vectorized, all_states)
                reward, done = player.apply_action(game, best_state, best_grid)
                total_reward += reward
                ep_reward += reward
                player.agent.store_transition(current_state, best_state, reward, done)
                player.agent.learn()

                if done:
                    break
            if ep_reward > highscore:
                highscore = ep_reward
                player.agent.save()
            print('Episode:', ep, 'Epsilon:', player.agent.epsilon, 'Reward:', ep_reward, 'Memory:', len(player.agent.memory), 'Highscore:', highscore)


# Display the trained model without training (showcase only)
elif mode == 'play':

    player.agent.load_model()
    player.agent.epsilon = 0

    for ep in range(episodes):

        game = GameManager()

        game.game.new_figure()

        done = False

        for step in range(500):
            render = True
            game.render_frame(render, mode)

            current_state, all_states_vectorized, all_states = game.current_state()
            best_state, best_grid = player.agent.choose_action(all_states_vectorized, all_states)
            reward, done = player.apply_action(game, best_state, best_grid)
            #player.agent.store_transition(current_state, best_state, reward, done)
            #player.agent.learn()

            if done:
                break



pygame.quit()
