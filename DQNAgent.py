import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras import datasets, layers, models
from keras.regularizers import l2



from ReplayBuffer import ReplayBuffer


def build_dqn():

    s = 1

    weight_decay = 0.005

    #Arquitetura da rede CNN

    model = models.Sequential([
        layers.Input((20, 10, 1)),
        keras.layers.Conv2D(filters=256, kernel_size=(7, 7),padding="same", strides=(1,1), activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, kernel_regularizer=l2(weight_decay), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, kernel_regularizer=l2(weight_decay), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, kernel_regularizer=l2(weight_decay), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3)
    ])

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=5000,
        decay_rate=0.96,
        staircase=True)

    model.compile(
                    loss='mean_squared_error',
                    optimizer=keras.optimizers.Adam(lr_schedule))

    return model


class DQNAgent:

    def __init__(self, learning_rate, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, memory_size=100000):
        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        print(input_dims)
        self.memory = ReplayBuffer(memory_size, input_dims)
        self.q_eval = build_dqn()
        self.played = False

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action)
        else:
            state = np.array([observation])
            state = np.expand_dims(state, axis=-1)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        self.played = True

        if self.memory.memory_counter < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        states = np.array(states)
        new_states = np.array(new_states)
        states = np.expand_dims(states, axis=-1)
        new_states = np.expand_dims(new_states, axis=-1)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(new_states)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon *= 0.99999
