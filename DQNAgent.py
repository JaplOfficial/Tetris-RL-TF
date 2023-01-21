import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random

from tensorflow import keras
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from collections import deque


from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras import datasets, layers, models
from keras.regularizers import l2



from ReplayBuffer import ReplayBuffer


def build_dqn(lr):

    model = models.Sequential([
        keras.layers.Input((4)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

    return model


class DQNAgent:

    """
        gamma -> discount factor in the bellman equation
        epsilon -> Exploration rate during training
        epsilon_dec -> decreasing factor for epsilon
        epsilon_min -> minimum epsilon
        batch_size -> number of sample used during training
        memory -> memory buffer to store previous experiences
        q_eval -> DQN model that infers best states
    """

    def __init__(self, learning_rate, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, memory_size=10000):
        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.q_eval = build_dqn(0.01)
        self.played = False

    def store_transition(self, state, new_state, reward, done):
        self.memory.append((state, new_state, reward, done))

    def choose_action(self, observations, all_states):
        if np.random.random() < self.epsilon:
            best_state = random.randint(0, len(observations) - 1)
        else:
            best_state = 0
            maximum_score = -1
            for i in range(len(observations)):
                state = np.array([observations[i]])
                score = self.q_eval.predict(state)
                if score > maximum_score:
                    maximum_score = score
                    best_state = i
        return observations[best_state], all_states[best_state]

    def learn(self):
        n = len(self.memory)
        batch_size = 512
        epochs = 3

        if n >= batch_size:

            # Sample a random batch of data from the agents memory
            batch = random.sample(self.memory, batch_size)

            # Predict Q-values of the next states
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.q_eval.predict(next_states)]

            x = []
            y = []

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.gamma * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.q_eval.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

        # Update exploration rate
        self.epsilon *= 0.9999



    def save(self):
        return self.q_eval.save('agent.h5')

    def load_model(self):
        self.q_eval = keras.models.load_model('agent.h5')

    # Copy the weight of the previously trained model and update learning
    def fine_tune(self, lr):
        self.load_model()
        fine_tune_model = build_dqn(lr)
        fine_tune_model.set_weights(self.q_eval.get_weights())
        self.epsilon = 0
        self.q_eval = fine_tune_model
