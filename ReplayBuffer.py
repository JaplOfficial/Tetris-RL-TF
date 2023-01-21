import numpy as np


class ReplayBuffer:

    def __init__(self, memory_size, input_dims):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.int32)

    def store_transition(self, state, new_state, reward, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = np.array(state)
        self.new_state_memory[index] = np.array(new_state)
        self.reward_memory[index] = reward
        #self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
