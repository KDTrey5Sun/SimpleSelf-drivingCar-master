import numpy as np
import os

class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, combined=False):
        # input_dims: 状态维度
        self.mem_size = int(max_mem)
        self.batch_size = int(batch_size)
        self.mem_cntr = 0
        self.combined = combined

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def save_buffer(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'state_memory.npy'), self.state_memory)
        np.save(os.path.join(path, 'new_state_memory.npy'), self.new_state_memory)
        np.save(os.path.join(path, 'action_memory.npy'), self.action_memory)
        np.save(os.path.join(path, 'reward_memory.npy'), self.reward_memory)
        np.save(os.path.join(path, 'terminal_memory.npy'), self.terminal_memory)
        np.save(os.path.join(path, 'meta.npy'), np.array([self.mem_size, self.batch_size, self.mem_cntr], dtype=np.int64))

    def load_buffer(self, path):
        self.state_memory = np.load(os.path.join(path, 'state_memory.npy'))
        self.new_state_memory = np.load(os.path.join(path, 'new_state_memory.npy'))
        self.action_memory = np.load(os.path.join(path, 'action_memory.npy'))
        self.reward_memory = np.load(os.path.join(path, 'reward_memory.npy'))
        self.terminal_memory = np.load(os.path.join(path, 'terminal_memory.npy'))
        if os.path.exists(os.path.join(path, 'meta.npy')):
            meta = np.load(os.path.join(path, 'meta.npy'))
            self.mem_size = int(meta[0])
            self.batch_size = int(meta[1])
            self.mem_cntr = int(meta[2])
        else:
            self.mem_size = len(self.action_memory)
            self.mem_cntr = min(self.mem_cntr, self.mem_size)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def sample_memory(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminals

    def is_sufficient(self):
        return self.mem_cntr >= self.batch_size