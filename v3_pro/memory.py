import numpy as np

class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, combined=False):
        # input_dims：状态向量的维度。例如状态是 [x, y, vx, vy]，则为4 每个状态的参数的个数
        # max_mem：记忆上限（最多存储多少条经验）
        # batch_size：每次训练从记忆中采样多少条

        # combined（默认False）：
            # 若为True，会在采样时强制将最近的一条经历加入训练批次，用于某些特殊算法中提高训练效果
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0 # 记录当前存储的经验数量
        self.combined = combined # 是否使用combined

        # 初始化存储经验的数组
        # state_memory：存储状态
        # new_state_memory：存储下一个状态
 
        self.state_memory = np.zeros((self.mem_size, input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                         dtype=np.float32)
        
        # action_memory：存储动作
        # reward_memory：存储奖励
        # terminal_memory：存储是否终止
        # 一维数组，长度为mem_size
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def save_buffer(self, path): 
        np.save(path+'/state_memory.npy', self.state_memory)
        np.save(path+'/new_state_memory.npy', self.new_state_memory)
        np.save(path+'/action_memory.npy', self.action_memory)
        np.save(path+'/reward_memory.npy', self.reward_memory)
        np.save(path+'/terminal_memory.npy', self.terminal_memory)

    def load_buffer(self, path):
        self.state_memory = np.load(path + 'state_memory.npy')
        self.new_state_memory = np.load(path + 'new_state_memory.npy')
        self.action_memory = np.load(path + 'action_memory.npy')
        self.reward_memory = np.load(path + 'reward_memory.npy')
        self.terminal_memory = np.load(path + 'terminal_memory.npy')


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1


    def sample_memory(self):
        # offset = 1 if self.combined else 0
        # max_mem = min(self.mem_cntr, self.mem_size) - offset
        # batch = np.random.choice(max_mem, self.batch_size - offset,
        #                          replace=False)
        

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # replace=False：表示无放回抽样（每个元素只能被选中一次）。 如果为 True，则允许重复选中同一元素。
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        # if self.combined:
        #     index = self.mem_cntr % self.mem_size - 1
        #     last_action = self.action_memory[index]
        #     last_state = self.state_memory[index]
        #     last_new_state = self.new_state_memory[index]
        #     last_reward = self.reward_memory[index]
        #     last_terminal = self.terminal_memory[index]

        #     actions = np.append(self.action_memory[batch], last_action)
        #     states = np.vstack((self.state_memory[batch], last_state))
        #     new_states = np.vstack((self.new_state_memory[batch],
        #                            last_new_state))
        #     rewards = np.append(self.reward_memory[batch], last_reward)
        #     terminals = np.append(self.terminal_memory[batch], last_terminal)

        return states, actions, rewards, new_states, terminals


    # 判断是否有足够的经验进行训练
    def is_sufficient(self):
        return self.mem_cntr >= self.batch_size