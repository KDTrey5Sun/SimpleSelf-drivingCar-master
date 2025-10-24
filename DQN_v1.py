import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from memory import ReplayMemory



# network 神经网络结构
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        # input_dims: 状态向量的维度
        # fc1_dims: 第一层的神经元个数
        # fc2_dims: 第二层的神经元个数
        # n_actions: 动作空间的大小
        # lr: 学习率
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # 输入 → 全连接层1（ReLU）→ 全连接层2（ReLU）→ 输出层（每个动作的 Q 值）

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # 前向传播函数
        # state: 输入的状态向量
        # x: 中间变量
        # actions: 输出的 Q 值
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions




class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 combined=False, max_mem_size=100000, eps_end=0.05,
                 eps_dec=5e-4):
        # gamma: 折扣因子
        # epsilon: 探索率
        # lr: 学习率
        # input_dims: 状态向量的维度
        # n_actions: 动作空间的大小
        # combined: 是否在采样中附加最新 transition
        # max_mem_size: Replay Buffer 最大容量
        # eps_end: 最小 epsilon
        # eps_dec: 衰减步长
        # eps_min, eps_dec: 最小 epsilon 和衰减步长
        # batch_size: 批次大小
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size


        self.action_space = [i for i in range(n_actions)]
        self.memory = ReplayMemory(input_dims, max_mem_size,
                                   batch_size, combined)
        self.iter_cntr = 0
        # 记录学习次数（训练的 step 数）。
        
        # self.replace_target = 50
        # 每隔 50 次学习就更新一次目标网络的参数

        # self.replace_target = 20
        # 每隔 20 次学习就更新一次目标网络的参数




        self.replace_target = 100
        # 每隔 100 次学习就更新一次目标网络的参数




        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=200)
        # Q_eval: 评估网络
        # Q_next: 目标网络
        # Q_eval 和 Q_next 的结构是一样的，只是参数不同
        # Q_eval 用于评估当前状态下每个动作的 Q 值
        # Q_next 用于计算下一个状态下每个动作的 Q 值
        # Q_eval 的参数是通过训练得到的
        # Q_next 的参数是通过 Q_eval 的参数更新得到的

        self.Q_next = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=200)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            # observation: 当前状态
            # state: 转换为 tensor
            state = state.to(torch.float32)
            # 转换为 float32 类型

            actions = self.Q_eval.forward(state)
            # 计算当前状态下每个动作的 Q 值
            
            action = T.argmax(actions).item()
            # 选择 Q 值最大的动作
            
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if not self.memory.is_sufficient():
            return

        self.Q_eval.optimizer.zero_grad()
        # 清空梯度 每次反向传播前先清除旧的梯度，避免梯度累积。


        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # batch_index: 0, 1, 2, ..., batch_size-1
        # np.arange: 生成一个从 0 到 batch_size-1 的数组
        #创建一个 [0, 1, ..., batch_size-1] 的数组，用来索引 batch 中每一条数据，方便之后从张量中按行选择元素

        states, actions, rewards, new_states, dones = self.memory.sample_memory()
        # print(f'states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, new_states: {new_states.shape}, dones: {dones.shape}')
        # print(f'states: {states}, actions: {actions}, rewards: {rewards}, new_states: {new_states}, dones: {dones}')



        #从经验回放中采样一个小批量（batch）数据。
        # states：当前状态集合
        # actions：每个状态下采取的动作
        # rewards：每个动作获得的奖励
        # new_states：每个动作后的新状态
        # dones：是否结束（终止）标志
            
        states = T.tensor(states).to(self.Q_eval.device)
        new_states = T.tensor(new_states).to(self.Q_eval.device)
        rewards = T.tensor(rewards).to(self.Q_eval.device)
        dones = T.tensor(dones).to(self.Q_eval.device)
        # 将数据转换为 tensor，并移动到 GPU 上（如果可用）

        


        q_eval = self.Q_eval.forward(states)[batch_index, actions]
        # 计算当前 Q 值（估计值）
        # 将 states 输入 Q 网络，输出每个动作的 Q 值。
        # 用 actions 按行提取对应动作的 Q 值。
        # 得到 q_eval：当前动作在当前状态下的 Q 值预测
        q_next = self.Q_next.forward(new_states)
        # 计算下一个状态下每个动作的 Q 值

        q_next[dones] = 0.0
        # 如果当前状态是终止状态，则下一个状态的 Q 值为 0

        q_target = rewards + self.gamma*T.max(q_next, dim=1)[0]
        # 计算目标 Q 值
        # Q_target = r + γ * max_a(Q(s', a))f

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)  # max_norm可根据实际调整
        self.Q_eval.optimizer.step()




        # 反向传播
        # loss.backward()：计算梯度
        # self.Q_eval.optimizer.step()：更新参数
        # loss：当前动作在当前状态下的 Q 值预测和目标 Q 值之间的均方误差
        # q_eval：当前动作在当前状态下的 Q 值预测
        # q_target：目标 Q 值
        # q_next：下一个状态下每个动作的 Q 值
        # dones：是否结束（终止）标志


        # print('learn')
        # print(f'learn - loss: {loss.item():.3f}, epsilon: {self.epsilon:.2f}')

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        return loss.item() if loss is not None else None    