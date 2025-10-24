"""
完善且工程化的 DQN 实现（兼容用户提供的 ReplayMemory）
该文件与用户给出的 memory.py 配合使用。主要特性：
- 兼容用户 memory.py 的构造函数（input_dims 为 int）、store/sampling 格式
- Double DQN（可选）、Dueling（可选）、Huber 损失、梯度裁剪
- learn_starts、软/硬 target 更新、模型保存/加载
- 完整中文注释，便于教学与二次开发

使用方式：
1. 将本文件与用户的 memory.py 放在同一目录
2. 运行：python improved_dqn_with_user_memory.py

"""

import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple

# 导入用户提供的 ReplayMemory（假定文件名为 memory.py）
from memory import ReplayMemory


# ----------------------------- 网络定义 -----------------------------
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim: int, fc1_dims: int, fc2_dims: int, n_actions: int, dueling: bool = False):
        super(DeepQNetwork, self).__init__()
        self.dueling = dueling

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        if self.dueling:
            self.value_stream = nn.Linear(fc2_dims, 1)
            self.adv_stream = nn.Linear(fc2_dims, n_actions)
        else:
            self.fc3 = nn.Linear(fc2_dims, n_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.dueling:
            val = self.value_stream(x)
            adv = self.adv_stream(x)
            q = val + (adv - adv.mean(dim=1, keepdim=True))
            return q
        else:
            return self.fc3(x)


# ----------------------------- Agent -----------------------------
class DQNAgent:
    def __init__(self,
                 input_dim: int,
                 n_actions: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 mem_size: int = 100_000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_dec: float = 1e-4,
                 learn_starts: int = 1000,
                 replace_target: int = 1000,
                 double_dqn: bool = True,
                 dueling: bool = False,
                 device: Optional[str] = None,
                 use_soft_update: bool = False,
                 tau: float = 0.005,
                 max_grad_norm: float = 1.0,
                 combined_memory: bool = False):
        """
        说明：本类专门兼容用户提供的 ReplayMemory
        - input_dim: 单个状态的维度（int），对应用户 memory.py 中的 input_dims
        - combined_memory: 若用户在创建 ReplayMemory 时传入 combined=True，请把该参数设为 True
        """
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size
        self.learn_starts = learn_starts
        self.replace_target = replace_target
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.use_soft_update = use_soft_update
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        # device
        if device is None:
            self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        else:
            self.device = T.device(device)

        # 与用户 memory.py 接口兼容
        # 用户的 ReplayMemory 构造函数为: ReplayMemory(input_dims, max_mem, batch_size, combined=False)
        self.memory = ReplayMemory(input_dim, mem_size, batch_size, combined_memory)

        # 网络
        self.Q_eval = DeepQNetwork(input_dim, 256, 200, n_actions, dueling=dueling).to(self.device)
        self.Q_next = DeepQNetwork(input_dim, 256, 200, n_actions, dueling=dueling).to(self.device)
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # 优化器 & 损失
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.iter_cntr = 0
        self.n_actions = n_actions

    def choose_action(self, observation: np.ndarray, greedy: bool = False) -> int:
        if (np.random.random() > self.epsilon) or greedy:
            state = T.tensor([observation], dtype=T.float32, device=self.device)
            q_vals = self.Q_eval.forward(state)
            action = int(T.argmax(q_vals).item())
        else:
            action = int(np.random.choice(self.n_actions))
        return action

    def store_transition(self, state, action, reward, state_, done):
        # 直接调用用户 memory 的接口
        self.memory.store_transition(state, action, reward, state_, done)

    def soft_update(self):
        for target_param, param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def learn(self) -> Optional[float]:
        # 与用户 memory.is_sufficient() 接口兼容
        if not self.memory.is_sufficient():
            return None
        if self.memory.mem_cntr < self.learn_starts:
            return None

        self.optimizer.zero_grad()

        # 用户的 sample_memory 返回：states, actions, rewards, new_states, terminals
        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        # 转为 tensor
        states = T.tensor(states, dtype=T.float32, device=self.device)
        new_states = T.tensor(new_states, dtype=T.float32, device=self.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.device)
        # 用户 terminal memory 是 bool 数组
        dones = T.tensor(dones, dtype=T.bool, device=self.device)
        actions = T.tensor(actions, dtype=T.long, device=self.device)

        batch_index = T.arange(actions.shape[0], dtype=T.long, device=self.device)

        # q_eval
        q_eval_all = self.Q_eval.forward(states)  # [B, n_actions]
        q_eval = q_eval_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with T.no_grad():
            if self.double_dqn:
                q_next_eval = self.Q_eval.forward(new_states)
                next_actions = T.argmax(q_next_eval, dim=1)
                q_next_target_all = self.Q_next.forward(new_states)
                q_next_values = q_next_target_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                q_next_target_all = self.Q_next.forward(new_states)
                q_next_values, _ = T.max(q_next_target_all, dim=1)

            q_next_values[dones] = 0.0
            q_target = rewards + self.gamma * q_next_values

        loss = self.loss_fn(q_eval, q_target)
        loss.backward()

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()

        self.iter_cntr += 1
        # epsilon 衰减在满足 learn_starts 后开始
        if self.memory.mem_cntr >= self.learn_starts and self.epsilon > self.eps_min:
            self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        # target 更新
        if self.use_soft_update:
            self.soft_update()
        else:
            if self.iter_cntr % self.replace_target == 0:
                self.hard_update()

        return float(loss.item())

    def save_models(self, folder: str = './models', prefix: str = 'dqn'):
        os.makedirs(folder, exist_ok=True)
        T.save(self.Q_eval.state_dict(), os.path.join(folder, f'{prefix}_eval.pth'))
        T.save(self.Q_next.state_dict(), os.path.join(folder, f'{prefix}_next.pth'))
        T.save(self.optimizer.state_dict(), os.path.join(folder, f'{prefix}_optim.pth'))

    def load_models(self, folder: str = './models', prefix: str = 'dqn'):
        self.Q_eval.load_state_dict(T.load(os.path.join(folder, f'{prefix}_eval.pth'), map_location=self.device))
        self.Q_next.load_state_dict(T.load(os.path.join(folder, f'{prefix}_next.pth'), map_location=self.device))
        self.optimizer.load_state_dict(T.load(os.path.join(folder, f'{prefix}_optim.pth'), map_location=self.device))


# ----------------------------- 示例：如何在你的环境中调用 -----------------------------
# 下面的伪代码展示如何把 agent 接入你现有的训练循环（伪代码）：
#
# agent = DQNAgent(input_dim=state_dim, n_actions=env.action_space.n, mem_size=100000, ...)
# for episode in range(N):
#     state = env.reset()
#     done = False
#     while not done:
#         action = agent.choose_action(state)
#         next_state, reward, done, info = env.step(action)
#         agent.store_transition(state, action, reward, next_state, done)
#         loss = agent.learn()
#         state = next_state
#
# 注意：这里直接使用了用户的 ReplayMemory，因此不需要额外适配。

print('已生成与用户 memory.py 完全兼容的 DQNAgent 类，文件名：improved_dqn_with_user_memory.py')
