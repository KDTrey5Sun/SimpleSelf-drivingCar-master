# /Desktop/SimpleSelf-drivingCar-master/v3_pro_windowless/DQN.py
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from memory import ReplayMemory
from torch.nn.utils.clip_grad import clip_grad_norm_

# mps 可用性：通过 try-import 避免静态检查对 torch.backends 的报错
try:
    import torch.backends.mps as _mps
    _HAS_MPS = bool(getattr(_mps, 'is_available', lambda: False)())
except Exception:
    _HAS_MPS = False

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        # 设备选择：优先 CUDA，其次 Apple Silicon MPS，最后 CPU。
        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        elif _HAS_MPS:
            self.device = T.device('mps')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
    


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

        # 设备选择：优先 CUDA，其次 Apple Silicon MPS，最后 CPU。
        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        elif _HAS_MPS:
            self.device = T.device('mps')
        else:
            self.device = T.device('cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 combined=False, max_mem_size=100000, eps_end=0.05,
                 eps_dec=1e-4, learn_starts=1000, replace_target=3000,
                 double_dqn=True, dueling=True, clip_reward=False, target_soft_tau=0.0):
        
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size
        self.learn_starts = max(learn_starts, batch_size)
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.clip_reward = clip_reward
        self.target_soft_tau = target_soft_tau

        self.action_space = [i for i in range(n_actions)]
        self.memory = ReplayMemory(input_dims, max_mem_size, batch_size, combined)

        self.iter_cntr = 0
        self.replace_target = replace_target

        Net = DuelingDeepQNetwork if self.dueling else DeepQNetwork
        self.Q_eval = Net(self.lr, input_dims=input_dims, fc1_dims=256, fc2_dims=200, n_actions=n_actions)
        self.Q_next = Net(self.lr, input_dims=input_dims, fc1_dims=256, fc2_dims=200, n_actions=n_actions)
        self.Q_next.load_state_dict(self.Q_eval.state_dict())


    def save_model(self, path: str):
        state = {
            'Q_eval': self.Q_eval.state_dict(),
            'Q_next': self.Q_next.state_dict(),
            'epsilon': self.epsilon,
            'iter_cntr': self.iter_cntr,
        }
        T.save(state, path)

    def load_model(self, path: str, strict: bool = True):
        state = T.load(path, map_location=self.Q_eval.device)
        self.Q_eval.load_state_dict(state['Q_eval'], strict=strict)
        self.Q_next.load_state_dict(state['Q_next'], strict=strict)
        if 'epsilon' in state:
            self.epsilon = float(state['epsilon'])
        if 'iter_cntr' in state:
            self.iter_cntr = int(state['iter_cntr'])

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            with torch.inference_mode():
                state = T.tensor([observation], dtype=T.float32, device=self.Q_eval.device)
                actions = self.Q_eval(state)
                action = int(T.argmax(actions).item())
        else:
            action = int(np.random.choice(self.action_space))
        return action

    def _soft_update_target(self, tau: float):
        with T.no_grad():
            for t_param, param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
                t_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def learn(self):
        if not self.memory.is_sufficient():
            return
        if self.memory.mem_cntr < self.learn_starts:
            return

        self.Q_eval.optimizer.zero_grad()

        batch_index = T.arange(self.batch_size, dtype=T.long, device=self.Q_eval.device)
        states, actions, rewards, new_states, dones = self.memory.sample_memory()

        states = T.tensor(states, dtype=T.float32, device=self.Q_eval.device)
        new_states = T.tensor(new_states, dtype=T.float32, device=self.Q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.Q_eval.device)
        if self.clip_reward:
            rewards = rewards.clamp(-1.0, 1.0)
        dones = T.tensor(dones, dtype=T.bool, device=self.Q_eval.device)
        actions = T.tensor(actions, dtype=T.long, device=self.Q_eval.device)

        q_eval_all = self.Q_eval(states)
        q_eval = q_eval_all[batch_index, actions]

        with T.no_grad():
            if self.double_dqn:
                q_next_eval = self.Q_eval(new_states)
                next_actions = T.argmax(q_next_eval, dim=1)
                q_next_target = self.Q_next(new_states)
                q_next_values = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q_next_values[dones] = 0.0
                q_target = rewards + self.gamma * q_next_values
            else:
                q_next = self.Q_next(new_states)
                q_next[dones] = 0.0
                q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_eval, q_target)
        loss.backward()
        clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        if self.memory.mem_cntr >= self.learn_starts and self.epsilon > self.eps_min:
            self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        if self.target_soft_tau and self.target_soft_tau > 0.0:
            self._soft_update_target(self.target_soft_tau)
        elif self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        return float(loss.item()) if loss is not None else None