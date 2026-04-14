import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score

# 环境定义
class OptimizationEnv:
    def __init__(self, mdm, pop, n):
        self.reset(mdm, pop, n)

    def reset(self, mdm, pop, n):
        # 生成 n 个解，每个解的特征和为 1
        self.mdm = mdm
        self.n = n
        self.selected_solutions = self.generate_normalized_solutions(mdm, pop, n)
        return self.selected_solutions

    def generate_normalized_solutions(self, mdm, pop, n):
        # 生成满足和为 1 的解集
        sorted_indices, sorted_solutions = mdm.rank_solutions(pop.get_f())
        return sorted_solutions[:n]

    def update_state_and_reward(self, flattened_state, candidates):
        # 合并当前状态中的 n 个解和新的 N 个候选解
        m = len(flattened_state)//self.n
        current_state = flattened_state.reshape(self.n, m)
        combined_solutions = np.vstack([current_state, candidates])

        # 根据用户偏好对合并解集进行排序
        sorted_indices, sorted_solutions = self.mdm.rank_solutions(combined_solutions)
        
        # 选取排名靠前的 n 个解作为新的状态
        new_state = sorted_solutions[:self.n].flatten()

        # 计算奖励：新状态中的解有多少来自于候选解
        reward = 0
        for i, idx in enumerate(sorted_indices[:self.n]):
            if idx >= self.n:  # 检查是否来自候选解
                reward += (0.9 ** i)  # 使用指数递减分配奖励
        
        return new_state, reward

# Actor网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出范围[-1, 1]

# Critic网络定义
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, action):
        x = x if x.dim() > 1 else x.unsqueeze(0)
        action = action if action.dim() > 1 else action.unsqueeze(0)
        
        # 拼接操作
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# LambdaMART模型定义
class LambdaMART:
    def __init__(self):
        # 初始化一个简单的GradientBoostingRegressor替代LambdaMART
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# DDPG算法
class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma, tau, noise_std, buffer_size=100000, batch_size=64):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # 复制网络参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 初始化经验回放池
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        # 添加噪声进行探索
        action = action + self.noise_std * np.random.normal(size=action.shape)
        return np.clip(action, -1, 1)  # 限制动作范围在[-1, 1]

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def update(self, transition=None):
        if transition is not None:
            self.store_transition(transition)
        if len(self.replay_buffer) < self.batch_size:
            return  # 当经验池中的经验不够时，不进行训练
        
        # 从经验回放池中随机采样一个小批量
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state = zip(*batch)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        
        # 更新Critic网络
        with torch.no_grad():
            target_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, target_action)
            y = reward + self.gamma * target_q

        critic_loss = nn.MSELoss()(self.critic(state, action), y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)