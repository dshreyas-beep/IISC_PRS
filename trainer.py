"""
trainer.py
PPO Reinforcement Learning Optimizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class AgentTrainer:
    def __init__(self, brain, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2, ppo_epochs=4):
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        
        self.memory = []

    def choose_action(self, state_tensor):
        with torch.no_grad():
            # Add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
            mean, std, value = self.brain(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        # action is shape (1, action_dim), so action.numpy()[0] returns the action_dim vector
        return action.numpy()[0], log_prob.numpy()[0], value.numpy()[0]

    def remember(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))

    def learn(self):
        if len(self.memory) == 0:
            return
            
        states, actions, old_log_probs, rewards, values, dones = zip(*self.memory)
        
        # Convert to tensors
        states = torch.cat(states)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute advantages
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(self.ppo_epochs):
            mean, std, state_values = self.brain(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratios = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.memory.clear()