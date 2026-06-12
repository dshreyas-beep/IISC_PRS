"""
models.py
Continuous Actor-Critic Network for PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalBrain(nn.Module):
    def __init__(self, input_size, action_dim=2):
        super(AnimalBrain, self).__init__()
        # Actor
        self.actor_fc1 = nn.Linear(input_size, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic
        self.critic_fc1 = nn.Linear(input_size, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_value = nn.Linear(64, 1)

    def forward(self, x):
        # Actor forward
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        action_mean = torch.tanh(self.actor_mean(a)) # Bounded mean
        action_std = self.actor_log_std.expand_as(action_mean).exp()
        
        # Critic forward
        v = F.relu(self.critic_fc1(x))
        v = F.relu(self.critic_fc2(v))
        state_value = self.critic_value(v)
        
        return action_mean, action_std, state_value