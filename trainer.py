"""
trainer.py
Reinforcement Learning Optimizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random

class AgentTrainer:
    def __init__(self, brain, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state_tensor, num_actions=4):
        if random.random() < self.epsilon:
            return random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.brain(state_tensor)
                return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        current_q = self.brain(state)[action]

        with torch.no_grad():
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32)
            else:
                max_next_q = torch.max(self.brain(next_state))
                target_q = reward + (self.gamma * max_next_q)

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay