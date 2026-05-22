"""
models.py
Universal Deep Q-Network for Multi-Species Simulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalBrain(nn.Module):
    def __init__(self, input_size, output_size):
        super(AnimalBrain, self).__init__()
        # Input: [Hunger, Stress, Dist_Water, Dist_Crops, Human_Density]
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Output: Q-Values for [North, South, East, West]
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_q_values = self.fc3(x)
        return actions_q_values