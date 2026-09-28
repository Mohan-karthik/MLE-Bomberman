import torch
from torch import nn


class DuelingDQN(nn.Module):
    """Small CPU-friendly dueling Q-network."""

    def __init__(self, input_dim: int, number_of_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_actions),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        hidden = self.trunk(x)
        value = self.value_stream(hidden)
        advantage = self.advantage_stream(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
