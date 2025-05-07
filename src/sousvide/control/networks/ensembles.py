import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim:int, hidden_sizes:list[int], output_dim:int):
        super().__init__()

        prev_size = input_dim
        self.net = nn.Sequential()
        for layer_size in hidden_sizes:
            self.net.append(nn.Linear(prev_size, layer_size))
            self.net.append(nn.ReLU())
            prev_size = layer_size
        self.net.append(nn.Linear(prev_size, output_dim))
        
    def forward(self, x):
        return self.net(x)

class Gating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)  # (B, E)