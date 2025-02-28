import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GHWNet(nn.Module):
    def __init__(self, input_dim: int=2, hidden_dim: int=16, output_dim: int=1):
        super(GHWNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class TitanicNet(nn.Module):
    def __init__(self, input_dim: int=12, hidden_dim_1: int=8, hidden_dim_2: int=4, output_dim: int=1):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x