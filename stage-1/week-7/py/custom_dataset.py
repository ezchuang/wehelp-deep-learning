import torch
from torch.utils.data import Dataset
import numpy as np
from data_model import DataModel


class CustomDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, device: torch.device):
        self.x = torch.from_numpy(x).to(device)
        self.y = torch.from_numpy(y).to(device)
        self._n_samples = x.shape[0]

    @classmethod
    def create_from_data_model(cls, data_model: DataModel, device: torch.device):
        return cls(data_model.get_inputs(), data_model.get_labels(), device)

    def __getitem__(self, index) -> torch.Tensor:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return self._n_samples
