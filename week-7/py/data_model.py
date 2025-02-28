from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch


@dataclass
class DataModel(ABC):
    _file_path: str
    data: np.ndarray
    mean: pd.Series
    std: pd.Series
    n_samples: int

    @abstractmethod
    def _clean_data(self, data: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_inputs(self, data: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_labels(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def _standardize_data(self, data: pd.DataFrame, columns:list) -> pd.DataFrame:
        """get z-score"""
        data[columns] = (data[columns] - self.mean[columns]) / self.std[columns]
        return data
    
    @abstractmethod
    def get_labels(self, x: float, y: float) -> float:
        pass


class GenderHeightWeight(DataModel):
    def __init__(self, file_path):
        # file_path = "./week-7/resource/gender-height-weight.csv"
        self._file_path = file_path

        xy_df = pd.read_csv(self._file_path)
        xy_np = self._clean_data(xy_df)

        self.data = xy_np

    def _clean_data(self, data):
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

        self.mean = data.mean()
        self.std = data.std()

        data.fillna(self.mean, inplace=True)
        data = self._standardize_data(data, ['Height', 'Weight'])

        return data.to_numpy(dtype=np.float32)
    
    def get_inputs(self):
        return self.data[:, :2]
    
    def get_labels(self):
        return self.data[:, [2]]
    
    def get_label_name(self) -> str:
        return 'Weight'
    
    def get_loss(self, x: float, y: float, key_std: float) -> float:
        return torch.abs(x - y) * key_std

    
class Titanic(DataModel):
    def __init__(self, file_path):
        # file_path = "./week-7/resource/titanic.csv"
        self._file_path = file_path

        xy_df = pd.read_csv(self._file_path)
        xy_np = self._clean_data(xy_df)

        # [Gender, Height, Weight]
        self.data = xy_np

    def _clean_data(self, data):
        pclass_mapping = {
            "1": [1, 0, 0],
            "2": [0, 1, 0],
            "3": [0, 0, 1]
        }
        data[['Pclass_1', 'Pclass_2', 'Pclass_3']] = pd.DataFrame(
            data['Pclass'].apply(
                lambda x: pclass_mapping.get(x, [0.33, 0.33, 0.33])
            ).tolist(),
            index=data.index
        )

        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Cabin'] = data['Cabin'].map(lambda x: 1 if x is not None else 0)

        embarked_mapping = {
            "Q": [1, 0, 0],
            "C": [0, 1, 0],
            "S": [0, 0, 1]
        }
        data[['Embarked_1', 'Embarked_2', 'Embarked_3']] = pd.DataFrame(
            data['Embarked'].apply(
                lambda x: embarked_mapping.get(x, [0, 0, 1])
            ).to_list(),
            index=data.index
        )

        data = data[[
            'Survived', 
            'Pclass_1', 
            'Pclass_2', 
            'Pclass_3', 
            'Sex', 
            'Age', 
            'SibSp', 
            'Parch',
            'Fare', 
            'Cabin', 
            'Embarked_1', 
            'Embarked_2', 
            'Embarked_3'
        ]].copy()
        self.mean = data.mean()
        self.std = data.std()

        data.fillna(self.mean, inplace=True)
        data = self._standardize_data(data, ['Age', 'SibSp', 'Parch', 'Fare'])
        return data.to_numpy(dtype=np.float32)
    
    def get_inputs(self):
        return self.data[:, 1:]
    
    def get_labels(self):
        return self.data[:, [0]]
    
    def get_loss(self, y: float, label: float) -> float:
        return (y >= 0.5) is (label >= 0.5)