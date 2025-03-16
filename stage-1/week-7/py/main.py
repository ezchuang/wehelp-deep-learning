import numpy as np
import data_model
from custom_dataset import CustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from custom_nn import GHWNet, TitanicNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def shuffle_and_split(dataset_size: int, random_seed: int, 
        shuffle: bool = True, split_ratio: int = 0.3) -> tuple[list[int], list[int]]:
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))

    if random_seed:
        np.random.seed(random_seed)
    if shuffle:
        np.random.shuffle(indices)

    return indices[split:], indices[:split]

class NNTaskHandler:
    @staticmethod
    def task_1():
        print("============== Task 1 ==============")
        model = GHWNet().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # data parameters
        test_split = 0.3
        random_seed = 1234
        shuffle_dataset = True

        # training parameters
        epochs = 10
        batch_size = 1

        # import data
        file_path = "./week-7/resource/gender-height-weight.csv"
        data_cleaned = data_model.GenderHeightWeight(file_path)
        dataset = CustomDataset.create_from_data_model(data_cleaned, device)

        train_indices, test_indices = shuffle_and_split(len(dataset), shuffle_dataset, random_seed, test_split)

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        # training
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                # x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = torch.Tensor(criterion(outputs, y))

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss = loss.item()
                # if (epoch + 1) % 50 == 0:
                #     print(f'epoch {epoch + 1}: loss = {loss:.8f}')

        # testing
        model.eval()
        total_loss = torch.zeros(batch_size, 1).to(device)
        std = data_cleaned.std[data_cleaned.get_label_name()]
        mean = data_cleaned.mean[data_cleaned.get_label_name()]

        with torch.no_grad():
            for x, y in test_loader:
                # x, y = x.to(device), y.to(device)
                output = model(x)
                total_loss += torch.abs(output - y)
        
        avg_loss = (total_loss.sum() / len(test_loader) / batch_size).item()
        print(f"Average Test Loss (z-score): {avg_loss:.8f}")
        print(f"Average Test Loss (pound): {avg_loss * std:.8f}")
        print(f"Average Test Loss (percentage): {avg_loss * std / mean * 100:.8f} %")

    @staticmethod
    def task_2():
        print("============== Task 2 ==============")
        model = TitanicNet().to(device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # data parameters
        test_split = 0.3
        random_seed = None
        shuffle_dataset = True

        # training parameters
        epochs = 100
        batch_size = 1

        # import data
        file_path = "./week-7/resource/titanic.csv"
        data_cleaned = data_model.Titanic(file_path)
        dataset = CustomDataset.create_from_data_model(data_cleaned, device)

        train_indices, test_indices = shuffle_and_split(len(dataset), shuffle_dataset, random_seed, test_split)

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        # training
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                # x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = torch.Tensor(criterion(outputs, y))

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss = loss.item()
                # if (epoch + 1) % 50 == 0:
                #     print(f'epoch {epoch + 1}: loss = {loss:.8f}')

        # testing
        model.eval()
        total_counts = 0
        threshold = 0.5

        with torch.no_grad():
            for x, y in test_loader:
                # x, y = x.to(device), y.to(device)
                output = model(x)
                # print("output shape:", output.shape, "y shape:", y.shape)
                total_counts += ((output >= threshold) == (y >= threshold)).int().sum()
        
        correct_rate = (total_counts.sum() / len(test_loader) / batch_size).item()
        print(f"Correct Rate (percentage): {correct_rate * 100} %")


if __name__ == "__main__":
    NNTaskHandler.task_1()
    NNTaskHandler.task_2()