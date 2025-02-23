import numpy as np
from network import Network
from activation_functions import ActivationFunctions
from loss_functions import LossFunctions
import data_reader
import torch


class NNTaskHandler:
    @staticmethod
    def task_1():
        print("============== Task 1 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.random.randn(2, 8) * 0.01,
                np.random.randn(8, 4) * 0.01,
            ],
            hidden_biases_matrices = [
                np.zeros((1, 8)),
                np.zeros((1, 4))
            ],
            hidden_activation_functions = [
                ActivationFunctions.RELU,
                ActivationFunctions.LINEAR
            ],
            output_weight_matrix = np.random.randn(4, 1) * 0.01,
            output_biases_matrix = np.zeros((1, 1)),
            output_activation_function = ActivationFunctions.LINEAR
        )

        learning_rate: float = 0.01
        training_size: float = 0.7
        epochs = 1

        dataset = data_reader.GenderHeightWeight()
        training_data = dataset.get_training_data(training_size)
        training_labels = dataset.get_training_labels(training_size)
        testing_data = dataset.get_testing_data(training_size)
        testing_labels = dataset.get_testing_labels(training_size)
        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(dataset.get_training_size_boundary(training_size)):
                inputs_1 = np.atleast_2d(training_data[i])
                expects = np.atleast_2d(training_labels[i])

                outputs = nn.forward(inputs_1)
                loss = LossFunctions.MSE.forward(outputs, expects)
                if np.isnan(loss):
                    print(f"NaN loss at index {i}")
                    return
                epoch_loss += loss
                output_losses = LossFunctions.MSE.derivative(outputs, expects)
                
                nn.backward(output_losses)
                nn.zero_grad(learning_rate)
                # print(output_losses)

            # avg_epoch_loss = epoch_loss / dataset.get_training_size_boundary(training_size)
            # print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss (z-score): {avg_epoch_loss}")

        # print("============== Evaluating Procedure ==============")
        loss_sum = 0
        for x, e in zip(testing_data, testing_labels):
            inputs = np.atleast_2d(x)
            expects = np.atleast_2d(e)
            outputs = nn.forward(inputs)
            # loss = LossFunctions.MSE.forward(outputs, expects)
            loss_sum += np.abs(outputs - expects)
        
        avg_loss = loss_sum / len(testing_data)
        print(f"Average Test Loss (z-score): {avg_loss[0][0]}")
        print(f"Average Test Loss (pound): {(avg_loss * dataset.std)[0][0]}")
        print(f"Average Test Loss (percentage): {(avg_loss * dataset.std / dataset.mean * 100)[0][0]} %")
    
    @staticmethod
    def task_2():
        print("============== Task 2 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.random.randn(12, 8),
                np.random.randn(8, 4),
            ],
            hidden_biases_matrices = [
                np.ones((1, 8)),
                np.ones((1, 4)),
            ],
            hidden_activation_functions = [
                ActivationFunctions.RELU,
                ActivationFunctions.RELU
            ],
            output_weight_matrix=np.random.randn(4, 1),
            output_biases_matrix=np.ones((1, 1)),
            output_activation_function = ActivationFunctions.SIGMOID
        )

        learning_rate: float = 0.01
        training_size: float = 0.8
        epochs = 1

        dataset = data_reader.Titanic()
        training_data = dataset.get_training_data(training_size)
        training_labels = dataset.get_training_labels(training_size)
        testing_data = dataset.get_testing_data(training_size)
        testing_labels = dataset.get_testing_labels(training_size)

        # print(f"Input shape: {training_data[0].shape}, Expected: [1, 5]")

        for epoch in range(epochs):
            # epoch_loss = 0

            for i in range(dataset.get_training_size_boundary(training_size)):
                inputs_1 = np.atleast_2d(training_data[i])
                expects = np.atleast_2d(training_labels[i])

                outputs = nn.forward(inputs_1)
                loss = LossFunctions.BINARYCROSSENTROPY.forward(outputs, expects)
                if np.isnan(loss):
                    print(f"NaN loss at index {i}")
                    return
                # epoch_loss += loss
                output_losses = LossFunctions.BINARYCROSSENTROPY.derivative(outputs, expects)
                
                nn.backward(output_losses)
                nn.zero_grad(learning_rate)

            # avg_epoch_loss = epoch_loss / dataset.get_training_size_boundary(training_size)
            # print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss (z-score): {avg_epoch_loss}")

        # print("============== Evaluating Procedure ==============")
        correct_count = 0
        threshold = 0.5

        for x, e in zip(testing_data, testing_labels):
            outputs = nn.forward(x)
            survival_status = 0
            if outputs >= threshold:
                survival_status = 1
            if survival_status == e:
                correct_count += 1
            # print(survival_status, e, survival_status == e)

        correct_rate = correct_count / len(testing_data)

        print(f"Correct Rate (percentage): {correct_rate * 100} %")

    @staticmethod
    def task_3():
        print("============== Task 3 ==============")
        # 3-1
        tensor = torch.tensor([[2, 3, 1], [5, -2, 1]])
        print(f"**3-1** tensor: {tensor} tensor.shape: {tensor.shape}, tensor.dtype: {tensor.dtype}")

        # # 3-2
        tensor = torch.rand(3, 4, 2)
        print(f"**3-2** tensor: {tensor} tensor.shape: {tensor.shape}, tensor.dtype: {tensor.dtype}")

        # 3-3
        tensor = torch.ones(2, 1, 5)
        print(f"**3-3** tensor: {tensor} tensor.shape: {tensor.shape}, tensor.dtype: {tensor.dtype}")

        # 3-4
        tensor = torch.tensor([[1, 2, 4], [2, 1, 3]]) @ torch.tensor([[5], [2], [1]])
        print(f"**3-4** tensor: {tensor} tensor.shape: {tensor.shape}, tensor.dtype: {tensor.dtype}")
        
        # 3-5
        tensor = torch.tensor([[1, 2], [2, 3], [-1, 3]]) * torch.tensor([[5, 4], [2, 1], [1, -5]])
        print(f"**3-5** tensor: {tensor} tensor.shape: {tensor.shape}, tensor.dtype: {tensor.dtype}")


if __name__ == "__main__":
    NNTaskHandler.task_1()
    NNTaskHandler.task_2()
    # NNTaskHandler.task_3()