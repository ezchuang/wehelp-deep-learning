import numpy as np
from network import Network
from activation_functions import ActivationFunctions
from loss_functions import LossFunctions


class NNTaskHandler:
    def __init__(self):
        pass

    def task_1(self):
        print("============== Task 1 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
                np.array([[0.8], [-0.5]])
            ],
            hidden_biases_matrices = [
                np.array([[0.3, 0.25]]),
                np.array([[0.6]])
            ],
            hidden_activation_functions = [
                ActivationFunctions.RELU,
                ActivationFunctions.LINEAR
            ],
            output_weight_matrix = np.array([[0.6, -0.3]]),
            output_biases_matrix = np.array([[0.4, 0.75]]),
            output_activation_function = ActivationFunctions.LINEAR
        )
        learning_rate: float = 0.01
        
        print("=========== Task 1-1 ===========")
        inputs_1 = np.array([1.5, 0.5])
        outputs = nn.forward(inputs_1)
        print("Output: ", outputs)
        expects = np.array([0.8, 1])
        loss = LossFunctions.MSE.forward(outputs, expects)
        print("Total Loss: ", loss)
        output_losses = LossFunctions.MSE.derivative(outputs, expects)
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)
        print("Hidden Weights: ", nn.hidden_weight_matrices)
        print("Output Weights: ", nn.output_weight_matrix)

        print("=========== Task 1-2 ===========")
        repeat_times = 999
        for i in range(repeat_times):
            inputs_1 = np.array([1.5, 0.5])
            outputs = nn.forward(inputs_1)
            # print("Output: ", outputs)
            expects = np.array([0.8, 1])
            loss = LossFunctions.MSE.forward(outputs, expects)
            output_losses = LossFunctions.MSE.derivative(outputs, expects)
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

        print("Total Loss: ", loss)
        print("Hidden Weights: ", nn.hidden_weight_matrices)
        print("Output Weights: ", nn.output_weight_matrix)
    
    def task_2(self):
        print("============== Task 2 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
            ],
            hidden_biases_matrices = [
                np.array([[0.3, 0.25]])
            ],
            hidden_activation_functions = [
                ActivationFunctions.RELU
            ],
            output_weight_matrix = np.array([[0.8], [0.4]]),
            output_biases_matrix = np.array([[-0.5]]),
            output_activation_function = ActivationFunctions.SIGMOID
        )
        learning_rate: float = 0.1
        
        print("=========== Task 2-1 ===========")
        inputs_2 = np.array([0.75, 1.25])
        outputs = nn.forward(inputs_2)
        print("Output: ", outputs)
        expects = np.array([1])
        loss = LossFunctions.BINARYCROSSENTROPY.forward(outputs, expects)
        print("Total Loss: ", loss)
        output_losses = LossFunctions.BINARYCROSSENTROPY.derivative(outputs, expects)
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)
        print("Hidden Weights: ", nn.hidden_weight_matrices)
        print("Output Weights: ", nn.output_weight_matrix)

        print("=========== Task 2-2 ===========")
        repeat_times = 999
        for i in range(repeat_times):
            inputs_2 = np.array([0.75, 1.25])
            outputs = nn.forward(inputs_2)
            # print("Output: ", outputs)
            expects = np.array([1])
            loss = LossFunctions.BINARYCROSSENTROPY.forward(outputs, expects)
            output_losses = LossFunctions.BINARYCROSSENTROPY.derivative(outputs, expects)
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

        print("Total Loss: ", loss)
        print("Hidden Weights: ", nn.hidden_weight_matrices)
        print("Output Weights: ", nn.output_weight_matrix)

if __name__ == "__main__":
    task_handler = NNTaskHandler()
    task_handler.task_1()
    task_handler.task_2()