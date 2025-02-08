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
                np.array([[0.5, 0.2], [0.6, -0.6]])
            ],
            hidden_biases_matrices = [
                np.array([[0.3], [0.25]])
            ],
            hidden_activation_function = ActivationFunctions.RELU,
            output_weight_matrices = [
                np.array([[0.8, -0.5], [0.4, 0.5]])
            ],
            output_biases_matrices = [
                np.array([[0.6], [-0.25]])
            ],
            output_activation_function = ActivationFunctions.LINEAR
        )
        
        inputs_1 = np.array([[1.5], [0.5]])
        outputs = nn.forward(inputs_1)
        print(outputs)
        expects = np.array([[0.8], [1]])
        print("Total Loss", LossFunctions.MSE(outputs, expects))

        inputs_1 = np.array([[0], [1]])
        outputs = nn.forward(inputs_1)
        print(outputs)
        expects = np.array([[0.5], [0.5]])
        print("Total Loss", LossFunctions.MSE(outputs, expects))
    
    def task_2(self):
        print("============== Task 2 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.array([[0.5, 0.2], [0.6, -0.6]])
            ],
            hidden_biases_matrices = [
                np.array([[0.3], [0.25]])
            ],
            hidden_activation_function = ActivationFunctions.RELU,
            output_weight_matrices = [
                np.array([[0.8, 0.4]])
            ],
            output_biases_matrices = [
                np.array([[-0.5]])
            ],
            output_activation_function = ActivationFunctions.SIGMOID
        )
        
        inputs_2 = np.array([[0.75], [1.25]])
        outputs = nn.forward(inputs_2)
        print(outputs)
        expects = np.array([[1]])
        print("Total Loss", LossFunctions.BINARYCROSSENTROPY(outputs, expects))

        inputs_2 = np.array([[-1], [0.5]])
        outputs = nn.forward(inputs_2)
        print(outputs)
        expects = np.array([[0]])
        print("Total Loss", LossFunctions.BINARYCROSSENTROPY(outputs, expects))

    def task_3(self):
        print("============== Task 3 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.array([[0.5, 0.2], [0.6, -0.6]])
            ],
            hidden_biases_matrices = [
                np.array([[0.3], [0.25]])
            ],
            hidden_activation_function = ActivationFunctions.RELU,
            output_weight_matrices = [
                np.array([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]])
            ],
            output_biases_matrices = [
                np.array([[0.6], [0.5], [-0.5]])
            ],
            output_activation_function = ActivationFunctions.SIGMOID
        )
        
        inputs_3 = np.array([[1.5], [0.5]])
        outputs = nn.forward(inputs_3)
        print(outputs)
        expects = np.array([[1], [0], [1]])
        print("Total Loss", LossFunctions.BINARYCROSSENTROPY(outputs, expects))

        inputs_3 = np.array([[0], [1]])
        outputs = nn.forward(inputs_3)
        print(outputs)
        expects = np.array([[1], [1], [0]])
        print("Total Loss", LossFunctions.BINARYCROSSENTROPY(outputs, expects))

    def task_4(self):
        print("============== Task 4 ==============")
        nn = Network(
            hidden_weight_matrices = [
                np.array([[0.5, 0.2], [0.6, -0.6]])
            ],
            hidden_biases_matrices = [
                np.array([[0.3], [0.25]])
            ],
            hidden_activation_function = ActivationFunctions.RELU,
            output_weight_matrices = [
                np.array([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]])
            ],
            output_biases_matrices = [
                np.array([[0.6], [0.5], [-0.5]])
            ],
            output_activation_function = ActivationFunctions.SOFTMAX
        )
        
        inputs_3 = np.array([[1.5], [0.5]])
        outputs = nn.forward(inputs_3)
        print(outputs)
        expects = np.array([[1], [0], [0]])
        print("Total Loss", LossFunctions.CATEGORICALCROSSENTROPY(outputs, expects))

        inputs_3 = np.array([[0], [1]])
        outputs = nn.forward(inputs_3)
        print(outputs)
        expects = np.array([[0], [0], [1]])
        print("Total Loss", LossFunctions.CATEGORICALCROSSENTROPY(outputs, expects))

if __name__ == "__main__":
    task_handler = NNTaskHandler()
    task_handler.task_1()
    task_handler.task_2()
    task_handler.task_3()
    task_handler.task_4()