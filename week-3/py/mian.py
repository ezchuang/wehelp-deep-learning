from dataclasses  import dataclass
import numpy as np

@dataclass
class Inputs:
    values: np.array

@dataclass
class Network:
    weight_matrices: list[np.array]
    biases_matrices: list[np.array]

    def forward(self, inputs: Inputs) -> np.array:
        res: np.array = inputs.values

        for i, weight_matrix in enumerate(self.weight_matrices):
            bias = self.biases_matrices[i]
            res = np.dot(res, weight_matrix) + bias

        return res

class NNTaskHandler:
    def __init__(self):
        pass

    def task_1(self):
        nn = Network(
            weight_matrices = [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
                np.array([[0.8], [0.4]])
            ],
            biases_matrices = [
                np.array([0.3, 0.25]),
                np.array([-0.5])
            ])
        
        inputs_1 = Inputs(np.array([1.5, 0.5]))
        outputs = nn.forward(inputs_1)
        print(outputs)

        inputs_2 = Inputs(np.array([0, 1]))
        outputs = nn.forward(inputs_2)
        print(outputs)
    
    def task_2(self):
        nn = Network(
            weight_matrices = [
                np.array([[0.5, 0.6], [1.5, -0.8]]),
                np.array([[0.6], [-0.8]]),
                np.array([[0.5, -0.4]])
            ],
            biases_matrices = [
                np.array([0.3, 1.25]),
                np.array([0.3]),
                np.array([0.2, 0.5])
            ])
        
        inputs_1 = Inputs(np.array([0.75, 1.25]))
        outputs = nn.forward(inputs_1)
        print(outputs)

        inputs_2 = Inputs(np.array([-1, 0.5]))
        outputs = nn.forward(inputs_2)
        print(outputs)

if __name__ == "__main__":
    task_handler = NNTaskHandler()
    task_handler.task_1()
    task_handler.task_2()