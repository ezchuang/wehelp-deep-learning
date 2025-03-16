import numpy as np
from dataclasses  import dataclass
from activation_functions import ActivationFunction
from loss_functions import LossFunction


@dataclass
class Network:
    hidden_weight_matrices: list[np.array]
    hidden_biases_matrices: list[np.array]
    hidden_activation_function: ActivationFunction
    output_weight_matrices: list[np.array]
    output_biases_matrices: list[np.array]
    output_activation_function: ActivationFunction

    def forward(self, inputs: np.array) -> np.array:
        res = inputs

        for i, hidden_weight_matrix in enumerate(self.hidden_weight_matrices):
            res = np.dot(hidden_weight_matrix, res) + self.hidden_biases_matrices[i]
            res = self.hidden_activation_function(res)

        for i, output_weight_matrix in enumerate(self.output_weight_matrices):
            res = np.dot(output_weight_matrix, res) + self.output_biases_matrices[i]
            res = self.output_activation_function(res)

        return res