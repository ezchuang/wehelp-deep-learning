import numpy as np
from dataclasses  import dataclass, field
from activation_functions import ActivationFunction


@dataclass
class Network:
    hidden_weight_matrices: list[np.array]
    hidden_biases_matrices: list[np.array]
    hidden_activation_functions: list[ActivationFunction]
    output_weight_matrix: np.array
    output_biases_matrix: np.array
    output_activation_function: ActivationFunction

    input_cache: np.array = None
    hidden_cache: list[tuple] = field(default_factory=list)
    output_cache: tuple = None

    hidden_weight_grads: list[np.array] = field(default_factory=list)
    hidden_bias_grads: list[np.array] = field(default_factory=list)
    output_weight_grad: np.array = None
    output_bias_grad: np.array = None

    def forward(self, inputs: np.array) -> np.array:
        self.hidden_cache.clear()
        self.input_cache = inputs
        a = inputs

        for i, hidden_weight_matrix in enumerate(self.hidden_weight_matrices):
            z = np.dot(a, hidden_weight_matrix) + self.hidden_biases_matrices[i]
            a = self.hidden_activation_functions[i].forward(z)
            self.hidden_cache.append((z, a))

        z_out = np.dot(a, self.output_weight_matrix) + self.output_biases_matrix
        a_out = self.output_activation_function.forward(z_out)
        self.output_cache = (z_out, a_out)

        return a_out
    
    def backward(self, output_loss: np.array):
        num_hidden_layers = len(self.hidden_weight_matrices)
        if len(self.hidden_weight_grads) != num_hidden_layers:
            self.hidden_weight_grads = [np.zeros_like(w) for w in self.hidden_weight_matrices]
        if len(self.hidden_bias_grads) != num_hidden_layers:
            self.hidden_bias_grads = [np.zeros_like(b) for b in self.hidden_biases_matrices]

        z_out = self.output_cache[0]
        dZ_out = output_loss * self.output_activation_function.derivative(z_out)
        a_in_for_output = self.hidden_cache[-1][1]

        self.output_weight_grad = np.dot(a_in_for_output.T, dZ_out)
        self.output_bias_grad = np.sum(dZ_out, axis=0, keepdims=True)
        dA_in_for_output = np.dot(dZ_out, self.output_weight_matrix.T)

        dA = dA_in_for_output
        for i in reversed(range(num_hidden_layers)):
            z_i = self.hidden_cache[i][0]
            dZ_i = dA * self.hidden_activation_functions[i].derivative(z_i)
            a_prev = self.hidden_cache[i-1][1] if i != 0 else self.input_cache

            self.hidden_weight_grads[i] = np.dot(a_prev.T, dZ_i)
            self.hidden_bias_grads[i] = np.sum(dZ_i, axis=0, keepdims=True)

            w_i = self.hidden_weight_matrices[i]
            dA = np.dot(dZ_i, w_i.T)

    def zero_grad(self, learning_rate: float):
        # update output layer
        if self.output_weight_grad is not None:
            self.output_weight_matrix -= learning_rate * self.output_weight_grad
        if self.output_bias_grad is not None:
            self.output_biases_matrix -= learning_rate * self.output_bias_grad

        # update hidden layers
        for i in range(len(self.hidden_weight_matrices)):
            if i < len(self.hidden_weight_grads) and self.hidden_weight_grads[i] is not None:
                self.hidden_weight_matrices[i] -= learning_rate * self.hidden_weight_grads[i]
            if i < len(self.hidden_bias_grads) and self.hidden_bias_grads[i] is not None:
                self.hidden_biases_matrices[i] -= learning_rate * self.hidden_bias_grads[i]

        # clear gradients
        self.output_weight_grad = None
        self.output_bias_grad = None
        for i in range(len(self.hidden_weight_grads)):
            self.hidden_weight_grads[i] = np.zeros_like(self.hidden_weight_grads[i])
        for i in range(len(self.hidden_bias_grads)):
            self.hidden_bias_grads[i] = np.zeros_like(self.hidden_bias_grads[i])