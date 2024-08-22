from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from neuralnet.layer import Layer
from neuralnet.calculus import CalcFunction


class DenseLayer(Layer):  # MARK: DenseLayer
    """
    A perceptron layer.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        learning_rate: float,
        momentum: float,
        activation_function: CalcFunction,
        weight_initialization_function: Callable[
            [int, int], np.ndarray[Any, float]
        ],
    ):
        super().__init__(n_in, n_out)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_function = activation_function

        self.weights = np.array(
            weight_initialization_function(n_in, n_out), dtype=np.longdouble
        )

        self.biases = np.zeros(n_out, dtype=np.longdouble)

        self.d_weights = np.zeros((n_out, n_in), dtype=np.longdouble)

        self.d_biases = np.zeros(n_out, dtype=np.longdouble)

    def __call__(self, v_in: Sequence):
        """
        Propagate the layer with the given input, `v_in`.
        """
        self.v_in = np.array(v_in, dtype=np.longdouble)

        self.preactivation = (
            self.weights @ self.v_in
        )  # The @ represents matrix multiplication.

        return self.activation_function(self.preactivation)

    def backprop(self, d_out: Sequence):
        """
        Backpropagate the layer without yet updating the parameters.
        """
        d_out = np.array(
            d_out, dtype=np.longdouble
        )  # Don't want any pesky lists messing things up!

        # This is just calculus bash and matrices:

        dCdZ = d_out * self.activation_function.d(self.preactivation)

        self.d_weights += dCdZ[:, None] @ self.v_in[None, :]

        self.d_biases += dCdZ

        return self.weights.T @ dCdZ

    def update(self, batch_size: int):
        """
        Update the layer.
        """

        # Funny story: I spent way too long trying to figure out why
        # the gradients kept exploding. In the end, it was because I
        # forgot to divide by batch size, making them change way too
        # quickly.

        # Not quite sure why I can't just use -= for the below, but
        # numpy kept throwing an error.

        self.weights = (
            self.weights - (self.learning_rate / batch_size) * self.d_weights
        )
        self.biases = (
            self.biases - (self.learning_rate / batch_size) * self.d_biases
        )
        
        self.d_weights *= self.momentum
        self.d_biases *= self.momentum
