import copy

import numpy as np

from .calculus import CalcFunction
from .layer import Layer


class Network:  # MARK: Network
    """
    A neural network.
    """

    def __init__(self, layers: list[Layer], cost_function: CalcFunction):
        self.layers = layers
        self.cost_function = cost_function

    def __call__(self, v_in: np.ndarray) -> np.ndarray:
        """
        Propagate the network with the given input, `v_in`.
        """
        v_out = np.array(v_in)

        for layer in self.layers:
            v_out = layer(v_out)

        return v_out

    def backprop(self, v_out: np.ndarray, expected_v_out: np.ndarray):
        """
        Backpropagate the network given the actual output, `v_out`, and desired output, `expected_v_out`.
        """
        d_out = self.cost_function.d(v_out, expected_v_out)
        for layer in reversed(self.layers):
            d_out = layer.backprop(d_out)

    def update(self, batch_size: int):
        """
        Updates each layer after a batch has finished running.
        """
        for layer in self.layers:
            layer.update(batch_size)

    def __add__(self, other: 'Network'):
        return Network(copy.deepcopy(self.layers + other.layers), other.cost_function)

    def save(self, path):
        """
        Save the network to the specified path.
        """
        arrays = {}
        for i, layer in enumerate(self.layers):
            layer.save(arrays, f'layer_{i}')
        np.savez(path, **arrays)

    def load(self, path):
        """
        Load the network from the specified path.
        """
        with np.load(path) as arrays:
            for i, layer in enumerate(self.layers):
                layer.load(arrays, f'layer_{i}')
