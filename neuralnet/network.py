from collections.abc import Callable, Sequence, MutableSequence
import copy

import numpy as np

from .layer import Layer
from .calculus import CalcFunction


class Network(Callable, MutableSequence):  # MARK: Network
    """
    A neural network.
    """

    def __init__(self, layers: list[Layer], cost_function: CalcFunction):
        self.layers = layers
        self.cost_function = cost_function

    def __call__(self, v_in: Sequence[float]) -> Sequence[float]:
        """
        Propagate the network with the given input, `v_in`.
        """
        self.v_out = np.array(v_in)

        for layer in self.layers:
            self.v_out = layer(self.v_out)

        return self.v_out

    def backprop(self, expected_v_out: Sequence[float]):
        """
        Backpropagate the network given the desired output, `expected_v_out`.
        """
        d_out = self.cost_function.d(self.v_out, expected_v_out)
        for layer in reversed(self.layers):
            d_out = layer.backprop(d_out)

    def update(self, batch_size: int):
        """
        Updates each layer after a batch has finished running.
        """
        for layer in self.layers:
            layer.update(batch_size)

    # Most of these are to conform to MutableSequence:

    def __add__(self, other: "Network"):
        return Network(copy.deepcopy(self.layers + other.layers))

    def __getitem__(self, *args, **kwargs):
        return self.layers.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.layers.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.layers.__delitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.layers.__len__(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.layers.insert(*args, **kwargs)
