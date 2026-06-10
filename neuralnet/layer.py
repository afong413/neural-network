from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):  # MARK: Layer
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out

    @abstractmethod
    def __call__(self, v_in: np.ndarray) -> np.ndarray:
        """
        Propagate the layer with the input, `v_in`.
        """
        pass

    @abstractmethod
    def backprop(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate the layer given the derivative of the cost
        function with respect to the previous layer.
        """
        pass

    @abstractmethod
    def update(self, batch_size: int):
        """
        Update the neural network after a batch has completed.
        """
        pass

    def save(self, arrays: dict, prefix: str):
        """
        Save the layer to `arrays`.
        """
        pass

    def load(self, arrays: dict, prefix: str):
        """
        Load the layer from its entries in `arrays`.
        """
        pass
