from abc import abstractmethod
from collections.abc import Callable, Sequence


class Layer(Callable):  # MARK: Layer
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out

    @abstractmethod
    def __call__(self, v_in: Sequence[float]) -> Sequence[float]:
        """
        Propagate the layer with the input, `v_in`.
        """
        pass

    @abstractmethod
    def backprop(self, d_out: Sequence[float]) -> Sequence[float]:
        """
        Backpropagate the layer given the derivative of the cost
        unction with respect to the previous layer.
        """
        pass

    @abstractmethod
    def update(self, batch_size: int):
        """
        Update the neural network after a batch has completed.
        """
        pass
