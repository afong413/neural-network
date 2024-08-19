#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import sympy as sym
import copy
from collections.abc import Callable
from inspect import signature


class Network:
    def __init__(
        self,
        layers: list["Layer"],
        cost_function: Callable[[float, float], float],
    ):
        self.layers = list(layers)
        self.cost_function = cost_function
        self.v_d_cost_function = np.vectorize(self.d_cost_function)

    def d_cost_function(self, v_out, desired_v_out):
        x, y = sym.symbols("x y")

        return sym.diff(self.cost_function(x, y), x).evalf(
            subs={x: v_out, y: desired_v_out}
        )

    def propagate(self, v_in: np.ndarray[float]) -> np.ndarray[float]:
        for layer in self.layers:
            v_in = layer.propagate(v_in)

        return v_in

    def backpropagate(self, desired_v_out: np.ndarray[float]):
        d_out = self.cost_function(desired_v_out)

        for layer in reversed(self.layers):
            d_out = layer.backpropagate(d_out)

    def __add__(self, other: "Layer | Network") -> "Network":
        if isinstance(other, Layer):
            return Network(copy.deepcopy(self.layers + [other]))
        elif isinstance(other, Network):
            return Network(copy.deepcopy(self.layers + other.layers))


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n = n_in
        self.m = n_out

    def propagate(self, v_in: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def backpropagate(self, d_out: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def __add__(self, other: "Layer | Network") -> "Network":
        if isinstance(other, Layer):
            return Network(copy.deepcopy([self, other]))
        elif isinstance(other, Network):
            return Network(copy.deepcopy([self] + other.layers))


class DenseLayer(Layer):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        learning_rate: float,
        activation_function: Callable[[float], float],
        weight_initialization: Callable[[int, int], float],
        bias_initialization: Callable[[int, int], float] = lambda i, j: 0,
    ):
        super().__init__(n_in, n_out)

        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.v_activation_function = np.vectorize(activation_function)
        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization

        self.weights = np.fromfunction(
            lambda i, j: self.weight_initialization(n_in, n_out), (n_out, n_in)
        )

        self.biases = np.fromfunction(
            lambda i: self.bias_initialization(n_in, n_out), [n_out]
        )

        self.d_biases = np.zeros((self.n_out, self.n_in))

        self.v_d_activation_function = np.vectorize(self.d_activation_function)

    def d_activation_function(self, z):
        x = sym.symbols("x")

        return sym.diff(self.activation_function(x)).evalf(subs={x: z})

    def propagate(self, v_in: np.ndarray[float]) -> np.ndarray[float]:
        self.v_in = v_in

        self.preactivation = self.weights @ v_in + self.biases

        return self.v_activation_function(self.preactivation)

    def backpropagate(self, d_out: np.ndarray[float]) -> np.ndarray[float]:
        d_biases = d_out * self.v_d_activation_function(self.preactivation)

        d_weights = d_biases[:, None] @ self.v_in[None, :]

        d_in = self.weights.T @ d_biases

        self.biases -= self.learning_rate * d_biases
        self.weights -= self.learning_rate * d_weights

        return d_in


class DenseNetwork(Network):
    def __init__(
        self,
        layer_sizes: list[int],
        cost_function: Callable[[float, float], float],
        learning_rate: float,
        activation_function: Callable[[float], float],
        weight_initialization: Callable[[int, int], float],
        bias_initialization: Callable[[int, int], float] = lambda i, j: 0,
    ):
        super().__init__(
            [
                DenseLayer(
                    s,
                    learning_rate,
                    activation_function,
                    weight_initialization,
                    bias_initialization,
                )
                for s in layer_sizes
            ],
            cost_function,
        )
