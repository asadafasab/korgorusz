from typing import Tuple, Callable, Optional, List
import numpy as np
import random


array = np.ndarray


class Element:
    def __init__(self, tensor: array):
        self.tensor = tensor
        self.gradient = np.zeros_like(self.tensor)


class Base:
    """
    Base class for implementing different
    types of layers
    """

    def __init__(self):
        self.elements: List[Element] = []

    def forward(self, X: array):
        raise NotImplementedError

    def create_element(self, tensor: array) -> Element:
        element = Element(tensor)
        self.elements.append(element)
        return element


class Linear(Base):
    def __init__(self, inputs: int, outputs: int, bias: bool = True):
        super().__init__()
        tensor = np.random.randn(inputs, outputs) * np.sqrt(1 / inputs)
        self.weights = self.create_element(tensor)
        if bias:
            self.bias = self.create_element(np.zeros(outputs))
        else:
            self.bias = None

    def forward(self, x: array) -> Tuple[array, Callable]:
        def backward(derivative: array) -> array:
            self.weights.gradient += x.T @ derivative
            if self.bias:
                self.bias.gradient += derivative.sum(axis=0)
            return derivative @ self.weights.tensor.T

        if self.bias:
            return x @ self.weights.tensor + self.bias.tensor, backward
        return x @ self.weights.tensor, backward


class Dropout(Base):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x) -> Tuple[array, Callable]:
        nodes = int(x.shape[1] * self.dropout_rate)
        indices = sorted(random.sample(range(0, x.shape[1]), nodes))
        mask = np.ones_like(x)
        mask[:, indices] = 0

        def backward(derivative: array) -> array:
            return derivative * mask

        return x * mask, backward


class ReLU(Base):
    def forward(self, x: array) -> Tuple[array, Callable]:
        def backward(derivative: array) -> array:
            return np.where(x > 0, derivative, 0)

        return x * (x > 0), backward


class Softmax(Base):
    def forward(self, x: array) -> Tuple[array, Callable]:
        y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]

        def backward(derivative: array) -> array:
            return y * (derivative - (derivative * y).sum(axis=1)[:, None])

        return y, backward


class Sigmoid(Base):
    def forward(self, x: array) -> Tuple[array, Callable]:
        s = 1 / (1 + np.exp(-x))

        def backward(derivative: array) -> array:
            return derivative * s * (1 - s)

        return s, backward
