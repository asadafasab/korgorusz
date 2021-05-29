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

    def forward(self, x: array):
        raise NotImplementedError

    def create_element(self, tensor: array) -> Element:
        element = Element(tensor)
        self.elements.append(element)
        return element


class Linear(Base):
    def __init__(self, inputs: int, outputs: int, bias: Optional[bool] = True):
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


class Norm(Base):
    def __init__(self, shape):
        super().__init__()
        self.eps = eps
        self.weights = self.create_element(np.ones(shape))
        self.bias = self.create_element(np.zeros(shape))

        # TODO parent class for layerNorm and batchNorm (?)


class LayerNorm(Base):
    def __init__(
        self, shape: Tuple[int,...], eps: Optional[float] = 1e-05
    ):
        super().__init__()
        self.eps = eps
        self.weights = self.create_element(np.ones(shape))
        self.bias = self.create_element(np.zeros(shape))

    def forward(self, x: array) -> array:
        x_mean = x.mean( keepdims=True)
        x_var = np.var(x ,keepdims=True)
        x_std = np.sqrt(x_var + self.eps)
        x_centered = x - x_mean
        x_norm = x_centered / x_std

        def backward(derivative: array) -> array:
            C = derivative.shape[-1]
            self.weights.gradient += (derivative * x_norm).sum(axis=-1)
            self.bias.gradient += derivative.sum(axis=-1)
            dx_norm = derivative * self.weights.tensor
            return (
                1
                / C
                / x_std
                * (
                    C * dx_norm
                    - dx_norm.sum(axis=-1)
                    - x_norm * (dx_norm * x_norm).sum(axis=-1)
                )
            )

        return (x_norm * self.weights.tensor) + self.bias.tensor, backward


class Embedding(Base):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        tensor = np.random.randn(num_embeddings, embedding_dim) * np.sqrt(
            1 / embedding_dim
        )
        if padding_idx:
            tensor = tensor[padding_idx] = 0
        self.weights = self.create_element(tensor)

    def forward(self, x: array) -> Tuple[array, Callable]:
        msg = "Error: Embeding input must int type"
        assert x.dtype == np.int64() or np.int32() or np.int16() or np.int8(), msg

        def backward(derivative: array) -> array:
            # TODO trainable
            return derivative @ self.weights.tensor.T

        return self.weights.tensor[x], backward


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
