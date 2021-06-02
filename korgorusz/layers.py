"""
Contains layer, activations
"""
import random
from typing import Callable, Iterator, List, Optional, Tuple
import numpy as np


Array = np.ndarray


class Element:
    """
    Base object for creating weights,biases etc.
    """

    def __init__(self, tensor: Array):
        self.tensor = tensor
        self.gradient = np.zeros_like(self.tensor)


class Base:
    """
    Base class for implementing different
    types of layers
    """

    def __init__(self):
        self.elements: List[Element] = []

    def forward(self, x: Array):
        """
        Calculates and returns an array.
        """
        raise NotImplementedError

    def create_element(self, tensor: Array) -> Element:
        """
        Helper method for creating weights,biases,etc.
        """
        element = Element(tensor)
        self.elements.append(element)
        return element


class Linear(Base):
    """
    Applies a linear transformation.
    """

    def __init__(self, inputs: int, outputs: int, bias: Optional[bool] = True):
        super().__init__()
        tensor = np.random.randn(inputs, outputs) * np.sqrt(1 / inputs)
        self.weights = self.create_element(tensor)
        self.isbias = bias
        if self.isbias:
            self.bias: Element = self.create_element(np.zeros(outputs))

    def forward(self, x: Array) -> Tuple[Array, Callable]:
        def backward(derivative: Array) -> Array:
            self.weights.gradient += x.T @ derivative
            if self.isbias:
                self.bias.gradient += derivative.sum(axis=0)
            return derivative @ self.weights.tensor.T

        if self.isbias:
            return x @ self.weights.tensor + self.bias.tensor, backward
        return x @ self.weights.tensor, backward


class Dropout(Base):
    """
    During training, randomly zeroes some of the elements of the input.
    """

    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x) -> Tuple[Array, Callable]:
        nodes = int(x.shape[1] * self.dropout_rate)
        indices = sorted(random.sample(range(0, x.shape[1]), nodes))
        mask = np.ones_like(x)
        mask[:, indices] = 0

        def backward(derivative: Array) -> Array:
            return derivative * mask

        return x * mask, backward


class LayerNorm(Base):
    """
    Applies Layer Normalization over a mini-batch of inputs
    """

    def __init__(self, shape: Tuple[int, ...], eps: Optional[float] = 1e-06):
        super().__init__()
        self.eps = eps
        self.weights: Element = self.create_element(np.ones(shape))
        self.bias: Element = self.create_element(np.zeros(shape))

    def forward(self, x: Array) -> Tuple[Array, Callable]:
        x_mean = x.mean(keepdims=True)
        x_var = np.var(x, keepdims=True)
        x_std = np.sqrt(x_var + self.eps)
        x_centered = x - x_mean
        x_norm = x_centered / x_std

        def backward(derivative: Array) -> Array:
            axis = derivative.shape[-1]
            self.weights.gradient += (derivative * x_norm).sum(axis=-1)
            self.bias.gradient += derivative.sum(axis=-1)
            dx_norm = derivative * self.weights.tensor
            return (
                1
                / axis
                / x_std
                * (
                    axis * dx_norm
                    - dx_norm.sum(axis=-1)
                    - x_norm * (dx_norm * x_norm).sum(axis=-1)
                )
            )

        return (x_norm * self.weights.tensor) + self.bias.tensor, backward


class Embedding(Base):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

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

    def forward(self, x: Array) -> Tuple[Array, Callable]:
        msg = "Error: Embeding input must int type"
        assert x.dtype == np.int64() or np.int32() or np.int16() or np.int8(), msg

        def backward(derivative: Array) -> Array:
            # trainable
            return derivative @ self.weights.tensor.T

        return self.weights.tensor[x], backward


class ReLU(Base):
    """
    Applies the rectified linear unit function element-wise.
    """

    def forward(self, x: Array) -> Tuple[Array, Callable]:
        def backward(derivative: Array) -> Array:
            return np.where(x > 0, derivative, 0)

        return x * (x > 0), backward


class Softmax(Base):
    """
    Applies the Softmax function to an n-dimensional input.
    """

    def forward(self, x: Array, dim: Optional[float] = 1) -> Tuple[Array, Callable]:
        y = np.exp(x) / np.exp(x).sum(axis=dim)[:, None]

        def backward(derivative: Array) -> Array:
            return y * (derivative - (derivative * y).sum(axis=1)[:, None])

        return y, backward


class Sigmoid(Base):
    """
    Applies the element-wise function
    """

    def forward(self, x: Array) -> Tuple[Array, Callable]:
        s = 1 / (1 + np.exp(-x))

        def backward(derivative: Array) -> Array:
            return derivative * s * (1 - s)

        return s, backward
