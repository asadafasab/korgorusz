"""
Mostly helper functions and classes that facilitate training.
"""

import gzip
import os
import pickle
from typing import Iterator, List, Tuple

import numpy as np

from korgorusz.layers import Array, BaseLayer


class Model:
    """Helper class for management of layers"""

    def __init__(self, layers: List[BaseLayer]):
        self.layers = layers

    def forward(self, x: Array) -> Array:
        """Feed-forward of all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, derivative: Array) -> None:
        """Backpropagation of error"""
        x = derivative
        for layer in reversed(self.layers):
            x = layer.backward(x)

    def layers_elements(self):
        """Returns all elements of layers"""
        elements = []
        for layer in self.layers:
            elements.extend(layer.elements)
        return elements

    def save(self, filename: str):
        """
        Pickles Model.
        """
        with open(filename, "wb") as file:
            pickle.dump(self.layers, file)

    def load(self, filename: str):
        """
        Loads pickled Model.
        """
        with open(filename, "rb") as file:
            self.layers = pickle.load(file)


def mnist(path, names: List[str]) -> Tuple[Array, Array, Array, Array]:
    """
    Helper function for loading MNIST dataset.
    :path: path to the gzipped files
    :names: 4 names of the gzipped files
    :return: image data and labels
    """
    with gzip.open(os.path.join(path, names[0])) as file:
        x = np.frombuffer(file.read(), "B", offset=16)
        x = x.reshape(-1, 784).astype("float32")

    with gzip.open(os.path.join(path, names[1])) as file:
        y = np.frombuffer(file.read(), "B", offset=8)
        y = y.astype("uint8")

    with gzip.open(os.path.join(path, names[2])) as file:
        x_test = np.frombuffer(file.read(), "B", offset=16)
        x_test = x_test.reshape(-1, 784).astype("float32")

    with gzip.open(os.path.join(path, names[3])) as file:
        y_test = np.frombuffer(file.read(), "B", offset=8)
        y_test = y_test.astype("uint8")

    return x, y, x_test, y_test


def normalize(values, l2=True, axis=1):
    """
    Squeezes values into a 0 to 1 range.
    l1(Least Absolute Deviations) - sum of the absolute values will always be up to 1
    l2(least squares) - each row the sum of the squares will always be up to 1
    """
    if l2:
        norm = np.linalg.norm(values, axis=axis)
    else:
        norm = np.linalg.norm(values, axis=axis, ord=1)
    return values / norm


def mse(x: Array, y: Array) -> Tuple[float, Array]:
    """
    Mean squared error loss.
    :return: loss and derivative
    """
    return ((x - y) ** 2).mean(), (x - y) / 1.5


def cross_entropy(x: Array, y: Array, eps: float = 1e-8) -> Tuple[float, Array]:
    """
    Cross entropy loss.
    :return: loss and derivative
    """
    x = x.clip(min=eps, max=None)
    cost = (np.where(y == 1, -np.log(x), 0)).sum(axis=1)
    deriv = np.where(y == 1, -1 / x, 0)
    return cost.sum(), deriv


def one_hot(array: Array, num_classes: int) -> Array:
    """
    :return: one hot vector :)
    """
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])


def minibatch(x: Array, y: Array, batch_size: int) -> Iterator[Tuple[Array, Array]]:
    """
    Divides dataset into parts.
    """
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i : min(i + batch_size, x.shape[0]), :]
        y_batch = y[i : min(i + batch_size, y.shape[0]), :]
        yield x_batch, y_batch
