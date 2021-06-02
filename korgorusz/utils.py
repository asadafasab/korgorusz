"""
Mostly helper functions and classes that facilitate training.
"""

import os
import gzip
import pickle
from typing import Tuple, Callable, Optional, List, Iterator
import numpy as np
from korgorusz.optimizers import Optimizer
from korgorusz.layers import Base, Array


class Model:
    """
    Base class for creating model
    """

    def __init__(self):
        self.derivatives = []

    def backpropagation(self, deriv: Callable) -> None:
        """
        Inints a backpropagation and calculates all of the derivatives.
        """
        for func in reversed(self.derivatives):
            deriv = func(deriv)
        self.derivatives = []

    def update(self, layers: List[Base], optim: Optimizer) -> None:
        """
        Updates weights of elemnets.
        """
        elements = []
        for layer in layers:
            elements.extend(layer.elements)

        optim.update(elements)

    def add_derivative(self, func: Callable) -> None:
        """
        Adds the derivative function to derivatives list.
        """
        self.derivatives.append(func)

    def save(self, filename: str):
        """
        Pickles Model.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, filename: str):
        """
        Loads pickled Model.
        """
        with open(filename, "rb") as f:
            self.layers = pickle.load(f)


def mnist(path, names: List[str]) -> Tuple[Array, Array, Array, Array]:
    """
    Helper function for loading MNIST dataset.
    :path: path to the gzipped files
    :names: 4 names of the gzipped files
    :return: image data and labels
    """
    with gzip.open(os.path.join(path, names[0])) as f:
        x = np.frombuffer(f.read(), "B", offset=16)
        x = x.reshape(-1, 784).astype("float32")

    with gzip.open(os.path.join(path, names[1])) as f:
        y = np.frombuffer(f.read(), "B", offset=8)
        y = y.astype("uint8")

    with gzip.open(os.path.join(path, names[2])) as f:
        x_test = np.frombuffer(f.read(), "B", offset=16)
        x_test = x_test.reshape(-1, 784).astype("float32")

    with gzip.open(os.path.join(path, names[3])) as f:
        y_test = np.frombuffer(f.read(), "B", offset=8)
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
