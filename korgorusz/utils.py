import numpy as np
from typing import Tuple, Callable, Optional, List, Iterator
from korgorusz.optimizers import Optimizer
from korgorusz.layers import Base
import pickle
import gzip
import os


array = np.ndarray


class Model:
    """
    Base class for creating model
    """

    def __init__(self):
        self.derivatives = []

    def backpropagation(self, d: Callable) -> None:
        for func in reversed(self.derivatives):
            d = func(d)
        self.derivatives = []

    def update(self, layers: List[Base], optim: Optimizer) -> None:
        elements = []
        for l in layers:
            elements.extend(l.elements)

        optim.update(elements)

    def add_derivative(self, func: Callable) -> None:
        self.derivatives.append(func)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            self.layers = pickle.load(f)


def mnist(path, names: List[str]) -> Tuple[array, array, array, array]:
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


def normalize(x, l2=True, axis=1):
    if l2:
        norm = np.linalg.norm(x, axis=axis)
    else:
        norm = np.linalg.norm(x, axis=axis, ord=1)
    return x / norm


def mse(x: array, y: array) -> Tuple[float, array]:
    """
    Mean squared error loss.
    :return: loss and derivative
    """
    return ((x - y) ** 2).mean(), (x - y) / 1.5


def cross_entropy(x: array, y: array, eps: float = 1e-8) -> Tuple[float, array]:
    """
    Cross entropy loss.
    :return: loss and derivative
    """
    x = x.clip(min=eps, max=None)
    cost = (np.where(y == 1, -np.log(x), 0)).sum(axis=1)
    d = np.where(y == 1, -1 / x, 0)
    return cost.sum(), d


def one_hot(a: array, num_classes: int) -> array:
    """
    :return: one hot vector :)
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def minibatch(x: array, y: array, batch_size: int) -> Iterator[Tuple[array, array]]:
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i : min(i + batch_size, x.shape[0]), :]
        y_batch = y[i : min(i + batch_size, y.shape[0]), :]
        yield x_batch, y_batch
