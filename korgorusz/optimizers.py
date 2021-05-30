from typing import Tuple, Callable, Optional, List, Dict
from korgorusz.layers import Element
import numpy as np


class Optimizer:
    """
    Base class for creating new optimizer
    """

    def update(self, elements: List[Element]) -> None:
        """
        Updates tensor of Element(s)
        :elements: list of all elements in the network
        """
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, elements: List[Element]) -> None:
        for element in elements:
            element.tensor -= self.lr * element.gradient
            element.gradient.fill(0)


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.5):
        self.lr = lr
        self.velocity: Dict = {}
        self.momentum = momentum

    def update(self, elements: List[Element]) -> None:
        for i in range(len(elements)):
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(elements[i].gradient)

            self.velocity[i] = self.momentum * self.velocity[i] + elements[i].gradient
            elements[i].tensor -= self.lr * self.velocity[i]

            elements[i].gradient.fill(0)


class Adam(Optimizer):
    """
    Optimization algorithm good in a wide range of cases.
    lr: learning rate
    beta1: coefficient used for computing
        running averages of gradient and its square
    beta2: coefficient used for computing
        running averages of gradient and its square
    eps: small positive number
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-07,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.before: Dict = {}

    def update(self, elements: List[Element]) -> None:
        for i in range(len(elements)):
            self.t += 1
            if i not in self.before:
                self.before[i] = {
                    "mean": np.zeros_like(elements[i].gradient),
                    "var": np.zeros_like(elements[i].gradient),
                }

            if np.linalg.norm(elements[i].gradient) > np.inf:
                elements[i].gradient = (
                    elements[i].gradient * np.inf / np.linalg.norm(elements[i].gradient)
                )

            mean = self.before[i]["mean"]
            var = self.before[i]["var"]

            mean = self.beta1 * mean + (1 - self.beta1) * elements[i].gradient
            var = self.beta2 * var + (1 - self.beta2) * (elements[i].gradient ** 2)
            self.before[i] = {"mean": mean, "var": var}

            var_corrected = var / (1 - self.beta2 ** self.t)
            mean_corrected = mean / (1 - self.beta1 ** self.t)
            update = self.lr * mean_corrected / (np.sqrt(var_corrected) + self.eps)

            elements[i].tensor -= update
            elements[i].gradient.fill(0)
