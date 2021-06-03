"""
Contains Optimizers
"""

from typing import Dict, List

import numpy as np

from korgorusz.layers import Element


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
    """
    Implements stochastic gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, elements: List[Element]) -> None:
        for element in elements:
            element.tensor -= self.learning_rate * element.gradient
            element.gradient.fill(0)


class Momentum(Optimizer):
    """
    Implements stochastic gradient descent with momentum.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.5):
        self.learning_rate = learning_rate
        self.velocity: Dict = {}
        self.momentum = momentum

    def update(self, elements: List[Element]) -> None:
        for i, element in enumerate(elements):
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(element.gradient)

            self.velocity[i] = self.momentum * self.velocity[i] + element.gradient
            element.tensor -= self.learning_rate * self.velocity[i]

            element.gradient.fill(0)


class Adam(Optimizer):
    """
    Optimization algorithm good in a wide range of cases.
    learning_rate: learning rate
    beta1: coefficient used for computing
        running averages of gradient and its square
    beta2: coefficient used for computing
        running averages of gradient and its square
    eps: small positive number
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-07,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.time = 0
        self.before: Dict = {}

    def update(self, elements: List[Element]) -> None:
        for i, element in enumerate(elements):
            self.time += 1
            if i not in self.before:
                self.before[i] = {
                    "mean": np.zeros_like(element.gradient),
                    "var": np.zeros_like(element.gradient),
                }

            if np.linalg.norm(element.gradient) > np.inf:
                element.gradient = (
                    element.gradient * np.inf / np.linalg.norm(element.gradient)
                )

            mean = self.before[i]["mean"]
            var = self.before[i]["var"]

            mean = self.beta1 * mean + (1 - self.beta1) * element.gradient
            var = self.beta2 * var + (1 - self.beta2) * (element.gradient ** 2)
            self.before[i] = {"mean": mean, "var": var}

            var_corrected = var / (1 - self.beta2 ** self.time)
            mean_corrected = mean / (1 - self.beta1 ** self.time)
            update = (
                self.learning_rate
                * mean_corrected
                / (np.sqrt(var_corrected) + self.eps)
            )

            element.tensor -= update
            element.gradient.fill(0)
