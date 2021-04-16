from typing import Tuple, Callable, Optional, List
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
    def __init__(self, lr: float = 0.01, momentum=0.5):
        self.lr = lr
        self.velocity = [0]
        self.momentum = momentum

    def update(self, elements: List[Element]) -> None:
        if len(self.velocity) == 1:
            self.velocity *= len(elements)

        for i in range(len(elements)):
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
        beta2: float = 0.99,
        eps: float = 1e-08,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m_d, self.v_d = [0], [0]

        msg = "There is something wrong with this..."
        raise ValueError(msg)  # TODO

    def update(self, elements: List[Element]) -> None:
        if len(self.m_d) == 1:
            self.m_d *= len(elements)
            self.v_d *= len(elements)

        for i in range(len(elements)):
            self.m_d[i] = (
                self.beta1 * self.m_d[i] + (1 - self.beta1) * elements[i].gradient
            )
            self.v_d[i] = self.beta2 * self.v_d[i] + (1 - self.beta1) * (
                elements[i].gradient ** 2
            )

            m_d_corr = self.m_d[i] / (1 - self.beta1 ** self.t)
            v_d_corr = self.v_d[i] / (1 - self.beta2 ** self.t)

            update = m_d_corr / (np.square(v_d_corr) + self.eps)

            elements[i].tensor -= self.lr * update
            elements[i].gradient.fill(0)
