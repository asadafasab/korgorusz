"""Contains activation functions e.g. ReLU,Sigmoid."""

from typing import Optional
import numpy as np

from korgorusz.layers import BaseLayer, Array


class ReLU(BaseLayer):
    """
    Applies the rectified linear unit function element-wise.
    """

    def forward(self, x: Array) -> Array:
        def backward(derivative: Array) -> Array:
            return np.where(x > 0, derivative, 0)

        self.backward = backward
        return x * (x > 0)


class Softmax(BaseLayer):
    """
    Applies the Softmax function to an n-dimensional input.
    """

    def __init__(self, dim: Optional[float] = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Array) -> Array:
        y = np.exp(x) / np.exp(x).sum(axis=self.dim)[:, None]

        def backward(derivative: Array) -> Array:
            return y * (derivative - (derivative * y).sum(axis=1)[:, None])

        self.backward = backward
        return y


class Sigmoid(BaseLayer):
    """
    Applies the element-wise function
    """

    def forward(self, x: Array) -> Array:
        sig = 1 / (1 + np.exp(-x))

        def backward(derivative: Array) -> Array:
            return derivative * sig * (1 - sig)

        self.backward = backward
        return sig
