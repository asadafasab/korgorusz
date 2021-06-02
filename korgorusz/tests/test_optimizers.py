"""
Tests if opimizers updates elements of layers.
"""

import pytest
import numpy as np

from korgorusz.optimizers import SGDOptimizer, Momentum, Adam
from korgorusz.tests.utils_for_test import isclose, generate_linear_dataset
from korgorusz.layers import Linear, ReLU, Sigmoid
from korgorusz.utils import mse, normalize


np.random.seed(42)


def optimizer_setup(dataset, optimizer, learning_rate, epochs):
    """
    Setups dataset, neural network,...
    """
    points, out_ = dataset
    shape = points.shape
    points = normalize(points.reshape(1, -1)).reshape(shape)
    correct = np.array([out_] * points.shape[0])
    optim = optimizer(learning_rate=learning_rate)
    fc1 = Linear(2, 4)
    activation1 = ReLU()
    fc2 = Linear(4, 1)
    activation2 = Sigmoid()

    for _ in range(epochs):
        out, deriv0 = fc1.forward(points)
        out, deriv1 = activation1.forward(out)
        out, deriv2 = fc2.forward(out)
        out, deriv3 = activation2.forward(out)
        _, loss = mse(out, out_)
        deriv0(deriv1(deriv2(deriv3(loss))))
        elements = []
        elements.extend(fc1.elements)
        elements.extend(fc2.elements)
        optim.update(elements)

    out, _ = fc1.forward(points)
    out, _ = activation1.forward(out)
    out, _ = fc2.forward(out)
    out, _ = activation2.forward(out)
    return out, correct


def test_sgd(generate_linear_dataset):
    """
    Tests SGD optimizer
    """
    out, correct = optimizer_setup(generate_linear_dataset, SGDOptimizer, 0.3, 16)
    assert isclose(correct, out, abs_tol=0.01)


def test_momentum(generate_linear_dataset):
    """
    Tests Momentum optimizer
    """
    out, correct = optimizer_setup(generate_linear_dataset, Momentum, 0.3, 16)
    assert isclose(out, correct, abs_tol=0.01)


def test_adam(generate_linear_dataset):
    """
    Tests Adam optimizer
    """
    out, correct = optimizer_setup(generate_linear_dataset, Adam, 0.2, 16)
    assert isclose(out, correct, abs_tol=0.1)
