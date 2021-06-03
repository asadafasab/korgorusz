# pylint: disable=redefined-outer-name,too-many-locals,unused-import
"""
Tests if opimizers updates elements of layers.
"""

import numpy as np

from korgorusz.optimizers import SGDOptimizer, Momentum, Adam
from korgorusz.tests.utils_for_test import isclose, generate_linear_dataset
from korgorusz.layers import Linear
from korgorusz.activations import ReLU, Sigmoid
from korgorusz.utils import mse, normalize, Model


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
    model = Model([Linear(2, 4), ReLU(), Linear(4, 1), Sigmoid()])
    for _ in range(epochs):
        out = model.forward(points)
        losss, loss = mse(out, out_)
        print(losss)
        model.backward(loss)
        optim.update(model.layers_elements())

    return model.forward(points), correct


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
    out, correct = optimizer_setup(generate_linear_dataset, Adam, 0.1, 6)
    assert isclose(out, correct, abs_tol=0.1)
