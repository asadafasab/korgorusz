import pytest
import numpy as np

from korgorusz.optimizers import *
from korgorusz.tests.test_utils import isclose, generate_linear_dataset
from korgorusz.layers import Linear
from korgorusz.utils import mse, normalize


np.random.seed(42)


def optimizer_setup(generate_linear_dataset, optimizer, lr):
    points, a_ = generate_linear_dataset
    s = points.shape
    points = normalize(points.reshape(1, -1)).reshape(s)
    correct = np.array([a_] * points.shape[0])
    optim = optimizer(lr=lr)
    l1 = Linear(2, 1)
    for _ in range(16):
        a, d = l1.forward(points)
        loss_mean, loss = mse(a, a_)
        d(loss)
        optim.update(l1.elements)

    return l1.forward(points)[0], correct


def test_SGD(generate_linear_dataset):
    a, correct = optimizer_setup(generate_linear_dataset, SGDOptimizer, 0.7)
    assert isclose(correct, a, abs_tol=0.05)


def test_momentum(generate_linear_dataset):
    a, correct = optimizer_setup(generate_linear_dataset, Momentum, 0.7)
    assert isclose(a, correct, abs_tol=0.05)
