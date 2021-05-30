import pytest
import numpy as np

from korgorusz.optimizers import SGDOptimizer, Momentum, Adam
from korgorusz.tests.utils_for_test import isclose, generate_linear_dataset
from korgorusz.layers import Linear, ReLU, Sigmoid
from korgorusz.utils import mse, normalize


np.random.seed(42)


def optimizer_setup(generate_linear_dataset, optimizer, lr, epochs):
    points, a_ = generate_linear_dataset
    s = points.shape
    points = normalize(points.reshape(1, -1)).reshape(s)
    correct = np.array([a_] * points.shape[0])
    optim = optimizer(lr=lr)
    l1 = Linear(2, 4)
    r = ReLU()
    l2 = Linear(4, 1)
    s = Sigmoid()

    for _ in range(epochs):
        a, d0 = l1.forward(points)
        a, d1 = r.forward(a)
        a, d2 = l2.forward(a)
        a, d3 = s.forward(a)
        _, loss = mse(a, a_)
        d0(d1(d2(d3(loss))))
        elements = []
        elements.extend(l1.elements)
        elements.extend(l2.elements)
        optim.update(elements)

    out, _ = l1.forward(points)
    out, _ = r.forward(out)
    out, _ = l2.forward(out)
    out, _ = s.forward(out)
    return out, correct


def test_SGD(generate_linear_dataset):
    a, correct = optimizer_setup(generate_linear_dataset, SGDOptimizer, 0.3, 16)
    assert isclose(correct, a, abs_tol=0.01)


def test_momentum(generate_linear_dataset):
    a, correct = optimizer_setup(generate_linear_dataset, Momentum, 0.3, 16)
    assert isclose(a, correct, abs_tol=0.01)


def test_adam(generate_linear_dataset):
    a, correct = optimizer_setup(generate_linear_dataset, Adam, 0.2, 16)
    assert isclose(a, correct, abs_tol=0.1)
