import numpy as np
import pytest


def isclose(a, b, abs_tol=1e-6):
    return np.isclose(a, b, atol=abs_tol).all()


@pytest.fixture
def sample_array():
    return np.array([[10, 1, -1, 3, 0, 0, 9], [5, 6, 2, 3, -2, 7, 2]])


@pytest.fixture
def generate_linear_dataset(a=0.2, b=1, samples=8):
    return np.array([[x, (a * x) + b] for x in range(-samples, samples)]), a
