import pytest
import numpy as np
from korgorusz.layers import *


@pytest.fixture
def sample_array():
    return np.array([[10, 1, -1, 3, 0, 0, 9],
                     [5, 6, 2, 3, -2, 7, 2]])


def test_relu(sample_array):
    correct = np.array([[10, 1, 0, 3, 0, 0, 9],
                        [5, 6, 2, 3, 0, 7, 2]])
    activation = ReLU()
    a, _ = activation.forward(sample_array)
    print(a)
    assert (correct == a).all()
