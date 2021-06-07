"""
Contains a set of helper functions for testing.
"""

import numpy as np
import pytest


def isclose(array1, array2, abs_tol=1e-6):
    """
    Checks if arrays have similar values.
    """
    return np.isclose(array1, array2, atol=abs_tol).all()


@pytest.fixture
def sample_array():
    """
    Simple example array
    """
    return np.array([[10, 1, -1, 3, 0, 0, 9], [5, 6, 2, 3, -2, 7, 2]])


@pytest.fixture
def rows_array():
    """
    Array for embedding test
    """
    return np.array([2, 5, 2, 0, 1])


@pytest.fixture
def loss_arrays():
    """2 arrays(prediction and answer(y)) for loss funcions"""
    return np.array(
        [
            [0.47342047, 0.65926937, 0.89619478],
            [0.10885469, -0.91478403, 0.67732042],
            [-1.56513146, 0.61440365, 0.17894966],
            [-0.40858872, -0.99496016, 0.30105507],
        ]
    ), np.array(
        [
            [0.07424372, 0.31392202, 1.97777693],
            [-0.7008325, 1.4005098, 0.25964299],
            [-0.84083587, 0.45322056, -0.96235116],
            [2.38246607, -0.57225619, 0.36363486],
        ]
    )


@pytest.fixture
def generate_linear_dataset(slope=0.2, bias=1, samples=16):
    """
    Generates simple linear function dataset
    """
    return (
        np.array(
            [
                [independent, (slope * independent) + bias]
                for independent in range(-samples, samples)
            ]
        ),
        slope,
    )
