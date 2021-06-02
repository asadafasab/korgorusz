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
