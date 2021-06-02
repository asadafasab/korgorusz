# pylint: disable=redefined-outer-name,unused-import
"""
Tests the layers.
"""
import numpy as np
from korgorusz.layers import (
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Dropout,
    LayerNorm,
    Embedding,
)
from korgorusz.tests.utils_for_test import sample_array, isclose, rows_array


np.random.seed(42)


def test_relu(sample_array):
    """
    Test a ReLU function
    """
    correct = np.array([[10, 1, 0, 3, 0, 0, 9], [5, 6, 2, 3, 0, 7, 2]])
    activation = ReLU()
    out, _ = activation.forward(sample_array)
    assert (correct == out).all()


def test_softmax(sample_array):
    """
    Test a Softmax function
    """
    correct = np.array(
        [
            [
                7.3044837e-01,
                9.0144487e-05,
                1.2199730e-05,
                6.6608272e-04,
                3.3162305e-05,
                3.3162305e-05,
                2.6871693e-01,
            ],
            [
                8.8158868e-02,
                2.3964065e-01,
                4.3891715e-03,
                1.1931006e-02,
                8.0390477e-05,
                6.5141082e-01,
                4.3891715e-03,
            ],
        ]
    )

    activation = Softmax()
    out, _ = activation.forward(sample_array)
    assert isclose(out, correct)


def test_sigmoid(sample_array):
    """
    Test a Sigmoid function
    """
    correct = np.array(
        [
            [
                0.9999546,
                0.7310586,
                0.2689414,
                0.9525741,
                0.5000000,
                0.5000000,
                0.9998766,
            ],
            [
                0.9933072,
                0.9975274,
                0.8807970,
                0.9525741,
                0.1192029,
                0.9990890,
                0.8807970,
            ],
        ]
    )

    activation = Sigmoid()
    out, _ = activation.forward(sample_array)
    assert isclose(out, correct)


def test_linear(sample_array):
    """
    Test a Linear layer
    """
    rand = np.random.randn(64, 20)
    lin = Linear(20, 30)
    assert lin.forward(rand)[0].shape == (64, 30)
    correct = np.array(
        [
            [0.2300000, 0.2300000, 0.2300000, 0.2300000],
            [0.2400000, 0.2400000, 0.2400000, 0.2400000],
        ]
    )

    lin = Linear(7, 4)
    lin.weights.tensor.fill(0.01)
    lin.bias.tensor.fill(0.01)
    out, _ = lin.forward(sample_array)
    assert isclose(out, correct)


def test_dropout(sample_array):
    """
    Test if dropout is working
    """
    sample_array += 1
    drop = Dropout(0.5)
    out, _ = drop.forward(sample_array)

    assert np.count_nonzero(out == 0) >= 3


def test_layer_norm(sample_array):
    """
    Test LayerNorm layer.
    """
    correct = np.array(
        [
            [
                1.9011403,
                -0.6203722,
                -1.1807083,
                -0.0600361,
                -0.9005402,
                -0.9005402,
                1.6209723,
            ],
            [
                0.5003000,
                0.7804681,
                -0.3402041,
                -0.0600361,
                -1.4608763,
                1.0606362,
                -0.3402041,
            ],
        ]
    )
    norm = LayerNorm(7)
    out, _ = norm.forward(sample_array)
    assert isclose(out, correct)


def test_embeding(rows_array):
    """
    Tests an embedding.
    """
    emb = Embedding(6, 4)
    out, _ = emb.forward(rows_array)
    assert out.shape == (5, 4)
