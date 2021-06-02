"""
Tests a derivatives of a layers, activations.
"""

import pytest
import numpy as np
from korgorusz.layers import Linear
from korgorusz.utils import mse

from korgorusz.tests.utils_for_test import isclose


# pytorch code
_ = """
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_printoptions(precision=7)

lin = nn.Linear(5,1)
# lin.weight = torch.nn.Parameter(torch.tensor([[-0.5057,  0.0027, -0.1124]], requires_grad=True))
# lin.bias = torch.nn.Parameter(torch.tensor([0.3479], requires_grad=True))
print(lin.weight)
print(lin.bias)
print("\n\n")
arr = torch.tensor([[1.0, 2, 3,2,1], [1,0,2, 1, 3], [3,3,3, 2, 1]])
target = torch.tensor([[0.5,0,1], [1,0,0], [1,1,0]])
target = torch.tensor([[0.5], [1], [1]])
optim = torch.optim.SGD(lin.parameters(), lr=1e-2)
loss = nn.MSELoss()
out = lin(arr)

output_loss = loss(out,target)
print("loss: ",output_loss,"\n\n")
print(target,out)
output_loss.backward()
print("parameters:")
for p in lin.parameters():
    print(p.grad)
"""


def test_linear_backward():
    """
    Tests a derivative of a Linear layer.
    """
    linear = Linear(5, 1)
    linear.weights.tensor = np.array(
        [[-0.4416076], [-0.2613170], [0.2723470], [0.4146298], [0.3575290]]
    )
    linear.bias.tensor = np.array([0.4020894])

    arr = np.array([[1.0, 2, 3, 2, 1], [1, 0, 2, 1, 3], [3, 3, 3, 2, 1]])
    target = np.array([[0.5, 0, 1], [1, 0, 0], [1, 1, 0]])
    target = np.array([[0.5], [1], [1]])
    out, back = linear.forward(arr)
    loss = mse(out, target)

    back(loss[1])

    correct_weight_grad = np.array(
        [[-0.1163296], [-0.1501397], [1.8008353], [0.9800253], [2.1440001]]
    )
    correct_bias_grad = np.array([0.8208102])

    assert isclose(correct_weight_grad, linear.weights.gradient)
    assert isclose(correct_bias_grad, linear.bias.gradient)
