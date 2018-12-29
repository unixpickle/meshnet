"""
A neural network based on spring meshes.
"""

import torch
import torch.autograd as autograd
import torch.nn as nn


class MeshLayer(nn.Module):
    """
    A linear layer that uses a mesh-based matrix
    factorization.
    """

    def __init__(self, num_inputs, num_outputs, space_dims=2, init_delta=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.space_dims = space_dims

        init_in_pos = torch.randn((num_inputs, 1, space_dims))
        init_out_pos = torch.randn((1, num_outputs, space_dims))
        self.init_dists = _distance_matrix(
            self.init_in_pos + init_delta * torch.randn((num_inputs, 1, space_dims)),
            self.init_out_pos + init_delta * torch.randn((1, num_outputs, space_dims)),
        )
        self.in_pos = autograd.Variable(init_in_pos)
        self.out_pos = autograd.Variable(init_out_pos)
        self.biases = autograd.Variable(torch.zeros(num_outputs))

    def forward(self, inputs):
        dists = _distance_matrix(self.in_pos, self.out_pos)
        weights = dists - self.init_dists
        return torch.matmul(inputs, weights) + self.Biases


def _distance_matrix(in_pos, out_pos):
    return torch.sqrt(torch.sum(torch.square(in_pos - out_pos), dim=-1))
