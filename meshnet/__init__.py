"""
A neural network based on spring meshes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshFC(nn.Module):
    """
    A linear layer that uses a mesh-based matrix
    factorization.
    """

    def __init__(self, num_inputs, num_outputs, space_dims=5, init_delta=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.space_dims = space_dims

        self.init_in_pos = nn.Parameter(torch.randn((num_inputs, 1, space_dims)))
        self.init_out_pos = nn.Parameter(torch.randn((1, num_outputs, space_dims)))
        self.in_pos = nn.Parameter(self.init_in_pos + init_delta *
                                   torch.randn((num_inputs, 1, space_dims)))
        self.out_pos = nn.Parameter(self.init_out_pos + init_delta *
                                    torch.randn((1, num_outputs, space_dims)))
        self.biases = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, inputs):
        return torch.matmul(inputs, self.weight_matrix()) + self.biases

    def weight_matrix(self):
        dists = _distance_matrix(self.in_pos, self.out_pos)
        init_dists = _distance_matrix(self.init_in_pos, self.init_out_pos)
        return dists - init_dists


class MeshConv2d(MeshFC):
    """
    A convolutional layer that uses mesh-based matrix
    factorization.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 space_dims=10,
                 init_delta=0.01,
                 **conv_kwargs):
        if isinstance(kernel_size, tuple):
            in_dims = kernel_size[0] * kernel_size[1] * in_channels
            in_shape = (in_channels,) + kernel_size
        else:
            in_dims = (kernel_size ** 2) * in_channels
            in_shape = (in_channels, kernel_size, kernel_size)
        super().__init__(in_dims, out_channels, space_dims=space_dims, init_delta=init_delta)
        self.filter_shape = (out_channels,) + in_shape
        self.conv_kwargs = conv_kwargs

    def forward(self, inputs):
        weights = torch.t(self.weight_matrix()).view(*self.filter_shape)
        return F.conv2d(inputs, weights, bias=self.biases)


def _distance_matrix(in_pos, out_pos):
    return torch.sqrt(torch.sum((in_pos - out_pos) ** 2, dim=-1))
