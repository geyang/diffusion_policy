from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import trange

import numpy as np
import torch
from torch import nn


class LFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = mapping_size
        self.linear = nn.Linear(in_features, self.output_dim)
        nn.init.uniform_(self.linear.weight, - scale / self.input_dim, scale / self.input_dim)
        nn.init.uniform_(self.linear.bias, -1, 1)

    def forward(self, x):
        x = self.linear(np.pi * x)
        return torch.sin(x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale)
        self.linear.requires_grad = False


class View(nn.Module):
    def __init__(self, *dims, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dims = dims

    def forward(self, x):
        if self.batch_first:
            return x.view(-1, *self.dims)
        else:
            return x.view(*self.dims)


class MLP(nn.Sequential):
    def __init__(self, in_features, latent_features, latent_layers, out_features, preproc=None):
        self.in_features = in_features
        self.latent_features = latent_features
        self.latent_layers = latent_layers
        self.out_features = out_features

        assert latent_layers >= 1

        layers = [] if preproc is None else [preproc]
        for in_dim, out in zip([in_features] + [latent_features] * latent_layers,
                               [latent_features] * latent_layers + [out_features]):
            layers += [
                nn.Linear(in_dim, out),
                nn.ReLU()
            ]

        # exclude non-linearity in output layer
        super().__init__(*layers[:-1])


# @torch.no_grad()
class RFFMLP(nn.Sequential):
    def __init__(self, in_features, in_scale, latent_features, latent_layers, out_features):
        self.in_features = in_features
        self.latent_features = latent_features
        self.latent_layers = latent_layers
        self.out_features = out_features

        assert latent_layers >= 1

        layers = [
            RFF(in_features, latent_features, scale=in_scale)
        ]
        for in_dim, out in zip([latent_features] * latent_layers,
                               [latent_features] * (latent_layers - 1) + [out_features]):
            layers += [
                nn.Linear(in_dim, out),
                nn.ReLU()
            ]

        super().__init__(*layers[:-1])


class StackedLFF(nn.Sequential):
    def __init__(self, in_features, in_scale, latent_features, latent_scale, latent_layers, out_features):
        self.in_features = in_features
        self.latent_features = latent_features
        self.latent_layers = latent_layers
        self.out_features = out_features

        assert latent_layers >= 1

        layers = []
        for in_dim, scale, out in zip([in_features] + [latent_features] * latent_layers,
                                      [in_scale] + [latent_scale] * latent_layers,
                                      [latent_features] * latent_layers + [out_features]):
            layers += [
                LFF(in_dim, out, scale),
            ]

        super().__init__(*layers[:-1])
