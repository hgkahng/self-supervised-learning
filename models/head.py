# -*- coding: utf-8 -*-

import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import GAPHeadBase
from layers.core import Flatten
from utils.initialization import initialize_weights


class HeadBase(nn.Module):
    def __init__(self, output_size: int):
        super(HeadBase, self).__init__()
        assert isinstance(output_size, int)

    def save_weights(self, path: str):
        """Save weights to a file with weights only."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load weights from a file with weights only."""
        self.load_state_dict(torch.load(path))

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])


class GAPHeadBase(HeadBase):
    def __init__(self, in_channels: int, output_size: int):
        super(GAPHeadBase, self).__init__(output_size)
        assert isinstance(in_channels, int), "Number of output feature maps of backbone."


class LinearHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout: float = 0.0):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output features.
        """
        super(LinearHead, self).__init__(in_channels, num_features)

        self.in_channels = in_channels
        self.num_features = num_features
        self.dropout = dropout
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features,
            dropout=self.dropout,
        )
        initialize_weights(self.layers)

    @staticmethod
    def make_layers(in_channels: int, num_features: int, dropout: float = 0.0):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('dropout', nn.Dropout(p=dropout)),
                    ('linear', nn.Linear(in_channels, num_features))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifier(LinearHead):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_classes: int, number of classes.
        """
        super(LinearClassifier, self).__init__(in_channels, num_classes, dropout)

    @property
    def num_classes(self):
        return self.num_features


class MLPHead(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int):
        """
        Arguments:
            in_channels: int, number of input feature maps.
            num_features: int, number of output units.
        """
        super(MLPHead, self).__init__(in_channels, num_features)

        self.in_channels = in_channels
        self.num_features = num_features
        self.layers = self.make_layers(
            in_channels=self.in_channels,
            num_features=self.num_features,
        )

    @staticmethod
    def make_layers(in_channels: int, num_features: int):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear1', nn.Linear(in_channels, in_channels, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(in_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('linear2', nn.Linear(in_channels, num_features, bias=True))
                ]
            )
        )

        return layers

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
