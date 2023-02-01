# -*- coding: utf-8 -*-

import math
import collections
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)

class FCHeadBase(HeadBase):
    def __init__(self, input_size: int, output_size: int):
        super(FCHeadBase, self).__init__(output_size)
        assert isinstance(input_size, int), "Number of input units."


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
            dropout: float, dropout ratio in the range [0, 1].
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
    def make_layers(in_channels: int, num_features: int, dropout: float = 0.0) -> nn.Module:
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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


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
    def make_layers(in_channels: int, num_features: int) -> nn.Module:
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear1', nn.Linear(in_channels, in_channels, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(in_channels)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('linear2', nn.Linear(in_channels, num_features, bias=True))
                ]
            )
        )

        return layers

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class MLPHeadWithMCDropout(GAPHeadBase):
    def __init__(self, in_channels: int, num_features: int, dropout_rate: float = 0.2):
        super(MLPHeadWithMCDropout, self).__init__(in_channels, num_features)
        
        self.in_channels: int = in_channels
        self.num_features: int = num_features
        self.dropout_rate: float = dropout_rate
        
        self.layers = nn.ModuleDict(
            {
                'gap': nn.AdaptiveAvgPool2d(1),
                'flatten': Flatten(),
                'linear1': nn.Linear(self.in_channels, self.in_channels, bias=False),
                'bnorm1': nn.BatchNorm1d(self.in_channels),
                'relu1': nn.ReLU(inplace=False),
                'linear2': nn.Linear(in_channels, num_features, bias=True),
            }
        )
        
    def forward(self, x: torch.FloatTensor, dropout: bool = False) -> torch.FloatTensor:
        
        y = self.layers['flatten'](self.layers['gap'](x))
        if dropout:
            y = F.dropout(y, p=self.dropout_rate, training=True)
        y = self.layers['linear1'](y)
        y = self.layers['bnorm1'](y)
        y = self.layers['relu1'](y)
        if dropout:
            y = F.dropout(y, p=self.dropout_rate, training=True)
        
        return self.layers['linear2'](y)

    @property
    def num_parameters(self) -> int:
        cnt: int = 0
        for _, layer in self.layers.items():
            cnt += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        return cnt
        
        

class BYOLProjectionHead(GAPHeadBase):
    def __init__(self, in_channels: int, hidden_size: int = 4096, output_size: int = 256):
        """
        3-layer MLP with one hidden layer, includes no batch normalization in output.
        Arguments:
            in_channels: int, number of feature maps (or channels) in input.
            hidden_size: int, number of units in hidden layer.
            output_size: int, number of units in output layer. 
        """
        super(BYOLProjectionHead, self).__init__(in_channels, output_size)

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = self.make_layers(in_channels=self.in_channels,
                                       hidden_size=self.hidden_size,
                                       output_size=self.output_size)
    
    @staticmethod
    def make_layers(in_channels: int, hidden_size: int, output_size: int):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('linear1', nn.Linear(in_channels, hidden_size, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(hidden_size)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('linear2', nn.Linear(hidden_size, output_size, bias=True)),
                ]
            )
        )

        return layers

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class BYOLPredictionHead(FCHeadBase):
    def __init__(self, input_size: int = 256, hidden_size: int = 4096, output_size: int = 256):
        """
        3-layer MLP with one hidden layer, includes no batch normalization in output.
        Arguments:
            input_size: int, number of units in input layer.
            hidden_size: int, number of units in hidden layer.
            output_size: int, number of units in output layer. 
        """
        super(BYOLPredictionHead, self).__init__(input_size, output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = self.make_layers(input_size=self.input_size,
                                       hidden_size=self.hidden_size,
                                       output_size=self.output_size)
    
    @staticmethod
    def make_layers(input_size: int, hidden_size: int, output_size: int):
        layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear1', nn.Linear(input_size, hidden_size, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(hidden_size)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('linear2', nn.Linear(hidden_size, output_size, bias=True)),
                ]
            )
        )

        return layers

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

class CLAPv2PredHead(FCHeadBase):
    def __init__(self,
                 input_size: int = 128,
                 hidden_size: int = 512,
                 output_size: int = 128):
        super(CLAPv2PredHead, self).__init__(input_size, output_size)
        
        if input_size != output_size:
            warnings.warn(f"`input_size'={input_size} and `output_size'={output_size}"
                          "mismatch may cause errors.")

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.layers = self.make_layers(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    @staticmethod
    def make_layers(input_size: int, hidden_size: int, output_size: int):
        layers = nn.Sequential(
            collections.OrderedDict(
                [   
                    ('bnorm0', nn.BatchNorm1d(input_size)),
                    ('linear1', nn.Linear(input_size, hidden_size, bias=False)),
                    ('bnorm1', nn.BatchNorm1d(hidden_size)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('linear2', nn.Linear(hidden_size, hidden_size, bias=False)),
                    ('bnorm2', nn.BatchNorm1d(hidden_size)),
                    ('relu2', nn.ReLU(inplace=False)),
                    ('linear3', nn.Linear(hidden_size, output_size, bias=True)),
                ]
            )
        )

        return layers
    
    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
    