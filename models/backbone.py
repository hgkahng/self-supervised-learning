# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101
from utils.initialization import initialize_weights


RESNET_FUNCTIONS = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
}


class BackboneBase(nn.Module):
    def __init__(self, in_channels: int):
        super(BackboneBase, self).__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def save_weights_to_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights_from_checkpoint(self, path: str, key: str):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetBackbone(BackboneBase):
    def __init__(self, name: str = 'resnet50', data: str = 'imagenet', in_channels: int = 3):
        super(ResNetBackbone, self).__init__(in_channels)

        self.name = name
        self.data = data

        self.layers = RESNET_FUNCTIONS[self.name](pretrained=False)
        self.layers = self._remove_gap_and_fc(self.layers)
        if self.in_channels != 3:
            self.layers = self._fix_first_conv_in_channels(self.layers, in_channels=self.in_channels)
        if not self.data.startswith('imagenet'):
            self.layers = self._fix_first_conv_kernel_size(self.layers)
        if self.data.startswith('cifar'):
            self.layers = self._remove_maxpool(self.layers)

        initialize_weights(self)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    @staticmethod
    def _fix_first_conv_in_channels(resnet: nn.Module, in_channels: int) -> nn.Module:
        """
        Change the number of incoming channels for the first layer.
        """
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                model.add_module(name, conv1)
            else:
                model.add_module(name, child)

        return model

    @staticmethod
    def _remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        """
        Remove global average pooling & fully-connected layer
        For torchvision ResNet models only."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)  # preserve original names

        return model

    @staticmethod
    def _fix_first_conv_kernel_size(resnet: nn.Module) -> nn.Module:
        """Fix first conv layer of ResNet. (7x7 -> 3x3)"""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.add_module(name, conv1)
            else:
                model.add_module(name, child)

        return model

    @staticmethod
    def _remove_maxpool(resnet: nn.Module):
        """Remove first max pooling layer of ResNet."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'maxpool':
                continue
            else:
                model.add_module(name, child)

        return model

    @property
    def out_channels(self):
        if self.name == 'resnet18':
            return 512
        elif self.name == 'resnet50':
            return 2048
        else:
            raise NotImplementedError


class WideResNet(nn.Module):
    def __init__(self,  depth: int, widen_factor: int):  # pylint: disable=unused-argument
        super(WideResNet, self).__init__()
    
    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError
