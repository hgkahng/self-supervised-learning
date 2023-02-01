
import torch.nn as nn


def initialize_weights(model: nn.Module, activation: str = 'relu'):
    """Add function docstring."""
    for _, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=activation)
            if m.bias is not None:
                m.bias.data.fill_(0)
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0)
            nn.init.constant_(m.weight.data, 1)
            try:
                m.bias.data.fill_(0)
                nn.init.constant_(m.bias.data, 0)
            except AttributeError:
                pass
        elif isinstance(m, nn.Linear):
            m.weight.data.fill_(0)
            nn.init.normal_(m.weight.data, 0, 0.02)
            try:
                m.bias.data.fill_(0)
                nn.init.constant_(m.bias.data, 0)
            except AttributeError:
                pass
