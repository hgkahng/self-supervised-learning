# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn


class SplitBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, num_splits: int, **kwargs):
        super(SplitBatchNorm2d, self).__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, input):  # pylint: disable=redefined-builtin
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean=running_mean_split,
                running_var=running_var_split,
                weight=self.weight.repeat(self.num_splits),
                bias=self.bias.repeat(self.num_splits),
                training=True,
                momentum=self.momentum,
                eps=self.eps
            )
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome.view(N, C, H, W)
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps
            )

    @classmethod
    def revert_batchnorm(cls, module: nn.Module):
        """Helper function to revert `SplitBatchNorm2d` layers in the module to
        `nn.BatchNorm2d` layers. Works recursively.
        """

        module_output = copy.deepcopy(module)
        if isinstance(module, cls):
            module_output = nn.BatchNorm2d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight.data.copy_(module.weight.data)
                    module_output.bias.data.copy_(module.bias.data)
            module_output.running_mean = module.running_mean                # pylint: disable=attribute-defined-outside-init
            module_output.running_var = module.running_var                  # pylint: disable=attribute-defined-outside-init
            module_output.num_batches_tracked = module.num_batches_tracked  # pylint: disable=attribute-defined-outside-init

        for name, child in module.named_children():
            module_output.add_module(name, cls.revert_batchnorm(child))
        del module

        return module_output

    @classmethod
    def convert_split_batchnorm(cls, module: nn.Module, num_splits: int = 8):
        """Helper function to convert `nn.BatchNorm2d` layers in the module to
        `SplitBatchNorm2d` layers. Works recursively.
        """

        module_output = copy.deepcopy(module)  # avoid inplace replacements.
        if isinstance(module, nn.BatchNorm2d):
            module_output = cls(
                num_features=module.num_features,
                num_splits=num_splits,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight.data.copy_(module.weight.data)
                    module_output.bias.data.copy_(module.bias.data)
            module_output.running_mean = module.running_mean                # pylint: disable=attribute-defined-outside-init
            module_output.running_var = module.running_var                  # pylint: disable=attribute-defined-outside-init
            module_output.num_batches_tracked = module.num_batches_tracked  # pylint: disable=attribute-defined-outside-init

        # Think recursively
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_split_batchnorm(child, num_splits))
        del module

        return module_output


if __name__ == '__main__':

    from torchvision.models import resnet18
    resnet = resnet18(pretrained=False)
    resnet_sbn = SplitBatchNorm2d.convert_split_batchnorm(resnet, num_splits=8)
    resnet_bn = SplitBatchNorm2d.revert_batchnorm(resnet_sbn)
