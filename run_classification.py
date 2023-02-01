# -*- coding: utf-8 -*-

import os
import sys
import time

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10, CIFAR100
from datasets.stl10 import STL10
from datasets.imagenet import ImageNet
from datasets.transforms import FinetuneAugment, TestAugment

from configs.task_configs import ClassificationConfig
from models.backbone import ResNetBackbone
from models.head import LinearClassifier
from layers.batchnorm import SplitBatchNorm2d
from tasks.classification import Classification
from utils.logging import get_rich_logger
from utils.wandb import initialize_wandb


num_classes_of_dataset = {
    'cifar10': 10,
    'cifar100': 100,
    'stl10': 10,
    'imagenet': 1000,
}


def fix_random_seed(s: int = 0):
    """Fix random seed for reproduction."""
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    """Main function for single/distributed linear classification."""

    config = ClassificationConfig.parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config.gpus])
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    console = Console()
    console.print(config.__dict__)
    config.save()

    if config.distributed:
        console.print(f"Distributed training on {world_size} GPUs.")
        mp.spawn(
            main_worker,
            nprocs=config.num_gpus_per_node,
            args=(config, )
        )
    else:
        console.print(f"Single node training on GPU {config.gpus[0]}.")
        main_worker(local_rank=0, config=config)


def main_worker(local_rank: int, config: object):
    """Single process for {linear evaulation, finetuning, end-to-end training}."""

    # Initialize the training process.
    torch.cuda.set_device(local_rank)
    if config.distributed:
        dist_rank = config.node_rank * config.num_gpus_per_node + local_rank
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=dist_rank
        )

    # For distributed training across multiple GPUs, the training batch size and
    # number of CPU workers for data loading must be manually divided by the
    # total number of GPUs.
    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Different data augmentations are used for training & evaluation.
    data_aug_config = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    train_transform    : nn.Module = FinetuneAugment(**data_aug_config)
    test_transform     : nn.Module = TestAugment(**data_aug_config)

    # Instantiate datasets used for training, evaluation, and testing.
    data_dir = os.path.join(config.data_root, config.data)  # e.g., './data/' + 'imagenet'
    if config.data == 'cifar10':
        train_set = CIFAR10(data_dir,
                            train=True,
                            transform=train_transform,
                            proportion=config.labels)
        eval_set = CIFAR10(data_dir,
                           train=False,
                           transform=test_transform)
        test_set = eval_set
    elif config.data == 'cifar100':
        train_set = CIFAR100(data_dir,
                             train=True,
                             transform=train_transform,
                             proportion=config.labels)
        eval_set = CIFAR100('./data/cifar100',
                            train=False,
                            transform=test_transform)
        test_set = eval_set
    elif config.data == 'stl10':
        train_set = STL10(data_dir,
                          split='train',
                          transform=train_transform,
                          proportion=config.labels)
        eval_set = STL10(data_dir,
                         split='test',
                         transform=test_transform)
        test_set = eval_set
    elif config.data == 'imagenet':
        train_set = ImageNet(data_dir,
                             split='train',
                             transform=train_transform,
                             proportion=config.labels)
        eval_set, test_set = ImageNet.split_into_two_subsets(
            data_dir, split='val', transforms=[test_transform, test_transform]
        )
    else:
        raise NotImplementedError(
            f"Invalid data argument: {config.data}. "
            f"Supports only one of the following: 'cifar10', 'cifar100', 'stl10', 'imagenet'."
        )

    # A wandb instance: https://wandb.ai
    if local_rank == 0:
        initialize_wandb(config)
    trainer = Classification(config=config, local_rank=local_rank)
    _ = trainer.run(dataset=train_set, eval_set=eval_set, test_set=test_set, )
    wandb.finish()


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        main()
    except KeyboardInterrupt:
        wandb.finish()
        sys.exit(0)
