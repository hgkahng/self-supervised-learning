# -*- coding: utf-8 -*-

import os
import sys
import typing
import warnings

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10, CIFAR10Pair
from datasets.cifar import CIFAR100, CIFAR100Pair
from datasets.stl10 import STL10, STL10Pair
from datasets.imagenet import ImageNet, ImageNetPair
from datasets.transforms import MoCoAugment, FinetuneAugment, TestAugment
from configs.task_configs import BYOLConfig
from tasks.byol import BYOL
from utils.wandb import initialize_wandb


def fix_random_seed(s: int = 0):
    np.random.seed(s)
    torch.manual_seed(s)


def main(config: BYOLConfig):
    """Main function for single or distributed BYOL training."""

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config.gpus])
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    console = Console()
    console.log(config.__dict__)
    config.save()

    fix_random_seed(config.seed)

    if config.distributed:
        console.print(f"Distributed training on {world_size} GPUs.")
        mp.spawn(
            main_worker,
            nprocs=config.num_gpus_per_node,
            args=(config, )
        )
    else:
        console.print(f"Single GPU training on GPU {config.gpus[0]}.")
        main_worker(0, config=config)


def main_worker(local_rank: int, config: object):
    """Single process of BYOL training."""

    # Initialize the training process
    torch.cuda.set_device(local_rank)
    if config.distributed:
        dist_rank = config.node_rank * config.num_gpus_per_node + local_rank
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=dist_rank
        )

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    data_aug_config = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    byol_transform = MoCoAugment(**data_aug_config)
    finetune_transform = FinetuneAugment(**data_aug_config)
    test_transform = TestAugment(**data_aug_config)

    # Instantiate datasets used for training, evaluation, and testing.
    data_dir = os.path.join(config.data_root, config.data)
    if config.data == 'cifar10':
        train_set = CIFAR10Pair(data_dir,
                                train=True,
                                transform=byol_transform)
        finetune_set = CIFAR10(data_dir, train=True, transform=finetune_transform)
        test_set     = CIFAR10(data_dir, train=False, transform=test_transform)
    elif config.data == 'cifar100':
        train_set = CIFAR100Pair(data_dir,
                                 train=True,
                                 transform=byol_transform)
        finetune_set = CIFAR100(data_dir, train=True, transform=finetune_transform)
        test_set     = CIFAR100(data_dir, train=False, transform=test_transform)
    elif config.data == 'stl10':
        train_set = STL10Pair(data_dir,
                              split='train+unlabeled',
                              transform=byol_transform)
        finetune_set = STL10(data_dir, split='train', transform=finetune_transform)
        test_set     = STL10(data_dir, split='test', transform=test_transform)
    elif config.data == 'imagenet':
        train_set    = ImageNetPair(data_dir,
                                    split='train',
                                    transform=byol_transform)
        finetune_set, test_set = ImageNet.split_into_two_subsets(
            data_dir, split='val', transforms=[finetune_transform, test_transform]
        )
    else:
        raise NotImplementedError(
            f"Invalid data argument: {config.data}. "
            f"Supports only one of the following: 'cifar10', 'cifar100', 'stl10', 'imagenet'."
        )

    # A wandb instance: https://wandb.ai
    if local_rank == 0:
        initialize_wandb(config)
    
    # Instantiate BYOL trainer
    trainer = BYOL(config=config, local_rank=local_rank)

    # Start training
    elapsed_sec = trainer.run(
        dataset=train_set,
        finetune_set=finetune_set,
        test_set=test_set,
        save_every=config.save_every,
        eval_every=config.eval_every,
    )
    if trainer.logger is not None:
        elapsed_mins = elapsed_sec / 60
        elapsed_hours = elapsed_mins / 60
        trainer.logger.info(f'Total training time: {elapsed_mins:,.2f} minutes ({elapsed_hours:,.2f} hours).')
        trainer.logger.handlers.clear()
        
    wandb.finish()


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    config = BYOLConfig.parse_arguments()

    try:
        main(config)
    except KeyboardInterrupt:
        wandb.finish()
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        sys.exit(0)
