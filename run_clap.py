
import os
import sys
import random

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10ForCLAP, CIFAR10
from datasets.cifar import CIFAR100ForCLAP, CIFAR100
from datasets.stl10 import STL10ForCLAP, STL10
from datasets.imagenet import ImageNetForCLAP, ImageNet
from datasets.transforms import MoCoAugment, RandAugment
from datasets.transforms import FinetuneAugment, TestAugment
from configs.task_configs import CLAPConfig
from tasks.clap import CLAP, CLAPv2
from utils.wandb import initialize_wandb


augmentations = {
    'rand': RandAugment,
    'moco': MoCoAugment,
}


def fix_random_seed(s: int = 0):
    """Fix random seed for reproduction."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)


def main(config: CLAPConfig):
    """Main function for single or distributed CLAP training."""

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
    """Single process of CLAP training."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        dist_rank = config.node_rank * config.num_gpus_per_node + local_rank
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=dist_rank,
        )

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.world_size

    # Data
    data_aug_config = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    query_transform    : nn.Module = augmentations[config.query_augment](**data_aug_config)
    key_transform      : nn.Module = augmentations[config.key_augment](**data_aug_config)
    teacher_transform  : nn.Module = augmentations[config.teacher_augment](**data_aug_config)
    finetune_transform : nn.Module = FinetuneAugment(**data_aug_config)
    test_transform     : nn.Module = TestAugment(**data_aug_config)

    data_dir = os.path.join(config.data_root, config.data)
    if config.data == 'cifar10':
        train_set = CIFAR10ForCLAP(data_dir,
                                   train=True,
                                   query_transform=query_transform,
                                   key_transform=key_transform,
                                   teacher_transform=teacher_transform)
        finetune_set = CIFAR10(data_dir, train=True, transform=finetune_transform)
        test_set     = CIFAR10(data_dir, train=False, transform=test_transform)
    elif config.data == 'cifar100':
        train_set = CIFAR100ForCLAP(data_dir,
                                    train=True,
                                    query_transform=query_transform,
                                    key_transform=key_transform,
                                    teacher_transform=teacher_transform)
        finetune_set = CIFAR100(data_dir, train=True, transform=finetune_transform)
        test_set     = CIFAR100(data_dir, train=False, transform=test_transform)
    elif config.data == 'stl10':
        train_set = STL10ForCLAP(data_dir,
                                 split='train+unlabeled',
                                 query_transform=query_transform,
                                 key_transform=key_transform,
                                 teacher_transform=teacher_transform)
        finetune_set = STL10(data_dir, split='train', transform=finetune_transform)
        test_set     = STL10(data_dir, split='test', transform=test_transform)
    elif config.data == 'imagenet':
        train_set = ImageNetForCLAP(data_dir,
                                    split='train',
                                    query_transform=query_transform,
                                    key_transform=key_transform,
                                    teacher_transform=teacher_transform)
        finetune_set, test_set = ImageNet.split_into_two_subsets(
            data_dir, split='val', transforms=[finetune_transform, test_transform]
        )
    else:
        raise NotImplementedError(
            f"Invalid data argument: {config.data}. "
            f"Supports only one of the following: cifar10, cifar100, stl10, imagenet."
        )

    # A wandb instance: https://wandb.ai
    if local_rank == 0:
        initialize_wandb(config)
    
    # Instantiate a CLAP trainer
    if config.__class__.__name__ == 'CLAPConfig':
        trainer = CLAP(config=config, local_rank=local_rank)
    elif config.__class__.__name__ == 'CLAPv2Config':
        trainer = CLAPv2(config=config, local_rank=local_rank)
    else:
        raise NotImplementedError

    # Start training. If the finetune and test sets are provided, the representations
    # will be evaluated using the nearest neighbors algorithm at the end of every epoch.
    # A fast implementation of logistic regression will be available in future versions.
    elapsed_sec = trainer.run(
        train_set=train_set,
        memory_set=finetune_set,
        test_set=test_set,
    )
    if trainer.logger is not None:
        elapsed_mins = elapsed_sec / 60
        elapsed_hours = elapsed_mins / 60
        trainer.logger.info(f'Total training time: {elapsed_mins:,.2f} minutes ({elapsed_hours:,.2f} hours).')
        trainer.logger.handlers.clear()

    wandb.finish()


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    config = CLAPConfig.parse_arguments()
    
    try:
        main(config)
    except KeyboardInterrupt:
        wandb.finish()
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        sys.exit(0)
