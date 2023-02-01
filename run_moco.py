
import os
import sys
import typing
import random
import warnings

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10ForMoCo, CIFAR10
from datasets.cifar import CIFAR100ForMoCo, CIFAR100
from datasets.stl10 import STL10, STL10ForMoCo
from datasets.imagenet import ImageNet, ImageNetForMoCo
from datasets.transforms import MoCoAugment, RandAugment
from datasets.transforms import FinetuneAugment, TestAugment
from configs.task_configs import MoCoConfig, SupMoCoConfig
from tasks.moco import MoCo, SupMoCoAttract, SupMoCoEliminate
from utils.wandb import initialize_wandb

from configs.task_configs import MoCoWithMixturesConfig
from tasks.moco_with_mixtures import MoCoWithMixtures


augmentations = {
    'rand': RandAugment,
    'moco': MoCoAugment,
    'finetune': FinetuneAugment,
    'test': TestAugment,
}


def fix_random_seed(s: int = 0):
    """Fix random seed for reproduction."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)


def main(config: typing.Union[MoCoConfig, SupMoCoConfig, MoCoWithMixturesConfig]):
    """
    Main function for single or distributed MoCo training.
    Note that this `main` function is also used to run models that 
    inherit the `tasks.moco.MoCo` class.
    Arguments:
        config;
    """

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


def main_worker(local_rank: int, config: typing.Union[MoCoConfig, SupMoCoConfig, MoCoWithMixturesConfig]):
    """Single process of MoCo training."""

    # Initialize the training process.
    torch.cuda.set_device(local_rank)
    if config.distributed:
        dist_rank = config.node_rank * config.num_gpus_per_node + local_rank
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=dist_rank,
        )

    # For distributed training across multiple GPUs, the training batch size and
    # number of CPU workers for data loading must be manually divided by the
    # total number of GPUs.
    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.world_size

    # Different data augmentations are used for training & intermediate model evaluation.
    data_aug_config = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    query_transform = augmentations[config.query_augment](**data_aug_config)
    key_transform = augmentations[config.key_augment](**data_aug_config)
    memory_transform = FinetuneAugment(**data_aug_config)
    test_transform = TestAugment(**data_aug_config)

    # Instantiate datasets used for training, evaluation, and testing.
    data_dir = os.path.join(config.data_root, config.data)  # e.g., './data/' + 'imagenet'
    if config.data == 'cifar10':
        train_set = CIFAR10ForMoCo(data_dir,
                                   train=True,
                                   query_transform=query_transform,
                                   key_transform=key_transform)
        memory_set = CIFAR10(data_dir, train=True, transform=memory_transform)
        test_set     = CIFAR10(data_dir, train=False, transform=test_transform)
    elif config.data == 'cifar100':
        train_set = CIFAR100ForMoCo(data_dir,
                                    train=True,
                                    query_transform=query_transform,
                                    key_transform=key_transform)
        memory_set = CIFAR100(data_dir, train=True, transform=memory_transform)
        test_set     = CIFAR100(data_dir, train=False, transform=test_transform)
    elif config.data == 'stl10':
        train_set = STL10ForMoCo(data_dir,
                                 split='train+unlabeled',
                                 query_transform=query_transform,
                                 key_transform=key_transform)
        memory_set = STL10(data_dir, split='train', transform=memory_transform)
        test_set     = STL10(data_dir, split='test', transform=test_transform)
    elif config.data == 'imagenet':
        train_set    = ImageNetForMoCo(data_dir,
                                       split='train',
                                       query_transform=query_transform,
                                       key_transform=key_transform)
        memory_set, test_set = ImageNet.split_into_two_subsets(
            data_dir, split='val', transforms=[memory_transform, test_transform]
        )
    else:
        raise NotImplementedError(
            f"Invalid data argument: {config.data}. "
            f"Supports only one of the following: cifar10, cifar100, stl10, imagenet."
        )
    
    # A wandb instance: https://wandb.ai
    if local_rank == 0:
        initialize_wandb(config)
    
    # Instantiate a MoCo trainer.
    if config.__class__.__name__ == 'MoCoConfig':
        trainer = MoCo(config=config, local_rank=local_rank)
    elif config.__class__.__name__ == 'SupMoCoAttractConfig':
        trainer = SupMoCoAttract(config=config, local_rank=local_rank)
    elif config.__class__.__name__ == 'SupMoCoEliminateConfig':
        trainer = SupMoCoEliminate(config=config, local_rank=local_rank)
    elif config.__class__.__name__ == 'MoCoWithMixturesConfig':
        trainer = MoCoWithMixtures(config=config, local_rank=local_rank)
    else:
        raise NotImplementedError

    # Start training. If the memory and test sets are provided, the representations
    # will be evaluated using the nearest neighbors algorithm at the end of every epoch.
    # A fast implementation of logistic regression will be available in future versions.
    _ = trainer.run(train_set=train_set,
                    memory_set=memory_set,
                    test_set=test_set)    
    wandb.finish()


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    config = MoCoConfig.parse_arguments()
    
    try:
        main(config)
    except KeyboardInterrupt:
        wandb.finish()
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        sys.exit(0)
