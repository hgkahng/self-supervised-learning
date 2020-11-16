# -*- coding: utf-8 -*-

import os
import sys
import time
import rich

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10Pair, CIFAR10, CIFAR10ForMoCo
from datasets.cifar import CIFAR100Pair, CIFAR100, CIFAR100ForMoCo
from datasets.stl10 import STL10Pair, STL10
from datasets.imagenet import TinyImageNetPair, TinyImageNet
from datasets.transforms import MoCoAugment, FinetuneAugment, TestAugment
from configs.task_configs import MoCoConfig
from models.backbone import ResNetBackbone
from models.head import LinearHead, MLPHead
from layers.batchnorm import SplitBatchNorm2d
from tasks.moco import MoCo, MemoryQueue, MoCoLoss
from utils.logging import get_rich_logger
from utils.wandb import configure_wandb


def main():
    """Main function for single or distributed MoCo training."""

    config = MoCoConfig.parse_arguments()
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

    if config.distributed:
        rich.print(f"Distributed training on {world_size} GPUs.")
        mp.spawn(
            main_worker,
            nprocs=config.num_gpus_per_node,
            args=(config, )
        )
    else:
        rich.print(f"Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: object):
    """Single process."""

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

    # Logging
    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile)
        if config.enable_wandb:
            configure_wandb(
                name='moco:' + config.hash,
                project=config.data,
                config=config
            )
    else:
        logger = None

    # Networks
    encoder = ResNetBackbone(name=config.backbone_type, data=config.data, in_channels=3)
    if config.projector_type == 'linear':
        head = LinearHead(encoder.out_channels, config.projector_dim)
    elif config.projector_type == 'mlp':
        head = MLPHead(encoder.out_channels, config.projector_dim)

    if config.split_bn:
        encoder = SplitBatchNorm2d.convert_split_batchnorm(encoder)

    # Data
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    pretrain_trans = MoCoAugment(**trans_kwargs)
    finetune_trans = FinetuneAugment(**trans_kwargs)
    test_trans = TestAugment(**trans_kwargs)

    if config.data == 'cifar10':
        train_set = CIFAR10Pair('./data/cifar10', train=True, transform=pretrain_trans)
        finetune_set = CIFAR10('./data/cifar10', train=True, transform=finetune_trans)
        test_set = CIFAR10('./data/cifar10', train=False, transform=test_trans)
    elif config.data == 'cifar100':
        train_set = CIFAR100Pair('./data/cifar100', train=True, transform=pretrain_trans)
        finetune_set = CIFAR100('./data/cifar100', train=True, transform=finetune_trans)
        test_set = CIFAR100('./data/cifar100', train=False, transform=test_trans)
    elif config.data == 'stl10':
        train_set = STL10Pair('./data/stl10', split='train+unlabeled', transform=pretrain_trans)
        finetune_set = STL10('./data/stl10', split='train', transform=finetune_trans)
        test_set = STL10('./data/stl10', split='test', transform=test_trans)
    elif config.data == 'tinyimagenet':
        train_set = TinyImageNetPair('./data/tiny-imagenet-200', split='train', transform=pretrain_trans, in_memory=True)
        finetune_set = TinyImageNet('./data/tiny-imagenet-200', split='train', transform=finetune_trans, in_memory=True)
        test_set = TinyImageNet('./data/tiny-imagenet-200', split='val', transform=test_trans, in_memory=True)
    elif config.data == 'imagenet':
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data argument: {config.data}")

    # Logging
    if local_rank == 0:
        logger.info(f'Data: {config.data}')
        logger.info(f'Observations: {len(train_set):,}')
        logger.info(f'Backbone ({config.backbone_type}): {encoder.num_parameters:,}')
        logger.info(f'Projector ({config.projector_type}): {head.num_parameters:,}')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')
    else:
        logger = None

    # Model (Task)
    model = MoCo(
        encoder=encoder,
        head=head,
        queue=MemoryQueue(size=(config.projector_dim, config.num_negatives),
                          device=local_rank),
        loss_function=MoCoLoss(config.temperature)
    )
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        cosine_warmup=config.cosine_warmup,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        key_momentum=config.key_momentum,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=config.mixed_precision,
        resume=config.resume
    )

    # Train & evaluate
    start = time.time()
    model.run(
        train_set,
        epochs=config.epochs,
        save_every=config.save_every,
        logger=logger,
        memory_set=finetune_set,
        query_set=test_set,
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        elapsed_hours = elapsed_mins / 60
        logger.info(
            f'Total training time: {elapsed_mins:,.2f} minutes ({elapsed_hours:,.2f} hours).'
        )
        logger.handlers.clear()


if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
