# -*- coding: utf-8 -*-

import os
import sys
import rich

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from datasets.cifar import CIFAR10ForSimCLR
from datasets.stl10 import STL10ForSimCLR
from datasets.imagenet import ImageNetForSimCLR
from datasets.transforms import SimCLRAugment
from configs.task_configs import SimCLRConfig
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.resnet import ResNetBackbone
from models.head import LinearHead, MLPHead
from tasks.simclr import SimCLR
from utils.logging import get_rich_logger


AVAILABLE_MODELS = {
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': LinearHead,
    'mlp': MLPHead,
}

IN_CHANNELS = {
    'cifar10': 3,
    'stl10': 3,
    'imagenet': 3,
}


def main():
    """Main function for single or distributed SimCLR training."""

    config = SimCLRConfig.parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config.gpus])
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)
    
    rich.print(config.__dict__)
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

    config.batch_size = config.batch_size // config.num_gpus_per_node
    config.num_workers = config.num_workers // config.num_gpus_per_node

    # Networks
    BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[config.backbone_type]
    Projector = PROJECTOR_TYPES[config.projector_type]
    encoder = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels=IN_CHANNELS[config.data])
    head = Projector(encoder.out_channels, config.projector_size)

    # Data
    data_transforms = dict(
        transform=SimCLRAugment(
            size=config.input_size,
            data=config.data,
            impl='albumentations'
        )
    )
    if config.data == 'cifar10':
        train_set = CIFAR10ForSimCLR('./data/cifar10/', train=True, **data_transforms)
        eval_set = CIFAR10ForSimCLR('./data/cifar10/', train=False, **data_transforms)
    elif config.data == 'stl10':
        train_set = STL10ForSimCLR('./data/stl10/', split='unlabeled', **data_transforms)
        eval_set = STL10ForSimCLR('./data/stl10/', split='train', **data_transforms)
    elif config.data == 'imagenet':
        train_set = ImageNetForSimCLR('../imagenet2012/', split='train', **data_transforms)
        eval_set = ImageNetForSimCLR('../imagenet2012/', split='val', **data_transforms)
    else:
        raise ValueError

    # Model (Task)
    model = SimCLR(backbone=encoder, projector=head)
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        temperature=config.temperature,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=True,
    )

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger  = get_rich_logger(logfile=logfile)
        logger.info(f'Data: {config.data}')
        logger.info(f'Observations: {len(train_set):,}')
        logger.info(f'Trainable parameters ({encoder.__class__.__name__}): {encoder.num_parameters:,}')
        logger.info(f'Trainable parameters ({head.__class__.__name__}): {head.num_parameters:,}')
        logger.info(f'Projection head: {config.projector_type} ({config.projector_size})')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')
    else:
        logger = None

    model.run(
        train_set,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        cosine_warmup=config.cosine_warmup,
        save_every=config.save_every,
        logger=logger
    )

    if logger is not None:
        logger.handlers.clear()


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
