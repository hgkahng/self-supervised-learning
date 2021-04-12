# -*- coding: utf-8 -*-

from datasets.transforms.supervised import FinetuneAugment
import os
import sys
import time
import rich

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from rich.console import Console

from datasets.cifar import CIFAR10, CIFAR10ForSimCLR
from datasets.cifar import CIFAR100, CIFAR100ForSimCLR
from datasets.transforms import SimCLRAugment, FinetuneAugment, TestAugment
from configs.task_configs import BYOLConfig
from models.backbone import ResNetBackbone
from models.head import BYOLProjectionHead, BYOLPredictionHead
from tasks.byol import BYOL, BYOLLoss
from utils.logging import get_rich_logger
from utils.wandb import configure_wandb


AUGMENTS = {
    'simclr': SimCLRAugment,
    'byol': SimCLRAugment,
}


def main():
    """Main function for single or distributed BYOL training."""

    config = BYOLConfig.parse_arguments()
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
        rich.print("Single GPU training.")
        main_worker(0, config=config)


def main_worker(local_rank: int, config: object):
    """Single process."""

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

    # Networks
    encoder = ResNetBackbone(name=config.backbone_type,
                             data=config.data,
                             in_channels=3)
    projector = BYOLProjectionHead(
        in_channels=encoder.out_channels,
        hidden_size=config.projector_hid_dim,
        output_size=config.projector_out_dim,
    )
    predictor = BYOLPredictionHead(
        input_size=config.projector_out_dim,
        hidden_size=config.projector_hid_dim,
        output_size=config.projector_out_dim,
    )

    # Data
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    ssl_trans = AUGMENTS['byol'](**trans_kwargs)
    finetune_trans = FinetuneAugment(**trans_kwargs)
    test_trans = TestAugment(**trans_kwargs)

    if config.data == 'cifar10':
        train_set = CIFAR10ForSimCLR('./data/cifar10',
                                     train=True,
                                     transform=ssl_trans)
        finetune_set = CIFAR10('./data/cifar10', train=True, transform=finetune_trans)
        test_set = CIFAR10('./data/cifar10', train=False, transform=test_trans)
    elif config.data == 'cifar100':
        train_set = CIFAR100ForSimCLR('./data/cifar100',
                                      train=True,
                                      transform=ssl_trans)
        finetune_set = CIFAR100('./data/cifar100', train=True, transform=finetune_trans)
        test_set = CIFAR100('./data/cifar100', train=False, transform=test_trans)
    else:
        raise NotImplementedError
    
    # Logging
    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile)
        if config.enable_wandb:
            configure_wandb(
                name='byol:' + config.hash,
                project=config.data,
                config=config
            )

        logger.info(f'Data: {config.data}')
        logger.info(f'Observations: {len(train_set):,}')
        logger.info(f'Backbone ({config.backbone_type}): {encoder.num_parameters:,}')
        logger.info(f'Projector ({config.projector_type}): {projector.num_parameters:,}')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')
    
    else:
        logger = None

    # Model
    model = BYOL(encoder=encoder,
                 projector=projector,
                 predictor=predictor,
                 loss_function=BYOLLoss()
    )
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        cosine_warmup=config.cosine_warmup,
        cosine_cycles=config.cosine_cycles,
        cosine_min_lr=config.cosine_min_lr,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=config.mixed_precision,
        resume=config.resume
    )

    # Train & evaluate
    start = time.time()
    model.run(
        dataset=train_set,
        memory_set=finetune_set,
        query_set=test_set,
        save_every=config.save_every,
        logger=logger,
        knn_k=config.knn_k,
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
        sys.exit()
