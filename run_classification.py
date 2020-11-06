# -*- coding: utf-8 -*-

import os
import sys
import time
import rich
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from datasets.cifar import CIFAR10, CIFAR100
from datasets.stl10 import STL10
from datasets.imagenet import TinyImageNet
from datasets.transforms import FinetuneAugment, TestAugment

from configs.task_configs import ClassificationConfig
from models.backbone import ResNetBackbone
from models.head import LinearClassifier
from tasks.classification import Classification
from layers.batchnorm import SplitBatchNorm2d
from utils.logging import get_rich_logger


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'stl10': 10,
    'tinyimagenet': 200,
    'imagenet': 1000,
}


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
            rank=dist_rank
        )

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    num_classes = NUM_CLASSES[config.data]

    # Networks
    encoder = ResNetBackbone(name=config.backbone_type, data=config.data, in_channels=3)
    classifier = LinearClassifier(encoder.out_channels, num_classes)

    # Data (transforms & datasets)
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    finetune_trans = FinetuneAugment(**trans_kwargs)
    test_trans = TestAugment(**trans_kwargs)

    if config.data == 'cifar10':
        train_set = CIFAR10('./data/cifar10', train=True, transform=finetune_trans, proportion=config.labels)
        eval_set = CIFAR10('./data/cifar10', train=False, transform=test_trans)
        test_set = eval_set
    elif config.data == 'cifar100':
        train_set = CIFAR100('./data/cifar100', train=True, transform=finetune_trans, proportion=config.labels)
        eval_set = CIFAR100('./data/cifar100', train=False, transform=test_trans)
        test_set = eval_set
    elif config.data == 'stl10':
        train_set = STL10('./data/stl10', split='train', transform=finetune_trans, proportion=config.labels)
        eval_set = STL10('./data/stl10', split='test', transform=test_trans)
        test_set = eval_set
    elif config.data == 'tinyimagenet':
        train_set = TinyImageNet('./data/tiny-imagenet-200', split='train', transform=finetune_trans, proportion=config.labels)
        eval_set = TinyImageNet('./data/tiny-imagenet-200', split='val', transform=test_trans)
        test_set = eval_set
    elif config.data == 'imagenet':
        raise NotImplementedError("Classification for standard 'ImageNet' is not available.")
    else:
        raise ValueError

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile=logfile)
    else:
        logger = None

    # Load pre-trained weights (if provided)
    if config.pretrained_file is not None:
        try:
            encoder.load_weights_from_checkpoint(path=config.pretrained_file, key='encoder')
        except KeyError:
            encoder.load_weights_from_checkpoint(path=config.pretrained_file, key='backbone')
        finally:
            if logger is not None:
                logger.info(f"Pre-trained model: {config.pretrained_file}")
    else:
        if logger is not None:
            logger.info("No pre-trained model.")

    # Finetune or freeze weights of backbone
    if config.freeze:
        encoder.freeze_weights()
        if logger is not None:
            logger.info("Freezing backbone weights.")

    # Reconfigure batch-norm layers
    if config.pretrained_task in ['moco', 'clapp']:
        encoder = SplitBatchNorm2d.revert_batchnorm(encoder)

    if local_rank == 0:
        logger.info(f'Data: {config.data}')
        logger.info(f'Observations: {len(train_set):,}')
        logger.info(f'Backbone ({config.backbone_type}): {encoder.num_parameters:,}')
        logger.info(f'Classifier (Linear): {classifier.num_parameters:,}')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')

    # Model (Task)
    model = Classification(encoder, classifier)
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        cosine_warmup=config.cosine_warmup,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=config.distributed,
        local_rank=local_rank,
        mixed_precision=True
    )

    # Train & evaluate
    start = time.time()
    model.run(
        train_set=train_set,
        eval_set=eval_set,
        test_set=test_set,
        save_every=config.save_every,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()


if __name__ == '__main__':

    seed = int(np.random.randint(100, size=1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
