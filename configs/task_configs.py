# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime


class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),            # task-agnostic
            cls.data_parser(),           # task-agnostic
            cls.model_parser(),          # task-agnostic
            cls.train_parser(),          # task-agnostic
            cls.logging_parser(),        # task-agnostic
            cls.task_specific_parser(),  # task-specific
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['model_name'] = self.model_name
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self.backbone_type

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,          # 'wm811k', 'cifar10', 'stl10', 'imagenet', ...
            self.task,          # 'scratch', 'denoising', 'pirl', 'simclr', ...
            self.model_name,    # 'resnet18', 'resnet50', ...
            self.hash           # ...
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument('--gpus', type=int, nargs='+', default=None, required=True, help='')
        parser.add_argument('--num_nodes', type=int, default=1, help='')
        parser.add_argument('--node_rank', type=int, default=0, help='')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
        parser.add_argument('--dist_backend', type=str, default='nccl', help='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data', type=str, choices=('cifar10', 'cifar100', 'svhn', 'stl10', 'tinyimagenet', 'imagenet'), required=True)
        parser.add_argument('--input_size', type=int, choices=(32, 64, 96, 224), required=True)
        parser.add_argument('--augmentation', type=str, default='torchvision',
                            choices=('torchvision', 'albumentations'), help='Package used for augmentation.')
        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("CNN Backbone", add_help=False)
        parser.add_argument('--backbone_type', type=str, default='resnet50', choices=('resnet18', 'resnet50'), required=True)
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=0, help='Number of CPU threads.')
        parser.add_argument('--optimizer', type=str, default='lars', choices=('sgd', 'adamw', 'lars'), help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=1e-2, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay factor.')
        parser.add_argument('--cosine_warmup', type=int, default=-1, help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1, help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=5e-3, help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', action='store_true', help='Use float16 precision.')
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=100, help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--enable_wandb', action='store_true', help='Use Weights & Biases plugin.')
        return parser


class PretrainConfigBase(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(PretrainConfigBase, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        raise NotImplementedError


class DenoisingConfig(PretrainConfigBase):
    """Configurations for Denoising."""
    def __init__(self, args=None, **kwargs):
        super(DenoisingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser("Denoising", add_help=False)
        parser.add_argument('--noise', type=float, default=0.05)
        return parser

    @property
    def task(self):
        return 'denoising'


class InpaintingConfig(PretrainConfigBase):
    """Configurations for Inpainting."""
    def __init__(self, args=None, **kwargs):
        super(InpaintingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'inpainting'


class JigsawConfig(PretrainConfigBase):
    """Configurations for Jigsaw."""
    def __init__(self, args=None, **kwargs):
        super(JigsawConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'jigsaw'


class RotationConfig(PretrainConfigBase):
    """Configurations for Rotation."""
    def __init__(self, args=None, **kwargs):
        super(RotationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'rotation'


class BiGANConfig(PretrainConfigBase):
    """Configurations for BiGAN."""
    def __init__(self, args=None, **kwargs):
        super(BiGANConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'bigan'


class PIRLConfig(PretrainConfigBase):
    """Configurations for PIRL."""
    def __init__(self, args=None, **kwargs):
        super(PIRLConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('PIRL', add_help=False)
        parser.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projector head.')
        parser.add_argument('--temperature',  type=float, default=0.07, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=5000, help='Number of negative examples.')
        parser.add_argument('--loss_weight', type=float, default=0.5, help='Weighting factor of two loss terms, [0, 1].')
        return parser

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,          # 'wm811k', 'cifar10', 'stl10', 'imagenet', ...
            self.task,          # 'scratch', 'denoising', 'pirl', 'simclr', ...
            self.model_name,    # 'resnet.50.original', ...
            self.augmentation,  # 'crop', 'cutout', ...
            self.hash           # ...
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self) -> str:
        return 'pirl'


class MoCoConfig(PretrainConfigBase):
    """Configurations for MoCo."""
    def __init__(self, args=None, **kwargs):
        super(MoCoConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('MoCo', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.2, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=65536, help='Number of negative examples to maintain.')
        parser.add_argument('--key_momentum', type=float, default=0.999, help='Momentum for updating key encoder.')
        parser.add_argument('--query_augment', type=str, default='moco', help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco', help='Augmentation applied to key (x_k).')
        parser.add_argument('--rand_k', type=int, default=5, help='Strength of RandAugment, if used.')
        parser.add_argument('--split_bn', action='store_true')
        parser.add_argument('--knn_k', type=int, nargs='+', default=[5, 200], help='')
        return parser

    @property
    def task(self) -> str:
        return 'moco'


class BYOLConfig(PretrainConfigBase):
    """Configurations for BYOL."""
    def __init__(self, args=None, **kwargs):
        super(BYOLConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('BYOL', add_help=False)
        parser.add_argument('--projector_type', type=str, default='byol', choices=('byol', ))
        parser.add_argument('--projector_dim', type=int, default=256, help='Output dimension of projection / predictor.')
        parser.add_argument('--projector_hidden_dim', type=int, default=4096, help='Hidden dimension of projector / predictor.')
        parser.add_argument('--knn_k', type=int, nargs='+', default=[5, 200], help='')
        return parser
    
    @property
    def task(self) -> str:
        return 'byol'


class CLAPPConfig(PretrainConfigBase):
    """Configurations for CLAPP."""
    def __init__(self, args=None, **kwargs):
        super(CLAPPConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('CLAPP', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--num_negatives', type=int, default=65536, help='Number of negative examples to maintain.')
        parser.add_argument('--temperature', type=float, default=0.2, help='Logit scaling factor.')
        parser.add_argument('--pseudo_temperature', type=float, default=0.1, help='Pseudo logit scaling factor.')
        parser.add_argument('--key_momentum', type=float, default=0.999, help='Momentum for updating key network.')
        parser.add_argument('--pseudo_momentum', type=float, default=0.5, help='Momentum for updating pseudo network.')
        parser.add_argument('--query_augment', type=str, default='rand', help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco', help='Augmentation applied to key (x_k).')
        parser.add_argument('--pseudo_augment', type=str, default='moco', help='Augmentation applied to pseudo labeler (x_p).')
        parser.add_argument('--rand_k', type=int, default=5, help='Strength of RandAugment, if used.')
        parser.add_argument('--contrast_mode', type=str, default='batch', choices=['queue', 'batch'], help='Queue vs. Batch.')
        parser.add_argument('--normalize', type=str, default='softmax', help='Method for normalizing logits to distribution.')
        parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for pseudo labeling.')
        parser.add_argument('--split_bn', action='store_true', help='Use ghost batch normalization.')
        parser.add_argument('--ramp_up', type=int, default=50, help='50 for CIFAR-10, 200 for CIFAR-100/TinyImageNet.')
        parser.add_argument('--knn_k', type=int, nargs='+', default=[5, 200], help='')
        return parser

    @property
    def task(self) -> str:
        return 'clapp'


class SimCLRConfig(PretrainConfigBase):
    """Configurations for SimCLR."""
    def __init__(self, args=None, **kwargs):
        super(SimCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('SimCLR', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp', 'attention'))
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.07, help='Logit scaling factor.')
        return parser

    @property
    def task(self) -> str:
        return 'simclr'


class PseudoCLRConfig(PretrainConfigBase):
    def __init__(self, args=None, **kwargs):
        super(PseudoCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('PseudoCLR', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
        parser.add_argument('--projector_dim', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.07, help='Logit scaling factor.')
        parser.add_argument('--num_pseudo', type=int, default=1)
        parser.add_argument('--ensemble', type=int, default=0)
        parser.add_argument('--kmeans', action='store_true')
        return parser

    @property
    def task(self) -> str:
        return 'pseudoclr'

class SemiCLRConfig(SimCLRConfig):
    """Configurations for SemiCLR."""
    def __init__(self, args=None, **kwargs):
        super(SemiCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @property
    def task(self) -> str:
        return 'semiclr'


class AttnCLRConfig(SimCLRConfig):
    """Configurations for AttnCLR."""
    def __init__(self, args=None, **kwargs):
        super(AttnCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @property
    def task(self) -> str:
        return 'attnclr'


class DownstreamConfigBase(ConfigBase):
    """Configurations for downstream tasks."""
    def __init__(self, args=None, **kwargs):
        super(DownstreamConfigBase, self).__init__(args, **kwargs)


class ClassificationConfig(DownstreamConfigBase):
    def __init__(self, args=None, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Linear evaluation of pre-trained model.', add_help=False)
        parser.add_argument('--labels', type=float, default=1.0, help='Size of labeled data (0, 1].')
        parser.add_argument('--pretrained_file', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--pretrained_task', type=str, default=None, help='Type of pretraining task.')
        parser.add_argument('--freeze', action='store_true', help='Freeze weights of CNN backbone.')
        return parser

    @property
    def task(self) -> str:
        if self.pretrained_file is not None:
            if self.pretrained_task is None:
                raise ValueError("Provide a proper name for the pretrained model type.")
            if self.freeze:
                return f"linear_{self.pretrained_task}"
            else:
                return f"finetune_{self.pretrained_task}"
        else:
            return "from_scratch"


class MixupConfig(ClassificationConfig):
    def __init__(self, args=None, **kwargs):
        super(MixupConfig, self).__init__(args, **kwargs)

    @property
    def task(self) -> str:
        return 'mixup'
