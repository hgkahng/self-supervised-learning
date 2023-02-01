   # -*- coding: utf-8 -*-

import os
import typing
import copy
import json
import argparse
import datetime


class ConfigBase(object):
    """Base class for Configurations."""
    def __init__(self, args: typing.Union[argparse.Namespace, dict] = None, **kwargs):

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
            setattr(self, 'hash', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    @classmethod
    def parse_arguments(cls) -> object:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),            # task-agnostic
            cls.data_parser(),           # task-agnostic
            cls.model_parser(),          # task-agnostic
            cls.train_parser(),          # task-agnostic
            cls.logging_parser(),        # task-agnostic
            cls.wandb_parser(),          # task-agnostic
            cls.task_specific_parser(),  # task-specific
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str) -> object:
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
            self.task,          # 'scratch', 'linear_*', 'finetune_*', 'simclr', 'moco', ...
            self.model_name,    # 'resnet18', 'resnet50', 'resnet28-10', ...
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
        parser.add_argument('--gpus', type=int, nargs='+', default=None, required=True,
                            help='GPU identifiers to use.')
        parser.add_argument('--num_nodes', type=int, default=1,
                            help='Number of heterogeneous machines participating in training.')
        parser.add_argument('--node_rank', type=int, default=0,
                            help='Rank of process (GPU) of in current node.')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500',
                            help='URL used to communicate between devices.')
        parser.add_argument('--dist_backend', type=str, default='nccl', choices=('nccl', ),
                            help='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--seed', type=int, default=0,
                            help='For reproducibility.')
        parser.add_argument('--data_root', type=str, default='./data',
                            help='Root directory holding datasets.')
        parser.add_argument('--data', type=str, choices=('cifar10', 'cifar100', 'stl10', 'imagenet'), required=True,
                            help='Data on which the model is trained.')
        parser.add_argument('--input_size', type=int, choices=(32, 64, 96, 224), required=True,
                            help='Spatial dimension of input images.')
        parser.add_argument('--augmentation', type=str, default='torchvision', choices=('torchvision', 'dali'),
                            help='Package used for data loading / augmentation.')
        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("CNN Backbone", add_help=False)
        parser.add_argument('--backbone_type', type=str, default='resnet50', required=True,
                            help='Type of neural network encoder.')
        parser.add_argument('--resume', type=str, default=None,
                            help='Path to checkpoint file to resume training from.')
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Global batch size used during training.')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Global number of CPU threads used to load data during training.')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw', 'lars'),
                            help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=3e-2,
                            help='Base learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='Weight decay factor.')
        parser.add_argument('--lr_warmup', type=int, default=10,
                            help='Number of linear warmup steps of LR before cosine decay.')
        parser.add_argument('--mixed_precision', action='store_true',
                            help='Use float16 precision for faster & lighter experiments.')
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/',
                            help='Top-level directory of checkpoints.')
        parser.add_argument('--save_every', type=int, default=100,
                            help='Save model checkpoint every `save_every` epochs.')
        parser.add_argument('--eval_every', type=int, default=1,
                            help='Evaluate model performance every `eval_every` epochs.')

        return parser

    @staticmethod
    def wandb_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("WandB", add_help=False)
        parser.add_argument('--wandb_project', type=str, default=None,
                            help='')
        parser.add_argument('--wandb_group', type=str, default=None,
                            help='')
        parser.add_argument('--wandb_name', type=str, default=None,
                            help='')
        parser.add_argument('--wandb_job_type', type=str, default=None,
                            help='')
        parser.add_argument('--wandb_entity', type=str, default=None,
                            help='')
        parser.add_argument('--wandb_id', type=str, default=None,
                            help='')

        return parser

    def remove_unused_arguments(self):
        raise NotImplementedError


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
        parser.add_argument('--noise', type=float, default=0.05,
                            help='Level of gaussian noise applied to input.')
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
        parser.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projector head.')
        parser.add_argument('--temperature',  type=float, default=0.07,
                            help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=5000,
                            help='Number of negative examples stored in the memory bank.')
        parser.add_argument('--loss_weight', type=float, default=0.5,
                            help='Weighting factor of two loss terms, [0, 1].')
        return parser

    @property
    def task(self) -> str:
        return 'pirl'


class SimCLRConfig(PretrainConfigBase):
    """Configurations for SimCLR."""
    def __init__(self, args=None, **kwargs):
        super(SimCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('SimCLR', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.07,
                            help='Logit scaling factor.')
        return parser

    @property
    def task(self) -> str:
        return 'simclr'


class MoCoConfig(PretrainConfigBase):
    """Configurations for MoCo."""
    def __init__(self, args=None, **kwargs):
        super(MoCoConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'MoCo') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('mlp', ),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=65536,
                            help='Number of negative examples to store in the memory queue.')
        parser.add_argument('--key_momentum', type=float, default=0.999,
                            help='Momentum for updating key encoder.')
        parser.add_argument('--query_augment', type=str, default='moco',
                            help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco',
                            help='Augmentation applied to key (x_k).')
        return parser

    @property
    def task(self) -> str:
        return 'moco'

class MoCoWithMixturesConfig(PretrainConfigBase):
    """Configurations for MoCoWithMixtures."""
    def __init__(self, args=None, **kwargs):
        super(MoCoWithMixturesConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'MoCoWithMixtures') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('mlp', ),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=65536,
                            help='Number of negative examples to store in the memory queue.')
        parser.add_argument('--key_momentum', type=float, default=0.999,
                            help='Momentum for updating key encoder.')
        parser.add_argument('--query_augment', type=str, default='moco',
                            help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco',
                            help='Augmentation applied to key (x_k).')

        parser.add_argument('--dropout_rate', type=float, default=0.2,
                            help='')
        parser.add_argument('--num_extra_positives', type=int, default=10,
                            help='')
        parser.add_argument('--mixture_n_components', type=int, default=None,
                            help='')
        parser.add_argument('--mixture_n_iter', type=int, default=100,
                            help='')
        parser.add_argument('--mixture_tol', type=float, default=1e-6,
                            help='')
        parser.add_argument('--threshold_lower', type=float, default=0.01,
                            help='')
        parser.add_argument('--threshold_upper', type=float, default=0.5,
                            help='')
        
        return parser
    
    @property
    def task(self) -> str:
        return 'mocoMixtures'

class SupMoCoConfig(PretrainConfigBase):
    """Configurations for SupMoCo."""
    def __init__(self, args=None, **kwargs):
        super(SupMoCoConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'SupMoCo') -> argparse.ArgumentParser:
        parser = MoCoConfig.task_specific_parser(name)
        parser.add_argument('--num_positives', type=int, default=1,
                            help='Number of positives to consider.')
        parser.add_argument('--loss_weight', type=float, default=.5,
                            help='Loss balancing term.')
        return parser

    @property
    def task(self) -> str:
        return 'supmoco'


class SupMoCoAttractConfig(PretrainConfigBase):
    """Configurations for SupMoCo, attraction version."""
    def __init__(self, args=None, **kwargs):
        super(SupMoCoAttractConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'SupMoCo-attract') -> argparse.ArgumentParser:
        parser = MoCoConfig.task_specific_parser(name)
        parser.add_argument('--num_positives', type=int, default=1,
                            help='Number of positives to consider.')
        parser.add_argument('--loss_weight', type=float, default=.5,
                            help='Loss balancing term.')
        return parser

    @property
    def task(self) -> str:
        return 'supmoco-attract'


class SupMoCoEliminateConfig(PretrainConfigBase):
    """Configurations for SupMoCo, elimination version."""
    def __init__(self, args=None, **kwargs):
        super(SupMoCoEliminateConfig, self).__init__(args, **kwargs)
    
    @staticmethod
    def task_specific_parser(name: str = 'SupMoCo-attract') -> argparse.ArgumentParser:
        parser = MoCoConfig.task_specific_parser(name)
        return parser

    @property
    def task(self) -> str:
        return 'supmoco-eliminate'

class BYOLConfig(PretrainConfigBase):
    """Configurations for BYOL."""
    def __init__(self, args=None, **kwargs):
        super(BYOLConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'BYOL') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_out_dim', type=int, default=256,
                            help='Output dimension of projection / predictor.')
        parser.add_argument('--projector_hid_dim', type=int, default=4096,
                            help='Hidden dimension of projector / predictor.')
        parser.add_argument('--target_decay_base', type=float, default=0.996,
                            help='Base momentum coefficient for updating target network.')
        return parser

    @property
    def task(self) -> str:
        return 'byol'


class SimSiamConfig(PretrainConfigBase):
    """Configurations for SimSiam."""
    def __init__(self, args=None, **kwargs):
        super(SimSiamConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('SimSiam', add_help=False)
        return parser

    @property
    def task(self) -> str:
        return 'simsiam'


class NNCLRConfig(PretrainConfigBase):
    """Configurations for NNCLR."""
    def __init__(self, args=None, **kwargs):
        super(NNCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('NNCLR', add_help=False)
        return parser

    @property
    def task(self) -> str:
        return 'nnclr'


class BarlowTwinsConfig(PretrainConfigBase):
    """Configurations for BarlowTwins."""
    def __init__(self, args=None, **kwargs):
        super(BarlowTwinsConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('BarlowTwins', add_help=False)
        return parser

    @property
    def task(self) -> str:
        return 'barlow-twins'


class UNCLConfig(PretrainConfigBase):
    """Configurations for UNCL."""
    def __init__(self, args=None, **kwargs):
        super(UNCLConfig, self).__init__(args, **kwargs)
    
    @staticmethod
    def task_specific_parser(name: str = 'UNCL') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('mlp', ),
                            help='')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='')
        parser.add_argument('--key_momentum', type=float, default=0.999,
                            help='')
        parser.add_argument('--teacher_momentum', type=float, default=0.999,
                            help='')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='')
        parser.add_argument('--teacher_temperature', type=float, default=0.2,
                            help='')
        parser.add_argument('--num_negatives', type=int, default=65536,
                            help='')
        parser.add_argument('--ensemble_num_estimators', type=int, default=128,
                            help='')
        parser.add_argument('--ensemble_dropout_rate', type=float, default=0.2,
                            help='')
        parser.add_argument('--uncertainty_threshold', type=float, default=0.5,
                            help='')
        parser.add_argument('--num_false_negatives', type=int, default=1,
                            help='')

        return parser

    @property
    def task(self) -> str:
        return 'uncl'

class CLAPConfig(PretrainConfigBase):
    """Configurations for CLAP."""
    def __init__(self, args=None, **kwargs):
        super(CLAPConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'CLAP') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('mlp', ),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projection head.')
        parser.add_argument('--key_momentum', type=float, default=0.999,
                            help='Momentum for updating key network.')
        parser.add_argument('--teacher_momentum', type=float, default=0.999,
                            help='Momentum for updating teacher network.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='Scaling factor for sharpening/broadening similarities between query and {key, queue}.')
        parser.add_argument('--num_negatives', type=int, default=65536,
                            help='Number of negative examples to maintain. use exponentials of 2.')
        parser.add_argument('--query_augment', type=str, default='moco',
                            help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco',
                            help='Augmentation applied to key (x_k).')
        parser.add_argument('--teacher_augment', type=str, default='moco',
                            help='Augmentation applied to teacher (x_t).')
        parser.add_argument('--ensemble_num_estimators', type=int, default=128,
                            help='Number of ensemble estimators.')
        parser.add_argument('--ensemble_dropout_rate', type=float, default=0.5,
                            help='Proportion of features to drop during ensembling.')
        parser.add_argument('--easy_pos_ub', type=float, default=0.5)
        parser.add_argument('--hard_pos_lb', type=float, default=0.5)
        parser.add_argument('--easy_pos_start_epoch', type=int, default=0)
        parser.add_argument('--hard_pos_start_epoch', type=int, default=250)
        parser.add_argument('--num_positives', type=int, default=1,
                            help='Number of positives to consider.')
        parser.add_argument('--loss_weight_easy', type=float, default=0.5)
        parser.add_argument('--loss_weight_hard', type=float, default=0.5)

        return parser

    @property
    def task(self) -> str:
        return 'clap'

    def remove_unused_arguments(self):
        # Remove `rand_k` when RandAugment is not used
        if (self.query_augment != 'rand') & (self.key_augment != 'rand') & (self.teacher_augment != 'rand'):
            setattr(self, 'rand_k', None)
        # Remove unused arguments based on `selection_scheme`
        if self.selection_scheme == 'batch':
            setattr(self, 'support_size', None)
            setattr(self, 'ensemble', None)
            setattr(self, 'nn_size', None)
        elif self.selection_scheme == 'queue':
            setattr(self, 'nn_size', None)
        elif self.selection_scheme == 'nn':
            setattr(self, 'normalize', None)
            setattr(self, 'teacher_temperature', None)
            setattr(self, 'threshold', None)
            setattr(self, 'support_size', None)
            setattr(self, 'ensemble', None)
        else:
            raise NotImplementedError

class CLAPv2Config(PretrainConfigBase):
    """Configurations for CLAPv2."""
    def __init__(self, args=None, **kwargs):
        super(CLAPv2Config, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser(name: str = 'CLAPv2') -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('mlp', ),
                            help='Type of projection head.')
        parser.add_argument('--projector_dim', type=int, default=128,
                            help='Dimension of projection head.')
        parser.add_argument('--key_momentum', type=float, default=0.999,
                            help='Momentum for updating key network.')
        parser.add_argument('--teacher_momentum', type=float, default=0.000,
                            help='Momentum for updating teacher network.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='Scaling factor for sharpening/broadening similarities between query and {key, queue}.')
        parser.add_argument('--num_negatives', type=int, default=65536,
                            help='Number of negative examples to maintain. use exponentials of 2.')
        parser.add_argument('--query_augment', type=str, default='moco',
                            help='Augmentation applied to query (x_q).')
        parser.add_argument('--key_augment', type=str, default='moco',
                            help='Augmentation applied to key (x_k).')
        parser.add_argument('--teacher_augment', type=str, default='moco',
                            help='Augmentation applied to teacher (x_t).')
        parser.add_argument('--loss_weight', type=float, default=0.5,
                            help='Loss balancing term.')
        parser.add_argument('--tau', type=float, default=1.,
                            help='Scaling factor used in gumbel softmax.')

        return parser


    @property
    def task(self) -> str:
        return 'clapv2'


class DownstreamConfigBase(ConfigBase):
    """Configurations for downstream tasks."""
    def __init__(self, args=None, **kwargs):
        super(DownstreamConfigBase, self).__init__(args, **kwargs)


class ClassificationConfig(DownstreamConfigBase):
    """Configurations for classification; {linear evaluation, finetuning, from scratch}."""
    def __init__(self, args=None, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('Linear evaluation of pre-trained model.', add_help=False)
        parser.add_argument('--labels', type=float, default=1.0,
                            help='Size of labeled data (0, 1].')
        parser.add_argument('--pretrained_file', type=str, default=None,
                            help='Path to pretrained model file (*.pth.tar).')
        parser.add_argument('--pretrained_task', type=str, default=None,
                            help='Type of pretraining task.')
        parser.add_argument('--finetune', action='store_true',
                            help='Enable backbone training.')
        return parser

    @property
    def task(self) -> str:
        if self.pretrained_file is not None:
            if self.pretrained_task is None:
                raise ValueError(
                    "Provide a proper name for the pretrained model type."
                    "supported names: simclr, moco, byol, clap"
                    )
            if self.finetune:
                return f"finetune_{self.pretrained_task}"
            else:
                return f"linear_{self.pretrained_task}"
        else:
            return "from_scratch"


class MixupConfig(ClassificationConfig):
    def __init__(self, args=None, **kwargs):
        super(MixupConfig, self).__init__(args, **kwargs)

    @property
    def task(self) -> str:
        return 'mixup'


class DetectionConfig(DownstreamConfigBase):
    def __init__(self, args=None, **kwargs):
        super(DetectionConfig, self).__init__(args, **kwargs)

    @property
    def task(self) -> str:
        return 'detection'
