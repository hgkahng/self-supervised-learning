# -*- coding: utf-8 -*-

import typing
import argparse
import wandb


def configure_wandb(name: str, project: str, config: argparse.Namespace, group: str = None):  # TODO: remove
    """
    Initialize wandb. Deprecated.
    Arguments:
        name: str, used to identify experiment.
        project: str, to which this experiment belongs.
        config: `argparse.Namespace` object holding experiment configurations.
    """

    wandb.init(
        project=project,
        group=group,
        name=name,
        config=config,
        sync_tensorboard=True,
    )


def initialize_wandb(config: typing.Union[object, dict, argparse.Namespace]):
    """
    Initialize wandb.
    Arguments:
        config: `argparse.Namespace` object holding experiment configurations.
            For backward compatibility, basic python dictionaries are also allowed.
    """
    entity = 'ku-dmqa' if config.wandb_entity is None else config.wandb_entity
    project = f'CL - {config.data}' if config.wandb_project is None else config.wandb_project
    group = config.task if config.wandb_group is None else config.wandb_group
    job_type = 'dev' if config.wandb_job_type is None else config.wandb_job_type
    identifier = wandb.util.generate_id() if config.wandb_id is None else config.wandb_id
    name = config.hash if config.wandb_name is None else config.wandb_name
    config_exclude_keys = [k for k, _ in config.__dict__.items() if k.startswith('wandb_')]
    
    wandb.init(
        project=project,
        group=group,
        name=name,
        job_type=job_type,
        entity=entity,
        id=identifier,
        config=config,
        config_exclude_keys=config_exclude_keys
    )
