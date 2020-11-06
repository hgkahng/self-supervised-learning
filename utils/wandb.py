# -*- coding: utf-8 -*-

import wandb


def configure_wandb(name: str, project: str, config: object):
    wandb.init(
        name=name,
        project=project,
        config=config.__dict__,
        sync_tensorboard=True,
    )
