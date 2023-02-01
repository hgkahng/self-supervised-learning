# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn


class Task(object):
    def __init__(self):
        self.checkpoint_dir = None

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self):
        raise NotImplementedError

    def load_history_from_checkpoint(self):
        raise NotImplementedError

    @staticmethod
    def move_optimizer_states(optimizer: torch.optim.Optimizer, device: str):
        for state in optimizer.state.values():  # dict; state of parameters
            for k, v in state.items():          # iterate over paramteters (k=name, v=tensor)
                if torch.is_tensor(v):          # If a tensor,
                    state[k] = v.to(device)     # configure appropriate device

    @staticmethod
    def freeze_params(net: nn.Module):
        with torch.no_grad():
            for p in net.parameters():
                p.requires_grad = False
