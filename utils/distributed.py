# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist


class DistributedGather(torch.autograd.Function):
    """
    Gather tensors from all processes. Supports backpropagation.
    Implementation borrowed from:
        https://github.com/open-mmlab/OpenSelfSup/
    """
    @staticmethod
    def forward(ctx, input):  # pylint: disable=redefined-builtin
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors  # pylint: disable=redefined-builtin
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class ForMoCo(object):
    """Functions for distributed training of MoCo."""

    @classmethod
    @torch.no_grad()
    def batch_shuffle_single_gpu(cls, x: torch.FloatTensor):
        idx_shuffle = torch.randperm(x.size(0)).to(x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @classmethod
    @torch.no_grad()
    def batch_shuffle_ddp(cls, x: torch.FloatTensor):
        """Add function docstring."""

        if not dist.is_initialized():
            return cls.batch_shuffle_single_gpu(x)

        local_rank = x.device
        local_batch_size = x.size(0)

        x_gathered = concat_all_gather(x)

        global_batch_size = x_gathered.size(0)
        num_nodes = global_batch_size // local_batch_size

        # Randomly shuffle indices
        idx_shuffle = torch.randperm(global_batch_size).to(local_rank)

        # Broadcast to all nodes
        dist.broadcast(idx_shuffle, src=0)

        # Keep indices for restoring the original order
        idx_unshuffle = torch.argsort(idx_shuffle)

        # Shuffle index for this node
        dist_rank = dist.get_rank()  # globally unique rank
        idx_shuffle_local = idx_shuffle.view(num_nodes, -1)[dist_rank]

        return x_gathered[idx_shuffle_local], idx_unshuffle

    @classmethod
    @torch.no_grad()
    def batch_unshuffle_single_gpu(cls, x: torch.FloatTensor, idx_unshuffle: torch.LongTensor) -> torch.FloatTensor:
        """Undo batch shuffle."""
        return x[idx_unshuffle]

    @classmethod
    @torch.no_grad()
    def batch_unshuffle_ddp(cls, x: torch.FloatTensor, idx_unshuffle: torch.LongTensor) -> torch.FloatTensor:
        """Add function docstring."""

        if not dist.is_initialized():
            return cls.batch_unshuffle_single_gpu(x, idx_unshuffle)

        local_batch_size = x.size(0)

        x_gathered = concat_all_gather(x)
        global_batch_size = x_gathered.size(0)

        num_nodes = global_batch_size // local_batch_size
        dist_rank = dist.get_rank()
        idx_unshuffle_local = idx_unshuffle.view(num_nodes, -1)[dist_rank]

        return x_gathered[idx_unshuffle_local]


def concat_all_gather(tensor: torch.Tensor, dim=0):
    """Gather tensors distributed across multiple processes."""

    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    buffer = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(buffer, tensor, async_op=False)
    return torch.cat(buffer, dim=dim)
