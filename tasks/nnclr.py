
import torch
import torch.nn as nn
import torch.functional as F


class NNCLRLoss(nn.Module):
    def __init__(self, num_neighbors: int = 1):
        super(NNCLRLoss, self).__init__()
        self.num_neighbors = num_neighbors
    
    def forward(self, anchor: torch.FloatTensor, key: torch.FloatTensor, support: torch.FloatTensor):
        """..."""
        similarities = torch.einsum('bf,fn->bn', *[key, support])                   # (B, N)
        sorting_idx = torch.argsort(similarities, dim=1)                            # (B, N)
        topk_idx = sorting_idx[:, self.num_neighbors].view(-1, self.num_neighbors)  # (B, k)
        logits = torch.einsum('bf,fn->bn', *[anchor, support])                      # (B, N)
        logits_pos = logits.gather(dim=1, index=topk_idx)                           # (B, k)
        raise NotImplementedError  # FIXME: after SimCLR implementation
