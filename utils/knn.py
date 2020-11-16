# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import get_rich_pbar


class KNNEvaluator(object):
    def __init__(self,
                 num_neighbors: int or list,
                 num_classes: int,
                 temperature: float = 0.1):
        
        if isinstance(num_neighbors, int):
            self.num_neighbors = [num_neighbors]
        elif isinstance(num_neighbors, list):
            self.num_neighbors = num_neighbors
        else:
            raise NotImplementedError
        self.num_classes = num_classes
        self.temperature = temperature

    @torch.no_grad()
    def predict(self,
                k: int,
                query: torch.FloatTensor,
                memory_bank: torch.FloatTensor,
                memory_labels: torch.LongTensor):

        C = self.num_classes
        T = self.temperature
        B, _ = query.size()

        # Compute cosine similarity
        sim_matrix = torch.einsum('bf,fm->bm', [query, memory_bank])       # (b, f) @ (f, M) -> (b, M)
        sim_weight, sim_indices = sim_matrix.sort(dim=1, descending=True)  # (b, M), (b, M)
        sim_weight, sim_indices = sim_weight[:, :k], sim_indices[:, :k]    # (b, k), (b, k)
        sim_weight = (sim_weight / T).exp()                                # (b, k)
        sim_labels = torch.gather(
            memory_labels.expand(B, -1),                                   # (1, M) -> (b, M)
            dim=1,
            index=sim_indices
        )                                                                  # (b, M)

        one_hot = torch.zeros(B * k, C, device=sim_labels.device)       # (bk, C)
        one_hot.scatter_(dim=-1, index=sim_labels.view(-1, 1), value=1) # (bk, C) <- scatter <- (bk, 1)
        pred = one_hot.view(B, k, C) * sim_weight.unsqueeze(dim=-1)     # (b, k, C) * (b, k, 1)
        pred = pred.sum(dim=1)                                          # (b, C)

        return pred.argsort(dim=-1, descending=True)                    # (b, C); first column gives label of highest confidence

    @torch.no_grad()
    def evaluate(self,
                 net,
                 memory_loader: torch.utils.data.DataLoader,
                 query_loader: torch.utils.data.DataLoader):
        """
        Evaluate model.
        Arguments:
            net: a `nn.Module` instance.
            memory_loader: a `DataLoader` instance of train data. Apply
                 minimal augmentation as if used for training for linear evaluation.
                 (i.e., HorizontalFlip(0.5), etc.)
            query_loader: a `DataLoader` instance of test data. Apply
                 minimal data augmentation as used for testing for linear evaluation.
                 (i.e., Resize + Crop (0.875 x size), etc.)
        """

        net.eval()
        device = next(net.parameters()).device

        with get_rich_pbar(transient=True, auto_refresh=True) as pg:

            desc_1 = "[bold yellow] Extracting features..."
            task_1 = pg.add_task(desc_1, total=len(memory_loader))
            desc_2 = f"[bold cyan] {self.num_neighbors}-NN score: "
            task_2 = pg.add_task(desc_2, total=len(query_loader))

            # 1. Extract memory features (train data to compare against)
            memory_bank, memory_labels = [], []
            for _, batch in enumerate(memory_loader):
                z = net(batch['x'].to(device, non_blocking=True))
                memory_bank += [F.normalize(z, dim=1)]
                memory_labels += [batch['y'].to(device)]
                pg.update(task_1, advance=1.)

            memory_bank = torch.cat(memory_bank, dim=0).T
            memory_labels = torch.cat(memory_labels, dim=0)

            # 2. Extract query features (test data to evaluate) and
            #  and evalute against memory features.
            scores = []
            for k in self.num_neighbors:
                total_correct = 0
                for _, batch in enumerate(query_loader):
                    z = net(batch['x'].to(device))
                    z = F.normalize(z, dim=1)
                    y = batch['y'].to(device)
                    y_pred = self.predict(k,
                                          query=z,
                                          memory_bank=memory_bank,
                                          memory_labels=memory_labels)[:, 0].squeeze()
                    total_correct += y.eq(y_pred).sum().item()
                    pg.update(task_2, advance=1.) 
                score = total_correct / len(query_loader.dataset)
                scores += [score]

        if len(scores) == 1:
            return scores[0]
        return scores
