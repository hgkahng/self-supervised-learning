# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import get_rich_pbar


class KNNEvaluator(object):
    def __init__(self,
                 num_neighbors: int,
                 num_classes: int,
                 temperature: float = 0.1):
        
        self.num_neighbors = num_neighbors
        self.num_classes = num_classes
        self.temperature = temperature

    @torch.no_grad()
    def predict(self,
                query: torch.FloatTensor,
                memory_bank: torch.FloatTensor,
                memory_labels: torch.LongTensor):

        k = self.num_neighbors
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
    def score(self,
              query: torch.FloatTensor,
              query_labels: torch.LongTensor,
              memory_bank: torch.FloatTensor,
              memory_labels: torch.LongTensor):

        pred_labels = self.predict(query, memory_bank, memory_labels)[:, 0]
        query_labels = query_labels.squeeze()
        num_correct = pred_labels.eq(query_labels).float().sum().item()

        return num_correct / len(query_labels)

    @torch.no_grad()
    def evaluate(self,
                 *nets,
                 query_loader: torch.utils.data.DataLoader,
                 memory_loader: torch.utils.data.DataLoader):
        """
        Evaluate model.
        Arguments:
            net: a `nn.Module` instance. Unlike the rest of the code, provide a
            whole `nn.Sequential` instance containing the CNN `backbone` and MLP `projector`.
            backbone: ...
            projector: ...
            query_loader: a `DataLoader` instance of test data. Apply
                 minimal data augmentation as used for testing for linear evaluation.
                 (i.e., Resize + Crop (0.875 x size), etc.)
            memory_loader: a `DataLoader` instance of train data. Apply
                 minimal augmentation as if used for training for linear evaluation.
                 (i.e., HorizontalFlip(0.5), etc.)
        """

        if len(nets) == 1:
            net = nets[0]
        else:
            net = nn.Sequential(*nets)
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
            for _, batch in enumerate(query_loader):
                z = net(batch['x'].to(device))
                z = F.normalize(z, dim=1)
                y = batch['y'].to(device)
                score = self.score(z, y, memory_bank, memory_labels)
                scores += [score]
                pg.update(task_2, desc=desc_2+f"{score*100:.2f}%", advance=1.)

        return sum(scores) / len(query_loader)
