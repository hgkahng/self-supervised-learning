# -*- coding: utf-8 -*-

import copy
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging import get_rich_pbar

class KNNEvaluator(object):
    """Evaluates representations based on the nearest neighbors algorithm."""
    def __init__(self,
                 num_neighbors: typing.Union[int, list, tuple],
                 num_classes: int,
                 temperature: float = 0.1):
        if isinstance(num_neighbors, int):
            self.num_neighbors = [num_neighbors]
        elif isinstance(num_neighbors, (list, tuple)):
            self.num_neighbors = list(num_neighbors)
        else:
            raise ValueError
        self.num_classes:   int = num_classes
        self.temperature: float = temperature

    @torch.no_grad()
    def evaluate(self,
                 net: nn.Module,
                 memory_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader) -> typing.Dict[str, float]:
        """
        Arguments:
            net: a `nn.Module` instance.
            memory_loader: a `DataLoader` instance of training data. Apply minimal
                augmentation as if used for training for linear evaluation.
                (i.e., HorizontalFlip(0.5), etc.)
            test_loader: a `DataLoader` instance of test data. Apply minimal
                data augmentation as used for testing for linear evaluation.
                (i.e., Resize + Crop (0.875 x size), etc.)
        """

        with get_rich_pbar(transient=True, auto_refresh=True) as pbar:

            # Evaluation mode (mind the batch norm)
            net.eval()

            device = next(net.parameters()).device

            # Separate progress bars for {feature extraction, lazy prediction}
            job_fe = pbar.add_task(":fire: KNN feature extraction", total=len(memory_loader))
            job_lp = pbar.add_task(":rocket: KNN prediction", total=len(test_loader))

            # Feature extraction
            memory_features, memory_labels = list(), list()
            for i, batch in enumerate(memory_loader):
                x = batch['x'].to(device, non_blocking=True)
                z = F.adaptive_avg_pool2d(net(x), output_size=1).squeeze()
                memory_features += [F.normalize(z, dim=1)]
                memory_labels += [batch['y'].to(device)]
                pbar.update(job_fe, advance=1.)
            memory_features = torch.cat(memory_features, dim=0).t().contiguous()
            memory_labels = torch.cat(memory_labels, dim=0)

            # Lazy prediction
            scores = dict()
            corrects = [0] * len(self.num_neighbors)
            for _, batch in enumerate(test_loader):
                x = batch['x'].to(device, non_blocking=True)
                z = F.normalize(F.adaptive_avg_pool2d(net(x), output_size=1).squeeze(), dim=1)
                y = batch['y'].to(device)
                for i, k in enumerate(self.num_neighbors):
                    y_pred = self.predict(k,
                                          query=z,
                                          memory_features=memory_features,
                                          memory_labels=memory_labels)[:, 0].squeeze()
                    corrects[i] += y.eq(y_pred).sum().item()
                pbar.update(job_lp, advance=1.)

            torch.cuda.empty_cache();
            for i, k in enumerate(self.num_neighbors):
                scores[k] = corrects[i] / len(test_loader.dataset)

            return scores

    @torch.no_grad()
    def predict(self,
                k: int,
                query: torch.FloatTensor,
                memory_features: torch.FloatTensor,
                memory_labels: torch.LongTensor):

        C = self.num_classes
        T = self.temperature
        B, _ = query.size()

        # Compute cosine similarity
        sim_matrix = torch.einsum('bf,fm->bm', [query, memory_features])   # (b, f) @ (f, M) -> (b, M)
        sim_weight, sim_indices = sim_matrix.sort(dim=1, descending=True)  # (b, M), (b, M)
        sim_weight, sim_indices = sim_weight[:, :k], sim_indices[:, :k]    # (b, k), (b, k)
        sim_weight = (sim_weight / T).exp()                                # (b, k)
        sim_labels = torch.gather(memory_labels.expand(B, -1),             # (1, M) -> (b, M)
                                  dim=1,
                                  index=sim_indices)                       # (b, M)

        one_hot = torch.zeros(B * k, C, device=sim_labels.device)          # (bk, C)
        one_hot.scatter_(dim=-1, index=sim_labels.view(-1, 1), value=1)    # (bk, C) <- scatter <- (bk, 1)
        pred = one_hot.view(B, k, C) * sim_weight.unsqueeze(dim=-1)        # (b, k, C) * (b, k, 1)
        pred = pred.sum(dim=1)                                             # (b, C)

        return pred.argsort(dim=-1, descending=True)                       # (b, C); first column gives label of highest confidence
