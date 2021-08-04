# -*- coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import precision, recall


class MultiAccuracy(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAccuracy, self).__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):

        assert logits.ndim == 2, "(B, F)"
        assert labels.ndim == 1, "(B,  )"
        assert len(logits) == len(labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = torch.eq(preds, labels).float()

        return torch.mean(correct)


class TopKAccuracy(nn.Module):
    def __init__(self, k: int):
        super(TopKAccuracy, self).__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):

        assert logits.ndim == 2, "(B, F)"
        assert labels.ndim == 1, "(B,  )"
        assert len(logits) == len(labels)

        with torch.no_grad():
            preds = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(preds, self.k, dim=1)
            labels = labels.view(-1, 1).expand_as(topk_indices)  # (B, k)
            correct = labels.eq(topk_indices) * (topk_probs)     # (B, k)
            correct = correct.sum(dim=1).bool().float()          # (B, ) & {0, 1}

        return torch.mean(correct)


class MultiPrecision(nn.Module):
    def __init__(self, num_classes: int, average: str = 'macro'):
        super(MultiPrecision, self).__init__()
        self.num_classes = num_classes
        self.average = average

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        
        assert logits.ndim == 2, "(B, F)"
        assert labels.ndim == 1, "(B,  )"

        with torch.no_grad():
            if self.average == 'macro':
                return precision(
                    pred=nn.functional.softmax(logits, dim=1),
                    target=labels,
                    num_classes=self.num_classes,
                    reduction='elementwise_mean'
                )
            elif self.average == 'micro':
                raise NotImplementedError
            elif self.average == 'weighted':
                raise NotImplementedError
            else:
                raise ValueError


class MultiRecall(nn.Module):
    def __init__(self, num_classes: int, average: str = 'macro'):
        super(MultiRecall, self).__init__()
        self.num_classes = num_classes
        self.average = average

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            if self.average == 'macro':
                return recall(
                    pred=nn.functional.softmax(logits, dim=1),
                    target=labels,
                    num_classes=self.num_classes,
                    reduction='elementwise_mean',
                )
            elif self.average == 'micro':
                raise NotImplementedError
            elif self.average == 'weighted':
                raise NotImplementedError
            else:
                raise ValueError


class MultiF1Score(nn.Module):
    def __init__(self, num_classes: int, average: str = 'macro'):
        super(MultiF1Score, self).__init__()

        self.num_classes = num_classes
        self.average = average

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            if self.average == 'macro':
                f1_scores = torch.zeros(self.num_classes, device=logits.device)
                for c in range(self.num_classes):
                    pred = logits.argmax(dim=1).eq(c)
                    true = labels.eq(c)
                    f1 = BinaryFBetaScore.macro_f_beta_score(pred, true, beta=1)
                    f1_scores[c] = f1
                return torch.mean(f1_scores)
            elif self.average == 'micro':
                raise NotImplementedError
            elif self.average == 'weighted':
                raise NotImplementedError
            else:
                raise ValueError


class BinaryFBetaScore(nn.Module):
    def __init__(self, beta: float = 1., threshold: float = .5, average: str = 'macro'):
        super(BinaryFBetaScore, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.average = average

    @torch.no_grad()
    def forward(self, logit: torch.Tensor, label: torch.Tensor):
        assert logit.ndim == 1
        assert label.ndim == 1

        with torch.no_grad():
            pred = torch.sigmoid(logit)
            pred = pred > self.threshold   # boolean
            true = label > self.threshold  # boolean

            if self.average == 'macro':
                return self.macro_f_beta_score(pred, true, self.beta)
            elif self.average == 'micro':
                return self.micro_f_beta_score(pred, true, self.beta)
            elif self.average == 'weighted':
                return self.weighted_f_beta_score(pred, true, self.beta)
            else:
                raise NotImplementedError

    @staticmethod
    def macro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):

        assert true.ndim == 1
        assert pred.ndim == 1

        pred = pred.float()  # inputs could be boolean values
        true = true.float()  # inputs could be boolean values

        tp = (pred * true).sum().float()          # True positive
        _  = ((1-pred) * (1-true)).sum().float()  # True negative
        fp = ((pred) * (1-true)).sum().float()    # False positive
        fn = ((1-pred) * true).sum().float()      # False negative

        precision_ = tp / (tp + fp + 1e-7)
        recall_ = tp / (tp + fn + 1e-7)

        f_beta = (1 + beta**2) * precision_ * recall_ / (beta**2 * precision_ + recall_ + 1e-7)

        return f_beta

    @staticmethod
    def micro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError

    @staticmethod
    def weighted_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError


class BinaryF1Score(BinaryFBetaScore):
    def __init__(self, threshold=.5, average='macro'):
        super(BinaryF1Score, self).__init__(beta=1, threshold=threshold, average=average)


if __name__ == '__main__':

    targets = torch.LongTensor([2, 2, 0, 2, 1, 1, 1])
    predictions = torch.FloatTensor(
        [
            [1, 2, 7],  # 2
            [1, 3, 7],  # 2
            [3, 9, 0],  # 1
            [1, 2, 3],  # 2
            [3, 7, 0],  # 1
            [8, 1, 1],  # 0
            [9, 1, 1],  # 0
        ]
    )

    f1_function = MultiF1Score(num_classes=3, average='macro')
    f1_val = f1_function(logits=predictions, labels=targets)
    print(f1_val)
