
import torch
import torch.nn.functional as F


def masked_mean(x: torch.FloatTensor, m: torch.BoolTensor, dim: int) -> torch.FloatTensor:
    m_sum = torch.clamp(m.sum(dim=dim), min=1.0)
    x_sum = torch.sum(x * m.float(), dim=dim)
    return x_sum.div(m_sum)


def masked_var(x: torch.FloatTensor, m: torch.BoolTensor, dim: int) -> torch.FloatTensor:
    m_sum = torch.clamp(m.sum(dim=dim), min=1.0)
    x_sum = torch.sum(x * m.float(), dim=dim)
    sq_x_sum = torch.sum(x.pow(2) * m.float(), dim=0)
    return sq_x_sum.div(m_sum) - x_sum.div(m_sum).pow(2)


def masked_softmax(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = -1) -> torch.FloatTensor:
    return F.softmax(x.masked_fill(~m, float('-inf')), dim=dim)
 