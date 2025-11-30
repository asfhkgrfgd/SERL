import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_loss_from_probs(probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log(probs + 1e-12)  # Shape: (N, C)
    entropy = -torch.sum(probs * log_probs, dim=1)  # Shape: (N,)
    return entropy.mean()


def structure_loss_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    num_classes = probs.size(1)
    device = probs.device

    mean_probs = torch.zeros(num_classes, device=device)

    for j in range(num_classes):
        mask = (labels == j)  # shape (N,)
        if mask.sum() > 0:
            mean_probs[j] = probs[mask, j].mean()
        else:
            mean_probs[j] = 0.0

    mean_probs = mean_probs + 1e-12

    entropy = -torch.sum(mean_probs * torch.log(mean_probs))

    return entropy


def structure_entrop_from_sim(logits: torch.Tensor,
                              labels: torch.Tensor,
                              num_classes: int) -> torch.Tensor:

    act = nn.Sigmoid()
    adj = act(logits @ logits.T)

    partition = F.one_hot(labels, num_classes=num_classes).to(
        dtype=adj.dtype, device=adj.device)

    C = partition.float()
    IsumC = torch.ones_like(C.t())
    adj = adj - torch.diagflat(torch.diag(adj))

    Deno_sumA = 1 / torch.sum(adj)

    Rate_p = (C.t() @ (adj @ C)) * Deno_sumA
    enco_p = (IsumC @ (adj @ C)) * Deno_sumA

    Rate_p = enco_p - Rate_p
    encolen = torch.log2(enco_p + 1e-20)

    se_loss = -torch.trace(Rate_p.mul(encolen))
    return se_loss


def structure_entropy_reg(logits: torch.Tensor,
                          labels: torch.Tensor,
                          num_classes: int,
                          knn_k=10) -> torch.Tensor:

    act = nn.Sigmoid()
    S = logits @ logits.T
    knn_k = min(knn_k, S.size(-1))
    topk = torch.topk(S, knn_k , dim=-1)  #
    mask = torch.zeros_like(S)
    mask.scatter_(1, topk.indices, 1.0)
    S = S * mask
    A = act(S)
    row_sum = A.sum(dim=1, keepdim=True) + 1e-8
    adj = A / row_sum
    partition_ = F.one_hot(labels, num_classes=num_classes).to(
        dtype=adj.dtype, device=adj.device)
    N = partition_.shape[0]  # number of node
    C = partition_.float().to(adj.device)
    probs = adj.mm(C)  # Shape: (N, C)
    log_probs = torch.log(probs / N + 1e-12)  # Shape: (N, C)
    structural_entropy = -torch.sum((1.0 - C) * probs * log_probs, dim=0) / N  # Shape: (,C)
    return structural_entropy.sum()
