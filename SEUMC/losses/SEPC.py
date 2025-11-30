import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class SELoss4Clustering(torch.nn.Module):
    """
    The structure entropy for regularization.
    """

    def __init__(self, num_clusters=None, knn_k=10):
        super(SELoss4Clustering, self).__init__()
        self.act = nn.Sigmoid()
        self.knn_k = knn_k
        self.num_clusters = num_clusters

    def gaussian_distance_probability(self, logits_np, centers, sigma):
        # [Batch, 1, Dim] - [1, Clusters, Dim] -> [Batch, Clusters, Dim]
        # norm -> [Batch, Clusters]
        distances = np.linalg.norm(logits_np[:, np.newaxis] - centers, axis=2)
        exp_distances = np.exp(-distances ** 2 / (2 * sigma ** 2))
        probabilities = exp_distances / (np.sum(exp_distances, axis=1, keepdims=True) + 1e-10)
        return probabilities

    def dis2C(self, logits, sigma, num_clusters):
        logits_np = logits.detach().cpu().numpy()
        k = num_clusters if num_clusters is not None else self.num_clusters
        if k is None:
            raise ValueError("num_clusters must be provided either in __init__ or forward")

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(logits_np)
        centers = kmeans.cluster_centers_  # [num_clusters, feature_dim]
        probabilities = self.gaussian_distance_probability(logits_np, centers, sigma)
        return probabilities

    def forward(self, logits, sigma, num_clusters=None):
        S = logits @ logits.T
        current_k = min(self.knn_k, S.size(-1))

        topk = torch.topk(S, current_k, dim=-1)
        mask = torch.zeros_like(S)
        mask.scatter_(1, topk.indices, 1.0)
        S = S * mask
        A = self.act(S)

        row_sum = A.sum(dim=1, keepdim=True) + 1e-8
        adj = A / row_sum

        # C shape: [N, K], adj shape: [N, N]
        Y_ = self.dis2C(logits, sigma, num_clusters)
        C = torch.tensor(Y_, dtype=torch.float32).to(adj.device)
        IsumC = torch.ones_like(C.t()).to(adj.device)
        adj = adj - torch.diagflat(torch.diag(adj))
        Deno_sumA = 1.0 / (torch.sum(adj) + 1e-10)  # 防止除零
        Rate_p = (C.t().mm(adj.mm(C))) * Deno_sumA
        enco_p = (IsumC.mm(adj.mm(C))) * Deno_sumA
        Rate_p = enco_p - Rate_p
        encolen = torch.log2(enco_p + 1e-20)
        se_loss = torch.trace(Rate_p.mul(encolen))


        return -se_loss