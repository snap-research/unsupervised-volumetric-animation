"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
import torch.nn as nn


class WhitenedEmbedding(nn.Module):
    """
    Wrapper around embedding module, that perform whitening on all embedding and only then query the embedding.
    """

    def __init__(self, num_embeddings, embedding_dim, eps=1e-3):
        """
        :param num_embeddings: number of embeddings, same as nn.Embedding
        :param embedding_dim: dimensionality of embedding, same as nn.Embedding
        :param eps: small epsilon for numerical stability of whitening
        """

        super(WhitenedEmbedding, self).__init__()
        self.eps = eps
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def get_random(self, num_random=1):
        """
        Generate embedding from standard normal distribution.
        :param num_random: number of embedding to generate
        :param device: device to place the embeddings

        :return: random embeddings[num_random, embedding_dim]
        """
        return torch.normal(0, 1, size=(num_random, self.embedding_dim), device=self.embedding.weight.device)

    def forward(self, video_id):
        """
        Return whitened embeddings

        :param video_id: id of the embedding [bs,]
        :return: whitened embeddings[bs, embedding_dim]
        """

        x = self.embedding(video_id)
        x_all = self.embedding.weight
        m = x_all.mean(0)
        xn = x - m
        xn_all = x_all - m

        f_cov = torch.mm(xn_all.permute(1, 0), xn_all) / (xn_all.shape[0] - 1)
        eye = torch.eye(self.embedding_dim, device=xn.device)
        f_cov_shrunk = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrunk), upper=False)[0]

        return xn @ inv_sqrt.permute(1, 0)
