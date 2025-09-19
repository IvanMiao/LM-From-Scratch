import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(w_tensor, mean=0.0, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(w_tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        pass


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # TODO
        return torch.zeros(1,1)

