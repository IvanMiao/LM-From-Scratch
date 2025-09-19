import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


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
        print(f"the shape of x in forward ({x.shape})")
        res = einops.einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')
        return res


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        """num_embeddings = vocab_size, embedding_dim = d_model"""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        w_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(w_tensor, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(w_tensor)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.W[token_ids]
