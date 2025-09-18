import torch
import torch.nn as nn


class Linear(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			device: torch.device | None = None,
			dtype: torch.dtype | None = None
	):
		pass


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# TODO
		return torch.zeros(1,1)


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

