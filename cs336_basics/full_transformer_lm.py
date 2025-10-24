import torch
import torch.nn as nn
from .transformer import PositonWise_FeedForward, Multihead_Self_Attention, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int
    ) -> None:
        super().__init__()
        self.attention = Multihead_Self_Attention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
        )
        self.feed_forward = PositonWise_FeedForward(d_model)
        self.feed_forward.dff = d_ff

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output = self.attention(attn_input, token_positions)
        y = x + attn_output

        ff_input = self.norm2(y)
        ff_output = self.feed_forward(ff_input)
        z = y + ff_output
        return z


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int
    ) -> None:
        super().__init__()
        pass
