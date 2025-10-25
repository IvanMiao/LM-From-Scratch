import torch
import torch.nn as nn
from .transformer import PositonWise_FeedForward, Multihead_Self_Attention, RMSNorm
from .transformer import Embedding, Linear

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
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # x.shape: batch_size, sequence_length
        transformer_res = self.embedding(x)
        batch_size, seq_len, _ = transformer_res.shape
        token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        for layer in self.layers:
            transformer_res = layer(transformer_res, token_positions)
        
        transformer_res_norm = self.norm(transformer_res)
        logits = self.linear(transformer_res_norm)

        return logits
