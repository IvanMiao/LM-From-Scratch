import torch
import torch.nn as nn
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
        print(f"the shape of x in forward is ({x.shape})")
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

        # the weight here is a look-up dictionary, but learnable
        w_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(w_tensor, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(w_tensor)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.W[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization(RMS Norm)"""
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Calculate Root Mean Square for each element in the last dimension(d_model)
        rms = torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True) + self.eps)
        normalized_x = (x / rms) * self.W

        return normalized_x.to(in_dtype)


# SwiGLU_feedforward：Linear -> SiwGLU(SiLU/Swish + GLU) -> Linear
class PositonWise_FeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            device = None,
            dtype = None
    ) -> None:
        super().__init__()
        dff = int((8/3) * d_model)
        self.dff = ((dff + 64 - 1) // 64 ) * 64  # the dim of the inner feed-forward layer is a multiple of 64
        self.w1 = Linear(d_model, self.dff, device=device, dtype=dtype)
        self.w2 = Linear(self.dff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.dff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor):
        silu = self.w1(x) * torch.sigmoid(self.w1(x))  # a sigmoid function
        data_value = self.w3(x)
        gated_output = silu * data_value
        swiglu_output = self.w2(gated_output)
        return swiglu_output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even!"

        # Precompute theta_i = theta^(-2i/d_k) for i in [0, 2, ..., d_k-2]
        theta_i = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k)
        m_theta = torch.outer(torch.arange(max_seq_len, device=device), theta_i)

        cos_cached = torch.cos(m_theta)
        sin_cached = torch.sin(m_theta)

        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token position shape: (..., seq_len)

        cos = self.cos_cached[token_positions]  # type: ignore
        sin = self.sin_cached[token_positions]  # type: ignore

        x_paired = einops.rearrange(x, '... (d p) -> ... d p', p=2)
        x1, x2 = x_paired[..., 0], x_paired[..., 1]

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        y_paired = torch.stack((y1, y2), dim=-1)

        return einops.rearrange(y_paired, '... d p -> ... (d p)')


# FUNCTION
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    apply softmax to the i-th dimension of the input tensor.
        x: the input tensor
        i: dimension
    
    To avoid numerical overflow(inf/inf -> NaN),
    we use a constant to subtract all inputs,
    which will not affect the result of softmax function
    """

    max_val, _ = torch.max(x, dim=i, keepdim=True)
    x_shifted = x - max_val

    x_exp = torch.exp(x_shifted)
    sum_exp = torch.sum(x_exp, dim=i, keepdim=True)

    return x_exp / sum_exp


# FUNCTION
# softmax((q @ k.T) / sqrt(d_k) + mask) @ v
def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
)-> torch.Tensor:
    """
    Implements the scaled dot-product attention mechanism
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., se_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)

    Output:
        The output of the attention mechanism, with shape (..., seq_len_q, d_v)
    """
    d_k = q.shape[-1]

    attention_scores = einops.einsum(q, k, '... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k')
    scaled_scores = attention_scores / (d_k ** 0.5)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask==False, -1e9)

    attention_weights = softmax(scaled_scores, i=-1)
    output = einops.einsum(attention_weights, v, '... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v')
    return output


class Multihead_Self_Attention(nn.Module):
    def __init__(
            self, d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float = 10000.0,
            use_rope: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len
            )
        else:
            self.rope = None

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)


    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        # token_position: (batch_size, seq_len)
        batch_size, seq_len, _ = x.shape

        # 1. Project to K, Q ,V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into multiple heads
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        # 3. Apply RoPE to Q and K (if use_rope is True)
        # The head dimension is treated as a batch dimension for RoPE
        if self.rope is not None:
            assert token_position is not None, "token_position must be provided when using RoPE"
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)

        # 4. Create causal mask
        # This prevent attention to future tokens
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        causal_mask = ~mask

        # 5. Scaled dot product attention
        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 6. Concatenate heads and apply final projection
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = einops.rearrange(attention_output, 'b h s d -> b s (h d)')

        return self.o_proj(output)
