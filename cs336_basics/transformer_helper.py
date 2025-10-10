import torch
import einops


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
