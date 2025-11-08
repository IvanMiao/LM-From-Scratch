import torch
import einops


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    Apply softmax to the i-th dimension of the input tensor.
    
    x: the input tensor
    i: dimension
    
    To avoid numerical overflow (inf/inf -> NaN),
    we use a constant to subtract all inputs,
    which will not affect the result of softmax function
    """

    max_val, _ = torch.max(x, dim=i, keepdim=True)
    x_shifted = x - max_val

    x_exp = torch.exp(x_shifted)
    sum_exp = torch.sum(x_exp, dim=i, keepdim=True)

    return x_exp / sum_exp


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


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: (B, V)
    targets: (B,)
    """

    # (B, V) -> (B, 1)
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    norm_logits = logits - max_logits # o_j - m

    log_denominator = torch.log(torch.sum(torch.exp(norm_logits), dim=-1, keepdim=True))

    # target: (B,) -> (B, 1)
    target_indices = targets.unsqueeze(1)
    log_numerator = torch.gather(norm_logits, -1, index=target_indices) # log(p_i)

    sample_loss = -(log_numerator - log_denominator)
    final_loss = torch.mean(sample_loss)
    return final_loss


def learning_rate_schedule(step, lr_max, lr_min, t_w, t_c):
    if step < t_w:
        return (step / t_w) * lr_max
    elif step >= t_w and step <= t_c:
        import math
        return (lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step - t_w) / (t_c - t_w) )))
    else:
        return lr_min


def gradient_clipping():
    pass
