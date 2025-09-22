# Transformer Language Model Architecture

## Overview

inputs -> Token embedding -> (num_layers) * Transformer Block -> Norm -> Linear(Output embedding or LM head) -> Softmax -> Output Probalilities

**A pre-norm Transformer block**

input tensor with shape (batch_size, seq_len, d_model)
	-> Norm -> Causal Multi-Head Self-Attention w/ RoPE ->
	-> Add
		-> Norm -> Position-Wise Feed-Forward ->
		-> Add
			-> Output tensor with shape (batch_size, seq_len, d_model)

Each block aggregates information across the sequence(via self-attention) and non-linearly transforms it (via the feed-forward layers)



## Token Embedding 

class `Embedding`

词嵌入（Token Embedding）层的核心任务是将输入的文本“ID序列”转换为“向量序列”，为 Transformer 模型后续的计算做准备。

#### **1. 为什么需要 Embedding？**

*   计算机无法直接理解 "猫"、"爱"、"吃鱼" 这样的词语。它们只能处理数字。我们需要一种方法，将每个词或“token”（词元）映射到一个固定维度的、稠密的浮点数向量（vector）。
*   **目标**:
    将离散的 token 转换为连续的向量。让模型在训练过程中学习这些向量，使得意思相近的词（如“国王”和“女王”）在向量空间中的位置也相近。
*   **实现方式**: 创建一个巨大的“查找表”（Lookup Table）。这个表就是一个矩阵，**行数是词汇表的大小（`vocab_size`），列数是每个词向量的维度（`d_model`）**。每一行就代表一个 token 对应的向量。

#### **2. 分步详解**

**`__init__` 构造函数：初始化查找表**

这个函数只在创建 `Embedding` 类的实例时运行一次，负责“搭建”查找表。

*   `class Embedding(nn.Module)`:
    *   定义一个名为 `Embedding` 的类，并继承 `torch.nn.Module`。这是所有 PyTorch 模型或层的标准做法，它能自动实现反向传播（梯度计算）等功能。
*   `num_embeddings: int`:
    *   即 `vocab_size`，词汇表中的 token 总数。如果词汇表有 50000 个 token，那么这个值就是 50000。它决定了查找表的**行数**。
*   `embedding_dim: int`:
    *   即 `d_model`，每个 token 要被转换成的向量的维度。在 Transformer 论文中，这个值通常是 512 或 768。它决定了查找表的**列数**。

**关键代码解释：**

1.  **`w_tensor = torch.empty(...)`**:
    *   创建一个形状为 `(num_embeddings, embedding_dim)` 的张量。
    *   `torch.empty` 创建的张量值是未初始化的，里面是内存中的随机值，所以下一步必须进行初始化。

2.  **`nn.init.trunc_normal_(...)`**:
    *   对 `w_tensor` 进行“截断正态分布”（Truncated Normal Distribution）初始化。
    *   **为什么不直接用随机数或者全零？**: 好的权重初始化对模型训练至关重要。截断正态分布可以确保初始化的向量值被限制在一个合理的范围内（这里是-3到3）。过大或过小的初始值可能会导致训练初期的梯度爆炸或消失，使模型难以收敛。另外，相比于全零初始化，随机初始化打破了对称性，保证了每个 token 的初始向量都是独一无二的，为模型学习不同的语义特征提供了可能性。
    *   `mean=0, std=1`: 从均值为0，标准差为1的正态分布中采样。
    *   `a=-3, b=3`: 截断范围。任何采样出的值如果小于-3或大于3，就会被丢弃并重新采样。这可以防止极端值的出现。

3.  **`self.W = nn.Parameter(w_tensor)`**:
    *   **这是最关键的一步！**
    *   `nn.Parameter` 是一个特殊的类，它告诉 PyTorch：“这个张量 `w_tensor` 是模型的一部分，它的数值**需要在训练过程中通过反向传播和优化器不断被更新和优化**。”
    *   没有 `nn.Parameter` 的包装，`self.W` 就只是一个普通的张量，其值在训练中不会改变，也就无法学习到任何词语的语义信息。

**`forward` 前向传播函数：执行查找**

这个函数定义了当数据输入该层时，应该如何进行计算。

```python
def forward(self, token_ids: torch.Tensor):
    return self.W[token_ids]
```

这行代码虽然简洁，但背后是 PyTorch 强大的张量索引（Tensor Indexing）能力。它**不是通过一个一个的循环去查找**，而是一个高度并行化的“收集”（gather）操作。它会根据 `token_ids` 张量中所有 ID 的值，同时从 `self.W` 矩阵中取出对应的行向量，并按照 `token_ids` 的原始布局重新组合成一个新的、更高维度的张量。

*   **维度的变化是关键**
    这个操作导致了数据维度的重要变化，这是理解 Transformer 数据流的第一步：
    *   **输入 `token_ids` 的形状**: `[批次大小 (Batch Size), 序列长度 (Sequence Length)]`
    *   **输出 `embeddings` 的形状**: `[批次大小, 序列长度, 嵌入维度 (Embedding Dim)]`

    这个新增的 `embedding_dim` 维度，就是模型真正开始进行数学计算的基础。后续所有的自注意力(self-attention) 、前馈网络(feed-forward network)等层，都是在这个三维张量的基础上进行的。

## Transformer Block

### RMS Norm

RMS Norm 是对标准 LayerNorm（归一化层） 的一种简化和优化。在 Transformer 中，归一化层的作用是一个“稳定器”，防止模型训练过程中的梯度爆炸或消失，从而加速收敛并提升性能。RMSNorm 的设计哲学是：去掉标准归一化中可能不那么必要的部分（均值中心化），只保留最核心的缩放（Scaling）部分，从而在保证效果的同时提升计算效率。

传统LayerNorm 的两步操作：
1. 中心化 Re-centering： 将输入向量减去其均值，使其均值为0
2. 缩放 Re-scaling: 将结果除以其标准差，使其方差为1

RMSNorm 的作者通过实验发现，“中心化”这一步对性能的贡献不大，但却消耗了计算资源。因此，他们大胆地将其移除，只保留了“缩放”这一步。但它不是用“标准差”来缩放，而是用一个更简单的统计量——均方根 (Root Mean Square, RMS)。

$$
\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} \cdot g_i
$$

$$
\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}
$$

$g_i$ 是一个可学习的参数，而 $\varepsilon$ 是一个通常固定为 $1e-5$ 的超参数。


```python
# 核心公式
rms = torch.sqrt(torch.mean(x**2, dim=(-1), keepdim=True) + self.eps)
normalized_x = x / rms
```

假设输入 x 的 shape为 (batch_size, seq_len, d_model):
- `x**2`: 对输入的每个元素求平方
- `torch.mean(..., dim=(-1), keepdim=True)`: 沿着最后一个维度(d_model) 求均值。这意味着对 seq 中每一个token向量，都独立计算其所有特征(d_model个) 的平方均值
- `keepdim=True`: 确保计算结果的维度仍然是 (batch_size, seq_len, d_model) 而不是 (batch_size, seq_len), 保持这个 "1" 的维度是为了下一部的广播
- `torch.sqrt(...)`: 开方，得到RMS值
- `x/rms`: **广播机制**。 shape 为 `[batch_size, seq_len, d_model]` 的 `x` 会被 shape 为 `[batch_size, seq_len, 1]` 的 `rms` 相除。PyTorch 会自动将 `rms` 的值沿着最后一个维度复制 `d_model` 次，使得每个 token 向量中的所有元素都除以该 token 自身的 RMS 值。
- 最终，`normalize_x` 中的每一个 token 向量的RMS值都近似为1。

单纯的归一化操作虽然稳定了数据，但也带来一个问题：它强制性地抹去了不同 token 向量之间以及向量内部不同特征之间的“幅度信息”，这可能会限制模型的表达能力。模型可能需要放大或缩小某些重要的特征。

```python
self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
# ... in forward ...
normalized_x = (x / rms) * self.W
```

`self.W` 即上面公式中的 $g_i$, 是一个shape为 (d_model) 的向量，被 `nn.Parameter` 包装，意味着它的值是可学习的。

`self.W` 初始化为1 (`torch.ones(...)`)，意味着训练开始时， `* self.W ` 这个操作什么也不做。

在训练中，模型会通过反向传播学习 `self.W` 中的每一个值。如果模型发现第 `i` 个特征非常重要，它就会让 `self.W[i]` 的值变大；反之，则可能让其变小。


### Positon-Wise Feed-Forward

在原始Transformer论文中，Transformer 块的 feed-forward network 用的是两个线性层 + 一个ReLU激活函数。

但是，现代大模型引入了门控机制 gating mechanism，使用的是 “SwiGLU” 激活函数。该激活函数合并了 SiLU(也叫Swish) 激活函数 和 GLU (Gated Linear Unit 线性门控单元)。

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}
$$

$\sigma$ 代表sigmoid函数

$$
\text{GLU}(x,W_1,W_2) = \sigma(W_1x) \otimes W_2(x)
$$

$\otimes$ 代表逐位相乘

将 SiLU 和 GLU 合并，便得到了SwiGLU —— 我们将用它来实现前馈网络FFN：

$$
\text{FFN}(x) = \text{SwiGLU}(x,W_1,W_2,W_3) = W_2(\text{SiLU}(W_1x) \otimes W_3x)
$$

在这里， $x \in \R^{d_{model}}, W1,W3 \in \R^{d_{ff} \times d_{model}}, W_2 \in \R^{d_{model} \times d_{ff}}$, 并且一般来说 $d_{ff} = \frac{8}{3}d_{model}$



