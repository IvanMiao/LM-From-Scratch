## 2 BPE Tokenizer

### 2.1 Unicode

(a) What Unicode character does `chr(0)` return?
`'\x00'`

(b) How does this character's string representation (__repr__()) differ from its printed representation?
`chr(0).__repr__()` -> `"'\\x00'"`
`print(chr(0))` -> `(nothing)`

(C) 
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

`chr(0)` -> `'\x00'`
`print(chr(0))` -> `(nothing)`
`"this is a test" + chr(0) + "string"` -> `'this is a test\x00string'`
`print("this is a test" + chr(0) + "string")` -> `this is a teststring`

This character will be ignored when calling `print()` to print it.

### 2.2 Unicode Encodings

(a) It's more economic when using UTF-8, since UTF-16 always use more than 2 bytes and UTF-32 4 bytes.

(b) Incorrect function:
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Suppose: The function will decode each single byte, but thers are some characters in UTF-8 are encoded by >=2 bytes.

(c) Two byte sequence that does not decode to any Unicode character(s):
#1 0xff, 0xff

#2 0xff, 0xff, 0xff

### 2.3 Subword Tokenization


### 2.4 BPE Tokenizer Training


### 2.5 Experimenting with BPE Tokenizer Training

- function `train_bpe` in [bpe.py](../cs336_basics/bpe.py)

- function `train_bpe_tinystories`
	- (a) without using `multiprocessing`, it would cost about 18 min. After applying that, [TODO]
	- (b) [TODO]

- function `train_bpe_expts_owt`
	- (a) [TODO]
	- (b) [TODO]


### 2.6 BPE Tokenizer: Encoding and Decoding


### 2.7 Experiments

(a)

(b)

(c)

(d)

## 3 Transformer Language Model Architecture


### 3.1


### 3.2 output normalization and embedding


### 3.3 Remark: Batching, Einsum and Eﬀicient Computation

pytorch -> einops


## 4

### 4.3 AdamW

为了方便表示，我们定义以下变量：
*   `B`: `batch_size`
*   `L`: `context_length` (序列长度)
*   `V`: `vocab_size`
*   `N`: `num_layers`
*   `D`: `d_model`
*   `H`: `num_heads`
*   `d_ff`: `4 * D` (根据题目假设)

1. 参数 (Parameters)

这是模型本身所占用的内存，由所有可学习的权重决定。

*   **Token Embedding**: `Embedding(V, D)` 的权重矩阵 `W`。
    *   尺寸: `V * D`
*   **Transformer Blocks** (`N` 个): 每个块包含：
    *   **Multi-head Self-Attention**:
        *   `q_proj`, `k_proj`, `v_proj`, `o_proj` 都是 `Linear(D, D)`。
        *   尺寸: `4 * (D * D)`
    *   **RMSNorm**: `norm1` 和 `norm2` 的权重 `W`。
        *   尺寸: `2 * D`
    *   **Position-Wise Feed-Forward**:
        *   `w1` 是 `Linear(D, d_ff)`，`w2` 是 `Linear(d_ff, D)`，`w3` 是 `Linear(D, d_ff)`。
        *   尺寸: `(D * d_ff) + (d_ff * D) + (D * d_ff) = 3 * D * d_ff`
        *   代入 `d_ff = 4 * D`: `3 * D * (4 * D) = 12 * D^2`
    *   每个块的总参数: `4*D^2 + 2*D + 12*D^2 = 16*D^2 + 2*D`
*   **Final RMSNorm**: `norm` 的权重 `W`。
    *   尺寸: `D`
*   **Output Embedding (LM Head)**: `Linear(D, V)` 的权重 `W`。
    *   尺寸: `D * V`

**总参数表达式**:
```
P = V*D (embedding) + N * (16*D^2 + 2*D) (layers) + D (final norm) + D*V (lm_head)
P ≈ 2*V*D + 16*N*D^2
```
(忽略了 `N*2*D` 和 `D` 等低阶项)

2. 梯度 (Gradients)

在反向传播后，每个参数都会有一个对应的梯度张量，其形状与参数完全相同。

**总梯度表达式**:
```
G = P
G ≈ 2*V*D + 16*N*D^2
```

3. 优化器状态 (Optimizer State)

AdamW 为每个参数维护两个状态：一阶矩 (`m`, `exp_avg`) 和二阶矩 (`v`, `exp_avg_sq`)。这两个状态的形状与参数相同。

**总优化器状态表达式**:
```
O = 2 * P
O ≈ 2 * (2*V*D + 16*N*D^2) = 4*V*D + 32*N*D^2
```

4. 激活 (Activations)

这是最复杂的部分，因为激活值是在前向传播过程中产生的，并且只有在计算梯度需要它们之前才需要保留在内存中。我们计算峰值内存，这通常发生在最后一个 Transformer 块的计算中。

*   **输入**: 词嵌入的输出。
    *   尺寸: `B * L * D`
*   **Transformer Blocks** (`N` 个):
    *   **`norm1`**: RMSNorm 的输出。
        *   尺寸: `B * L * D`
    *   **MHA**:
        *   Q, K, V 投影输出: `3 * (B * L * D)`
        *   `Q @ K.T` (注意力分数): `B * H * L * L`
        *   `softmax` 输出 (注意力权重): `B * H * L * L`
        *   加权 V (上下文向量): `B * L * D`
        *   输出投影: `B * L * D`
    *   **残差连接后**: `x + attn_output`
        *   尺寸: `B * L * D`
    *   **`norm2`**: RMSNorm 的输出。
        *   尺寸: `B * L * D`
    *   **FFN**:
        *   `w1` 和 `w3` 的输出: `2 * (B * L * d_ff) = 2 * B * L * 4 * D = 8 * B * L * D`
        *   SiLU 输出: `B * L * d_ff = 4 * B * L * D`
        *   `w2` 输出: `B * L * D`
    *   **残差连接后**: `y + ff_output`
        *   尺寸: `B * L * D`
*   **Final RMSNorm**: 输出。
    *   尺寸: `B * L * D`
*   **Output Embedding**: Logits。
    *   尺寸: `B * L * V`
*   **Cross-Entropy**:
    *   `log_softmax` 的输出 (用于计算损失)。
    *   尺寸: `B * L * V`

峰值激活内存约等于所有这些组件在计算图中最深处同时存在的最大值。一个合理的近似是注意力分数矩阵（通常是最大的）加上沿途传递的 `(B, L, D)` 形状的张量。

**总激活表达式 (近似)**:
```
A ≈ B*H*L^2 (attention scores) + B*L*V (logits) + (一些常数) * B*L*D
A ≈ B*H*L^2 + B*L*V
```
(忽略多个 `B*L*D` 和 `B*L*d_ff` 的张量，因为它们通常比注意力分数和 logits 小)

**总结**

将所有部分乘以 4 (字节/float32)，总峰值内存 `M` 为：

*   **Parameters**: `4 * (2*V*D + 16*N*D^2)`
*   **Gradients**: `4 * (2*V*D + 16*N*D^2)`
*   **Optimizer State**: `4 * (4*V*D + 32*N*D^2)`
*   **Activations**: `4 * (B*H*L^2 + B*L*V)`

**总内存 (Total)**:
`M = 4 * [ (2VD + 16ND^2) + (2VD + 16ND^2) + (4VD + 32ND^2) + (BHL^2 + BLV) ]`
`M = 4 * [ 8*V*D + 64*N*D^2 + B*H*L^2 + B*L*V ]`
