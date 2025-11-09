"""BPE Algo"""

import os
import regex
from tqdm import tqdm
import json


# 用于 pre-tokenizaiton 的正则表达式，将原始文本分隔成初始的chunks。这里用的是GPT-2所使用的模式。
# (?:[sdmt]|ll|ve|re) 匹配常见的英文缩写，如's, 'll
# ?\p{L}+ | ?\p{N}+ 匹配一个或多个 Unicode 字母或数字，前面可能有一个空格
# ?[^\s\p{L}\p{N}]+ 匹配一个或多个非空格，非字母，非数字的字符，前面可能有一个空格
# \s+(?!\S) | \s+ 匹配一个或多个空白字符
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge(
    indices: list[int],
    pair: tuple[int, int],
    new_index: int
) -> list[int]:
    """Return `indices`(list of ints, representing the byte sequence of a token),
    but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """创建初始词汇表 vocab, 先储存0-255的整数-字节表示的映射, 然后从256开始为特殊token分配ID"""
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for tok in special_tokens:
        b = tok.encode('utf-8')
        vocab[next_id] = b
        next_id += 1

    return (vocab)


def get_token_data(
    special_tokens: list[str],
    text:str,
    token_data: dict[bytes, tuple[int, list[int]]]
) -> None:
    """
    pre-tokenization, 收集初始 token 的数据
    return: token_data
            key:    初始token的字节表示
            value:  一个tuple (count, seq),
                    count 为该token在文本中出现的次数,
                    seq 是该token每个字节对应的整数列表
    """

    if special_tokens:
        special_pattern = "|".join(map(regex.escape, special_tokens))
        all_chunks = regex.split(f"({special_pattern})", text)
    else:
        all_chunks = [text]

    for chunk in all_chunks:
        if chunk in special_tokens:
            token = chunk.encode('utf-8')
            count, seq = token_data.get(token, (0, list(token)))
            token_data[token] = (count + 1, seq)
        else:
            for match_seq in regex.finditer(PAT, chunk):
                token = match_seq.group().encode('utf-8')
                count, seq = token_data.get(token, (0, list(token)))
                token_data[token] = (count + 1, seq)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # with open(input_path, 'r', encoding='utf-8') as f:
    #     text = f.read()

    # 1. initialize vocab
    vocab = initialize_vocab(special_tokens)
    next_id = 256 + len(special_tokens)

    # =========================================
    # 2. pre-tokenizaiton
    # token_data: key     以bytes表示的token，
    #             value   一个 tupel, (这个token的出现次数, 这个token的每个字节组成的整数列表)
    # =========================================
    token_data: dict[bytes, tuple[int, list[int]]] = {}
    # token_data = get_token_data(special_tokens ,text)
    
    print("Reading and pre-tokenizing file ...")
    with open (input_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing file"):
            get_token_data(special_tokens, line, token_data)

    if len(vocab) > vocab_size:
        return (vocab, merges)

    # =========================================
    # 3. Main Loop
    # =========================================
    num_merges = vocab_size - len(vocab)
    print(f"Traninig begin! File path: {input_path}")
    # while len(vocab) < vocab_size:
    for i in tqdm(range(num_merges), desc="Training BPE"):
        pair_counts: dict[tuple[int, int], int] = {}
        # 对每个seq，统计 pair 频次
        for tok_bytes, (count, seq) in token_data.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # update the token_data dict with merge rules
        for tok_bytes in list(token_data.keys()):
            count, seq = token_data[tok_bytes]
            new_seq = merge(seq, best_pair, next_id)
            if len(new_seq) >= 2:
                token_data[tok_bytes] = (count, new_seq)
            else:
                del token_data[tok_bytes]

        next_id += 1

    # print('-'*20)
    # print(f"NEW word seqs: {word_seqs}")
    # print(f"merges: {merges}")
    # vocab_new = {k: v for k, v in vocab.items() if k > 255}
    # print(f"vocab: {vocab_new}")
    return (vocab, merges)


def train_bpe_tinystories():
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 62265
    vocab, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"])

    vocab_filepath = "tinystories_vocab.json"
    print(f"Saving vocabulary to {vocab_filepath}")
    # We need to decode bytes to strings to save as JSON.
    # We use 'latin-1' because it can represent any byte value,
    # preventing errors with bytes that aren't valid UTF-8.
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()} # NOTE: doubt for latin-1
    with open(vocab_filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

    # Save the merges to a file
    merges_filepath = "tinystories_merges.txt"
    print(f"Saving merges to {merges_filepath}")
    with open(merges_filepath, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            # Decode bytes to strings for writing to the text file
            f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")
    
    print("BPE training and saving for TinyStories complete.")


def train_bpe_expts_owt():
    pass


train_bpe_tinystories()
