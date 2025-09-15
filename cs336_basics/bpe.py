import os
import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer():
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        # Note: this is a very slow implementation
        for pair, new_index in self.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, 
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

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. initialize vocab
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    special_token_bytes = set()
    for tok in special_tokens:
        b = tok.encode('utf-8')
        vocab[next_id] = b
        special_token_bytes.add(b)
        next_id += 1

    # =========================================
    # 2. pre-tokenizaiton
    # token_counts:  key 以bytes表示的token， value 这个token的出现次数
    # =========================================
    token_counts: dict[bytes, int] = {}

    if special_tokens:
        special_pattern = "|".join(map(regex.escape, special_tokens))
        special_chunks = regex.split(f"({special_pattern})", text)
    else:
        special_chunks = [text]
    
    for chunk in special_chunks:
        if chunk in special_tokens:
            spe_token = chunk.encode('utf-8')
            token_counts[spe_token] = token_counts.get(spe_token, 0) + 1
        else:
            for match_seq in regex.finditer(PAT, chunk):
                token = match_seq.group().encode('utf-8')
                token_counts[token] = token_counts.get(token, 0) + 1

    # =========================================
    # 3 Construct word_seqs
    # word_seqs: key 以bytes表示的token， value 这个token的每个字节组成的整数列表
    # =========================================
    word_seqs: dict[bytes, list] = {}
    for tok_bytes, count in token_counts.items():
        if tok_bytes in special_token_bytes:
            continue
        word_seqs[tok_bytes] = list(tok_bytes)
    # print(f"word seqs: {word_seqs}")

    if len(vocab) > vocab_size:
        return (vocab, merges)

    # =========================================
    # 4. Main Loop
    # =========================================
    while len(vocab) < vocab_size:
        pair_counts: dict[tuple[int, int], int] = {}
        # 对每个seq，统计 pair 频次
        for tok_bytes, seq in word_seqs.items():
            if len(seq) < 2:
                continue
            freq: int = token_counts[tok_bytes]
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
            
        if not pair_counts:
            break
        
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        new_id = next_id
        if new_id >= vocab_size:
            break
        next_id += 1

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_word_seqs = {}
        for tok_bytes, seq in word_seqs.items():
            new_word_seqs[tok_bytes] = merge(seq, best_pair, new_id)
        word_seqs = new_word_seqs

    # print('-'*20)
    # print(f"NEW word seqs: {word_seqs}")
    # print(f"merges: {merges}")
    # vocab_new = {k: v for k, v in vocab.items() if k > 255}
    # print(f"vocab: {vocab_new}")
    return (vocab, merges)

# train_bpe("./test", 400, [])
