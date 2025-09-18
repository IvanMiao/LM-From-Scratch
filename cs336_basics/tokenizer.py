from typing import Iterable
import json
import re


class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
            ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        self.special_pattern = ""
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "|".join(re.escape(st) for st in self.special_tokens)


    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
            ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, encoding='utf-8') as f:
            vocab_str = json.load(f)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_str.items()}

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) == 2:
                    token1, token2 = parts
                    token_tuple = (token1.encode('utf-8'), token2.encode('utf-8'))
                    merges.append(token_tuple)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        ids = []

        if self.special_pattern:
            parts = re.split(f"({self.special_pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if self.special_tokens and part in self.special_tokens:
                token_id = self.reversed_vocab[part.encode('utf-8')]
                ids.append(token_id)
            else:
                tokens = [bytes([b]) for b in part.encode('utf-8')]

                for p1, p2 in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == p1 and tokens[i+1] == p2:
                            new_tokens.append(p1 + p2)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens

                for token in tokens:
                    if token in self.reversed_vocab:
                        ids.append(self.reversed_vocab[token])
                    else:
                        raise ValueError(f"Token {token} not found in vocabulary")
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory
        """
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8')
        return text
