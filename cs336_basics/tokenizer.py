from typing import Iterable
import json


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

        with open(merges_filepath, encoding='utf-8') as f:
            merges_str = json.load(f)
        merges = [(p1.encode('utf-8'), p2.encode('utf-8')) for p1, p2 in merges_str]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        text_bytes = text.encode('utf-8')
        # for b in text_bytes:
        # TODO
        return []


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory
        """
        # TODO
        return []


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        # TODO
        return ""
