"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
import json
import os
import re


class CachedBPETokenizer:
    # Optimized pre-tokenization regex:
    # 1. Alphanumeric words (English & Hindi) with optional leading space
    # 2. Spaces
    # 3. Individual punctuation/fallback characters
    PATTERN = re.compile(
        r' ?[\u0900-\u097Fa-zA-Z0-9]+|\s+|[^\s]'
    )

    def __init__(self, merges):
        self.merges = merges
        self.vocab_size = 256 + len(merges)
        self.cache = {}

    def encode(self, text):
        if not text:
            return []
        words = self.PATTERN.findall(text)
        res = []
        for w in words:
            w_bytes = w.encode("utf-8")
            if w_bytes not in self.cache:
                ids = list(w_bytes)
                for pair, new_id in self.merges.items():
                    p0, p1 = pair
                    if p0 not in ids or p1 not in ids:
                        continue
                    i = 0
                    while i < len(ids) - 1:
                        if ids[i] == p0 and ids[i+1] == p1:
                            ids[i] = new_id
                            del ids[i+1]
                        else:
                            i += 1
                self.cache[w_bytes] = ids
            res.extend(self.cache[w_bytes])
        return res

    def decode(self, ids):
        if not ids:
            return ""
        vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        res_bytes = b"".join(vocab[idx] for idx in ids if idx in vocab)
        return res_bytes.decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by evaluate.py. Replace as needed."""
    if path is None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "bpe_merges.json")
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            serializable_merges = json.load(f)
        merges = {}
        for k, v in serializable_merges.items():
            p0, p1 = map(int, k.split(","))
            merges[(p0, p1)] = v
        return CachedBPETokenizer(merges)
    else:
        # Fallback to byte-level tokenizer if merges json doesn't exist
        class ByteTokenizer:
            vocab_size = 256
            def encode(self, text):
                return list(text.encode("utf-8"))
            def decode(self, ids):
                return bytes(ids).decode("utf-8", errors="replace")
        return ByteTokenizer()

