import json
import os
import re
import time
from collections import Counter

# Optimized pre-tokenization: alphanumeric words (English & Hindi) with optional leading space | spaces | single chars
# Must be IDENTICAL to the pattern in tokenizer.py so merges are consistent
BILINGUAL_PATTERN = re.compile(
    r' ?[\u0900-\u097Fa-zA-Z0-9]+|\s+|[^\s]'
)

def get_stats(word_freqs):
    counts = {}
    for word_ids, freq in word_freqs.items():
        for pair in zip(word_ids, word_ids[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts

def merge_vocab(word_freqs, pair, new_id):
    new_word_freqs = {}
    p0, p1 = pair
    for word_ids, freq in word_freqs.items():
        if len(word_ids) <= 1:
            new_word_freqs[word_ids] = freq
            continue
        new_ids = []
        i = 0
        n = len(word_ids)
        while i < n:
            if i < n - 1 and word_ids[i] == p0 and word_ids[i+1] == p1:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(word_ids[i])
                i += 1
        new_word_freqs[tuple(new_ids)] = freq
    return new_word_freqs

def train_bpe(text, vocab_size):
    print(f"Training BPE with vocab_size={vocab_size} on ENTIRE corpus...")
    t0 = time.time()
    
    # Split text into words using bilingual-aware regex, count frequencies
    words = BILINGUAL_PATTERN.findall(text)
    word_counts = Counter(w.encode("utf-8") for w in words)
    print(f"  Unique words: {len(word_counts):,}")
    
    # Convert to dict of tuple of ints -> freq
    word_freqs = {tuple(w): count for w, count in word_counts.items()}
    
    num_merges = vocab_size - 256
    merges = {}
    
    for i in range(num_merges):
        stats = get_stats(word_freqs)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        new_id = 256 + i
        word_freqs = merge_vocab(word_freqs, best_pair, new_id)
        merges[best_pair] = new_id
        if (i + 1) % 100 == 0 or (i + 1) == num_merges:
            print(f"  Merge {i+1}/{num_merges}: {best_pair} -> {new_id}")
            
    t1 = time.time()
    print(f"BPE training took {t1 - t0:.2f}s")
    return merges

def encode(text, merges):
    words = BILINGUAL_PATTERN.findall(text)
    cache = {}
    res = []
    for w in words:
        w_bytes = w.encode("utf-8")
        if w_bytes not in cache:
            ids = list(w_bytes)
            for pair, new_id in merges.items():
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
            cache[w_bytes] = ids
        res.extend(cache[w_bytes])
    return res

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    res_bytes = b"".join(vocab[idx] for idx in ids if idx in vocab)
    return res_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    data_path = "../data/train_corpus.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Train BPE with vocab size 2048 on the ENTIRE corpus
    vocab_size = 2048
    merges = train_bpe(text, vocab_size)
    
    # Save merges to JSON
    # Convert tuple keys to strings for JSON
    serializable_merges = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    with open("bpe_merges.json", "w") as f:
        json.dump(serializable_merges, f)
        
    print("Saved merges to bpe_merges.json")
    
    # Test encoding/decoding on a sample
    test_text = text[:10000]
    encoded = encode(test_text, merges)
    decoded = decode(encoded, merges)
    assert decoded == test_text, "Decoding is not lossless!"
    print("Lossless test passed! Compression ratio:", len(test_text.encode("utf-8")) / len(encoded))
