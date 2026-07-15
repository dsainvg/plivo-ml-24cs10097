import json
import os
import time

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    p0, p1 = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i+1] == p1:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def train_bpe(text, vocab_size, max_train_bytes=500000):
    print(f"Training BPE with vocab_size={vocab_size} on first {max_train_bytes} bytes...")
    t0 = time.time()
    tokens = list(text[:max_train_bytes].encode("utf-8"))
    
    num_merges = vocab_size - 256
    merges = {} # (p0, p1) -> new_id
    
    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        new_id = 256 + i
        tokens = merge(tokens, best_pair, new_id)
        merges[best_pair] = new_id
        if (i + 1) % 100 == 0 or (i + 1) == num_merges:
            print(f"  Merge {i+1}/{num_merges}: {best_pair} -> {new_id} (remaining tokens: {len(tokens)})")
            
    t1 = time.time()
    print(f"BPE training took {t1 - t0:.2f}s")
    return merges

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        tokens = merge(tokens, pair, new_id)
    return tokens

def decode(ids, merges):
    # Inverse merges mapping
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        
    res_bytes = b"".join(vocab[idx] for idx in ids)
    return res_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    data_path = "../data/train_corpus.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Train BPE with vocab size 1024 on 500k bytes of text
    vocab_size = 1024
    merges = train_bpe(text, vocab_size, max_train_bytes=500000)
    
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
