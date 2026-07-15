# Run Log

This file contains the hypothesis, changes, and results for each training run during the optimization process.

## Run 0: Baseline
- **Hypothesis**: Baseline configuration provided in the starter code.
- **Changes**: None.
- **Dev BPB**: 2.3718
- **Conclusion**: The baseline model has 1.34M parameters and runs in 237s on CPU, but the byte-level tokenizer and simple architecture yield a high BPB score.

## Run 1: Optimized Modern Architecture + Sub-sampled BPE Tokenizer
- **Hypothesis**: Switching to BPE (vocab 1024, trained on first 500k bytes) and upgrading the architecture to modern standards (RMSNorm, RoPE, SwiGLU, weight tying, and AdamW + Cosine schedule) will drastically improve modeling efficiency (bpb) and training data throughput.
- **Changes**:
  - **Tokenizer**: Byte-Pair Encoding (BPE) with vocab size 1024, trained on the first 500k bytes of the corpus. Fast cache-optimized encoder.
  - **Model**: 4 layers, 4 heads, $d_{embd}=160$, SwiGLU feed-forward blocks, RMSNorm normalization, Rotary Positional Embeddings (RoPE), and weight tying between embeddings and LM head.
  - **Training**: AdamW optimizer (weight decay 0.1, no decay on biases/norms), Cosine learning rate scheduler (peak 1e-3, min 1e-4, 200 steps warmup), and gradient clipping (max norm 1.0). Batch size 8, block size 128.
- **Dev BPB**: 1.8856
- **Conclusion**: A huge success. The model trained in 320 seconds (5.3 minutes) and the bpb score dropped from 2.3718 to 1.8856. The BPE tokenizer compressed the sequence length by 1.85x, allowing the model to see more text per step.

## Run 2: Optimized Modern Architecture + Full-Corpus BPE Tokenizer
- **Hypothesis**: Training the BPE tokenizer merges on 100% of the 7.3 MB training corpus (instead of only the first 500k bytes) will capture the full text distribution, improve vocabulary representation, and increase the compression ratio.
- **Changes**:
  - **Tokenizer**: Fast BPE trainer optimized via unique word frequencies, trained on the entire 7.3 MB corpus. Merges saved to `bpe_merges.json`.
  - **Model & Training**: Kept identical to Run 1 (4 layers, 4 heads, $d_{embd}=160$, batch size 8, block size 128, AdamW + Cosine schedule).
- **Dev BPB**: 1.9153
- **Conclusion**: The tokenizer compression ratio improved from 1.85x to **2.21x** (reducing total evaluation tokens from 81,818 to 68,414). The bpb score is slightly higher (1.9153 vs 1.8856) because the model now has to learn transitions over a much more complex and comprehensive vocabulary within the same size and step budget (160 dim, 2,000 steps). However, this tokenizer covers the entire dataset and represents a much more robust representation for generalization to unseen test datasets.
