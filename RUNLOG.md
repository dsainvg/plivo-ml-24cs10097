# Run Log

This file contains the hypothesis, changes, and results for each training run during the optimization process.

## Run 0: Baseline
- **Hypothesis**: Baseline configuration provided in the starter code.
- **Changes**: None.
- **Dev BPB**: 2.3718
- **Conclusion**: The baseline model has 1.34M parameters and runs in 237s on CPU, but the byte-level tokenizer and simple architecture yield a high BPB score.

## Run 1: Optimized Modern Architecture + BPE Tokenizer
- **Hypothesis**: Switching to BPE (vocab 1024) and upgrading the architecture to modern standards (RMSNorm, RoPE, SwiGLU, weight tying, and AdamW + Cosine schedule) will drastically improve modeling efficiency (bpb) and training data throughput.
- **Changes**:
  - **Tokenizer**: Byte-Pair Encoding (BPE) with vocab size 1024, trained on the first 500k bytes of the corpus. Fast cache-optimized encoder.
  - **Model**: 4 layers, 4 heads, $d_{embd}=160$, SwiGLU feed-forward blocks, RMSNorm normalization, Rotary Positional Embeddings (RoPE), and weight tying between embeddings and LM head.
  - **Training**: AdamW optimizer (weight decay 0.1, no decay on biases/norms), Cosine learning rate scheduler (peak 1e-3, min 1e-4, 200 steps warmup), and gradient clipping (max norm 1.0). Batch size 8, block size 128.
- **Dev BPB**: 1.8856
- **Conclusion**: A huge success. The model trained in 320 seconds (5.3 minutes) and the bpb score dropped from 2.3718 to 1.8856. The BPE tokenizer compressed the sequence length by 1.95x, allowing the model to see more text per step and improving the context window in terms of characters.
