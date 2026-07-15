# Notes

1. Our best configuration uses a vocabulary size of 1024 BPE tokens, a context block size of 128, a 4-layer model with 4 attention heads, an embedding dimension of 160, and a SwiGLU feed-forward hidden dimension of 426.
2. Weight tying between token embeddings and the output language modeling head reduces the model parameter count by 163,840, permitting a larger vocabulary size without exceeding the parameter cap.
3. The Byte-Pair Encoding (BPE) tokenizer provides a 2.21x compression ratio on the training corpus, which effectively increases the sequence length and context window of the model without additional compute.
4. Rotary Positional Embeddings (RoPE) replace absolute learned positional embeddings, saving parameters and improving the model's relative positional generalization.
5. SwiGLU activation replaces GELU in the MLP block, providing smoother gradients and faster learning.
6. RMSNorm is used instead of standard LayerNorm, simplifying normalization calculations on the CPU.
7. The AdamW optimizer with a cosine learning rate scheduler and 200 warmup steps prevents early gradient saturation and ensures stable convergence.
8. Gradient clipping at a maximum norm of 1.0 prevents training instability at the peak learning rate of 1e-3.
9. Weight decay of 0.1 is applied only to 2D weights, regularizing the model and preventing overfitting.
10. This configuration achieves a bits-per-byte (bpb) of 1.9153, representing a massive improvement over the baseline (2.3718) while training in 239 seconds on CPU.
