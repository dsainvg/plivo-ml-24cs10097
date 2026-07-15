# Notes

1. Our best configuration uses a vocabulary size of 2048 BPE tokens, a context block size of 128, and a hybrid 5-layer model combining 3 dense layers and 2 Mixture of Experts (MoE) layers (layers 2 and 3).
2. Weight tying between token embeddings and the output head reduces parameter count by 327,680, enabling an expanded vocabulary size of 2048 without exceeding total parameter caps.
3. An optimized, space-preserving pre-tokenization regex (` ?[\u0900-\u097Fa-zA-Z0-9]+|\s+|[^\s]`) merges leading spaces directly into words, boosting the BPE tokenizer's compression ratio to **3.35x** on the corpus.
4. With a compression ratio of 3.35x, a sequence length of 128 tokens effectively spans an average context of **429 bytes** of text.
5. In the 5-layer layout, the last layer (layer 4) remains dense to stabilize representation features before final classification.
6. The MoE layers contain 2 experts each, which are gated dynamically per-token using a learned linear router.
7. A load-balancing auxiliary loss with a weight of 0.01 is added to the main training loss to prevent expert collapse and ensure balanced routing.
8. The model has **1,824,624 total parameters** (under the 2.0M cap) and **1,499,200 active parameters** per step (under the 1.5M active cap).
9. Normalization is handled via parameter-free RMSNorm (with scaling weight), simplifying computation and speeding up CPU forward passes.
10. This configuration achieves our lowest Bits Per Byte (bpb) of **1.8285** on the dev set, training in 286 seconds on CPU.
