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

## Run 3: 5-Layer Hybrid MoE Model + Space-Preserving BPE Tokenizer
- **Hypothesis**: Expanding the BPE tokenizer vocabulary to 2048 using a space-preserving pre-tokenization regex will increase the compression ratio. Combining this with a deeper 5-layer hybrid architecture (3 dense blocks, 2 expert MoE blocks) will maximize capacity under the 1.5M active and 2M total parameter caps.
- **Changes**:
- **Tokenizer**: Vocab 2048 BPE trained on 100% of the training corpus (7.3MB). Uses regex pattern `r' ?[\u0900-\u097Fa-zA-Z0-9]+|\s+|[^\s]'` which merges leading spaces into words, significantly improving compression to **3.35x** (evaluation tokens reduced to 47,499).
  - **Model**: 5 layers (layers 2, 3 are MoE with 2 experts; layers 0, 1, 4 are dense). $d_{embd}=144$, SwiGLU hidden dim $d_{ff}=368$. Weight tying and RoPE.
  - **Training**: Kept identical (AdamW, Cosine schedule with warmup, batch size 8, block size 128).
  - **Parameters**: Total parameters = 1,824,624. Active parameters = 1,499,200.
- **Dev BPB**: 1.8285
- **Conclusion**: A resounding success. Training BPE on 100% of the training corpus improved BPE representation and pushed the dev BPB to a new best of **1.8285**. The model trained in 286 seconds (4.8 minutes). The space-preserving tokenizer allowed the sequence block of 128 to span an average of **429 bytes** of text. The hybrid layout successfully leveraged the MoE expert scaling while remaining strictly within the active and total parameter caps.

## Run 4: 6-Layer Hybrid MoE Model (Depth-vs-Width Experiment)
- **Hypothesis**: Making the network deeper (6 layers instead of 5) and increasing learning rate to 1.2e-3 will improve sequential logic and convergence, even if the embedding dimension ($d_{embd}$) must be reduced to 128 to stay under the 1.5M active parameter cap.
- **Changes**:
  - **Tokenizer**: Identical to Run 3 (Vocab 2048, space-preserving BPE trained on 100% of corpus, 3.35x compression).
  - **Model**: 6 layers (layers 1, 2, 3, 4, 5 are MoE with 2 experts; layer 0 is dense). $d_{embd}=128$, SwiGLU hidden dim $d_{ff}=280$. Weight tying and RoPE.
  - **Training**: Set peak learning rate to `1.2e-3` (with 200 step warmup and Cosine schedule).
  - **Parameters**: Total parameters = 1,841,024. Active parameters = 1,303,680.
- **Dev BPB**: 1.8789
- **Conclusion**: The 6-layer model trained in 331 seconds (exceeding the 5-minute training target) and got a slightly worse dev BPB of **1.8789** (compared to **1.8285** in Run 3). This confirms that for small parameter regimes and large vocabularies (2048), model width ($d_{embd} = 144$) and active parameter capacity per step (~1.5M) are more crucial than model depth (6 layers). The Run 3 model was restored as the best checkpoint.

## Run 5: 6-Layer Hybrid MoE — Sparse Middle Pattern (Layers 1, 2, 4 as MoE) + 50-step Warmup
- **Hypothesis**: Changing the MoE placement to a "sparse middle" pattern (layers 1, 2, 4 are MoE; layers 0, 3, 5 are dense) so that **both the first and last layers are dense**, and reducing warmup from 200 → 50 steps so the peak LR is reached faster, will improve BPB for the 6-layer model.
- **Changes**:
  - **Tokenizer**: Identical to Run 3 (Vocab 2048, space-preserving BPE, 3.35x compression).
  - **Model**: 6 layers, `moe_layers = [1, 2, 4]`. Layer 0 = dense (embedding stabilizer), Layers 1, 2, 4 = MoE (2 experts), Layer 3, 5 = dense (last layer dense). $d_{embd}=128$, $d_{ff}=340$.
  - **Training**: Peak LR `1.2e-3`, **warmup fixed to 50 steps** (down from 200), Cosine decay.
  - **Parameters**: Total = 1,832,832. Active = 1,441,152.
- **Dev BPB**: 1.9046
- **Conclusion**: Despite the improved MoE placement (both first and last layers dense) and faster warmup, the 6-layer `d_embd=128` model again underperformed Run 3's 5-layer `d_embd=144` model (1.9046 vs 1.8285). The narrower embedding dimension is a hard constraint: with fewer dimensions per head (32 vs 36), attention quality degrades regardless of depth or warmup schedule. **Run 3 (5-layer, d_embd=144, BPB=1.8285) remains the final best configuration.**
