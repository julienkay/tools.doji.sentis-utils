# Attention: slicing and flash attention

## What it does

Attention computes a score for every pair of query and key, a matrix of size queries × keys per head. For a big image model, or a long text prompt, that matrix can be the largest tensor in the model. Both options here give the same result (apart from rounding) without ever holding the full matrix:

- **Slicing** (`Attention.Slice`, `ModelFile.SliceAttention`) computes the scores for a few queries at a time, at most a given number of bytes.
- **Flash attention** (`Attention.UseFlashAttention`, `ModelFile.UseFlashAttention`) computes attention in one GPU kernel that keeps the scores in fast on-chip memory and never writes them out.

## Similar approaches

- **diffusers `enable_attention_slicing`** does the same slicing.
- **FlashAttention** (Tri Dao), **xFormers memory-efficient attention**, and PyTorch's **`scaled_dot_product_attention`** compute softmax in tiles of keys with a running maximum and sum ("online softmax"), so the score matrix never exists.

## How it works with Inference Engine

Inference Engine has no attention operator. An exported model spells attention out as separate layers: MatMul (query × key), optionally a scale (ScalarMad) and a mask (Add), Softmax, then MatMul with the values.

1. **Find the pattern.** We walk the layers from each Softmax on the last axis backwards and forwards to find that chain. The middle results must be used only inside the chain.
2. **Replace it.** The chain becomes one custom `AttentionLayer` that takes query, key, value and mask.
3. **Run it.**
   - Sliced: the layer computes the same operations with the backend's own MatMul and Softmax, but for a block of query rows at a time.
   - Flash: on the GPU backend, it dispatches our `FlashAttention` compute shader (64 queries per thread group, online softmax over tiles of keys). Shapes the kernel can't handle fall back to slicing: a head size above 128, broadcast between query, key and value, or other backends.

## Limits

- The changed model can't be saved with `ModelWriter`.
- A model whose attention is written differently (for example a fused operator, or a different layer order) is not matched and stays as it is.
