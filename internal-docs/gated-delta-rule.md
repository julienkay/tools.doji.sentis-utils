# GatedDeltaRule

`LinearAttention.GatedDeltaRule` is a layer for the linear attention of Gated DeltaNet, which Qwen3.5 and Qwen3-Next use. A `.sentis` file can hold it.

## The problem

Each linear attention layer keeps a state S per head, of key size x value size (128 x 128 in Qwen3.5). Each token updates S, and the output of the token reads S:

```
S = S * exp(g[t])
delta = (v[t] - Sᵀ k[t]) * beta[t]
S = S + k[t] deltaᵀ
o[t] = Sᵀ (q[t] * scale)
```

The tokens depend on each other, so a prompt of T tokens needs T steps in order. ONNX expresses that with Loop or Scan, and Inference Engine supports neither. With standard ops, one run can compute only a fixed number of tokens.

## The fix

One layer computes all tokens of a run:

- **GPU.** One thread for each column of S. A column of S depends only on itself, so each thread keeps its column (up to 128 values) in registers and walks the tokens in order. The 32 threads of a group share the key and the query of each token in groupshared memory.
- **CPU.** The same loop in C#, for the CPU backend and the tests.

The inputs and outputs follow transformers' `torch_recurrent_gated_delta_rule`, in the (batch, tokens, heads, size) layout. The layer does not normalize the query and the key. The model does that with standard ops before the layer, as transformers does with `use_qk_l2norm_in_kernel`.

## Similar to

- `fused_recurrent_gated_delta_rule` of flash-linear-attention, with the same inputs and `scale`.
- transformers' `torch_recurrent_gated_delta_rule`, which runs the same loop in PyTorch.

## How

- The class is `Unity.InferenceEngine.Layers.GatedDeltaRule`, with the `[Operator]`, `[Inputs]` and `[Outputs]` attributes of the Inference Engine layers. sentis-utils compiles into the Inference Engine assembly, so the source generator of Inference Engine treats it like its own layers. The generator writes the code that saves and loads the layer, the graph node, and `FunctionalLayer.GatedDeltaRule`. A model with this layer can be saved again, which is not true for the other custom layers of sentis-utils.
- `LinearAttention.GatedDeltaRule` is the public functional API.
- A `.sentis` file with this layer loads only where sentis-utils is installed.
- An ONNX model holds the layer as a node `GatedDeltaRule` in the domain `ai.doji`, with the attribute `scale`. tools.doji.onnx-model-loader maps such a node to the layer of the same name.

## Limits

- On the GPU, each thread runs the tokens of a prompt one after the other. The chunked algorithm of flash-linear-attention would use the GPU better for long prompts.
- The key size is at most 128 on the GPU.
- Only the CPU and GPUCompute backends.

## Used by

tools.doji.transformers: the linear attention layers of Qwen3.5.
