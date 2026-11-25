# How sentis-utils works (internal notes)

Short pages, one per idea. Each one says what the feature does, which well-known approach it copies, and how it is built on top of Unity Inference Engine (formerly Sentis). These pages are for maintainers and are not part of the package.

| Page | In one line | Similar to |
|---|---|---|
| [ModelFile](model-file.md) | Weights stay in the `.sentis` file until a worker needs them. | safetensors lazy loading, `low_cpu_mem_usage` |
| [GpuUpload](gpu-upload.md) | Uploads data through two small reused staging buffers. | Upload ring buffers in D3D12/Vulkan engines |
| [FittedModelStorage](fitted-model-storage.md) | A GPU buffer pool that wastes less and lets the KV cache pass through without copies. | PyTorch's CUDA caching allocator |
| [StreamingWorker](streaming-worker.md) | The GPU holds the weights of a few layers at a time. | diffusers `enable_sequential_cpu_offload` |
| [Attention](attention.md) | Attention without the full score matrix in memory. | diffusers attention slicing, FlashAttention |
| [QuantizedGather](quantized-gather.md) | A uint8 embedding table dequantizes only the rows it reads. | llama.cpp `get_rows` on quantized tensors |
| [HalfPrecision](half-precision.md) | Chosen inputs and outputs stay float16 between runs. | float16 KV cache in transformers, vLLM, llama.cpp |
| [ScheduleInParts](schedule-in-parts.md) | One run of a worker spread over frames, with one command buffer execute per part. | Time-sliced jobs with a frame budget |
| [Readbacks](readbacks.md) | Small CPU values that a GPU layer reads get a CPU copy, so the CPU never downloads them. | Sizes kept on the host in PyTorch |
| [GatedDeltaRule](gated-delta-rule.md) | The linear attention of Gated DeltaNet in one layer, for any number of tokens, that a `.sentis` file can hold. | flash-linear-attention `fused_recurrent_gated_delta_rule` |

For the KV cache and chunked prefill of language models, see `internal-docs/` in tools.doji.transformers.

## The tricks these pages share

- **Inference Engine internals.** sentis-utils has an `.asmref` into `Unity.InferenceEngine`, so its code compiles into that assembly and can use internal types such as `Layer`, `IModelStorage`, `ComputeFunction` and `PartialInferenceContext`. A few things that are private (for example `Worker.m_Storage`) are reached by reflection.
- **Custom layers.** A feature that changes how a model runs edits `Model.layers` before a worker is created: it replaces a group of layers with one custom `Layer` subclass whose `Execute` dispatches our own compute shader. Such a model cannot be saved again with `ModelWriter`. A layer in the namespace `Unity.InferenceEngine.Layers` with the `[Operator]` attribute of Inference Engine can be saved (see [GatedDeltaRule](gated-delta-rule.md)).
- **Custom storage.** `ModelFile.CreateWorker` gives the worker a `FittedModelStorage` instead of Inference Engine's `ModelStorage`, through the internal `Worker(Model, IBackend, IModelStorage)` constructor.
