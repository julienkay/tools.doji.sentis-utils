# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Ops.NonZero`, which returns the indices of the non-zero elements like ONNX NonZero, with shape (rank, count).
- `Ops.ThrowIfDisposed`, to check a tensor before a worker reads it. A worker that reads a disposed tensor gives wrong values with no exception.
- `WorkerUtils.CreateWorkerWithWeightsOnGpu`, which uploads a model's weights to the GPU in chunks and frees their CPU copy, so the weights are not in RAM and on the GPU at the same time. This also frees the RAM of a model loaded from a `.sentis` file.
- `ModelFile`, an open `.sentis` file whose weights stay in the file. `ModelFile.CreateWorker` reads each weight from the file into native memory, and on the GPU backend uploads it while it reads the next one, so the weights are never in managed memory. It can create a worker again later. `KeepWeightsInMemory` keeps a copy of the weights in native memory instead, so a new worker does not read the file. The file stays open until the `ModelFile` is disposed.
- `Attention.Slice` and `ModelFile.SliceAttention`, like diffusers' attention slicing: each attention of a model (MatMul, optional scale and mask, Softmax, MatMul) becomes one layer that computes its scores for a few queries at a time. The result is the same, and an attention holds at most the given bytes of scores.
- `Attention.UseFlashAttention` and `ModelFile.UseFlashAttention`: each attention of a model becomes one layer that runs a flash attention kernel on the GPU backend. The kernel computes the softmax in tiles of keys, so the scores are never in memory. Other backends, and shapes that the kernel does not support, use the sliced path.
- `QuantizedGather.Fuse`: a uint8 embedding table (DequantizeUint8, then Gather) dequantizes only the rows it gathers, not the whole table. `ModelFile` does this when it opens a file.
- `ModelFile.CreateStreamingWorker`, a GPU worker that uploads each weight before the first layer that reads it and frees it after the last one, so the GPU holds the weights of a few layers at a time. Run it with `StreamingWorker.ScheduleIterable`.
- `WorkerUtils.DisposeAndMoveWeightsToCpu`, which disposes such a worker and moves the weights back into the model, so the model can create a worker again.
- `WorkerUtils.TryGiveInput`, which sets an input and gives it to the worker: the next run frees it after the last layer that reads it. `WorkerUtils.TakeOutput` takes an output from the worker without a copy. A KV cache can go through a worker of `ModelFile.CreateWorker` this way and stay in memory only once.
- `HalfPrecision.Apply` and `ModelFile.UseHalfPrecision` make the named float inputs and outputs of a model float16 (`DataType.Short`). The model casts each such input to float32 right before the first layer that reads it, and each such output to float16 right after the layer that writes it. A KV cache can then stay in float16 between runs, at half the memory. The float16 outputs need the GPU backend and a worker of `ModelFile.CreateWorker`. `ModelFile.HasHalfPrecisionWeights` tells whether a model has float16 weights.

### Changed

- `Ops.TakeOwnership` and `Ops.WaveOwnership` throw an `ObjectDisposedException` for a disposed tensor.
- `ModelFile.CreateWorker` and `WorkerUtils.CreateWorkerWithWeightsOnGpu` upload the weights through two small staging buffers and a copy kernel. Before, each upload used `ComputeBuffer.SetData`, whose staging memory the graphics driver frees only some frames later, so loading a model without frames between needed RAM for all of its weights at once.
- The workers of `ModelFile.CreateWorker` reuse a free GPU buffer for a tensor only if the buffer is at most twice the size of the tensor. When a run starts, they free the buffers that no tensor took during the last run. Above 256 KiB, they round the size of a new buffer up to the next step of a quarter of its power of two (at most 4 MiB), so a tensor that grows, such as a KV cache, can take the buffer of a shorter one. Before, a small output that the worker keeps between runs, such as a KV cache entry, could keep a large buffer, and each run allocated more.

## [0.1.1] - 2026-02-17

- Compatibility update for Sentis v2.5.0

## [0.1.0] - 2025-04-01

- Compatibility update for Sentis v2.1.0


## [0.0.0] - 2024-08-16

- Initial Release
