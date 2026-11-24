# ModelFile: weights stay in the file

## What it does

`ModelFile.Open(path)` reads only the description of a `.sentis` model: its layers, inputs, outputs, and where each weight is in the file. The weights stay on disk. `CreateWorker` then reads them one by one and puts each one straight on the GPU. So a 7 GB model never has a 7 GB copy in managed memory, and there is never a copy in RAM and on the GPU at the same time.

The `ModelFile` can create a worker again later, and it is also where the other features are switched on before the worker exists (`SliceAttention`, `UseFlashAttention`, `UseHalfPrecision`, `CreateStreamingWorker`).

## Similar approaches

- **safetensors** files are memory-mapped and read tensor by tensor, so a loader never holds the whole file at once.
- **transformers `low_cpu_mem_usage=True`** and **accelerate `init_empty_weights`** first build the model without weights, then fill each weight in from the file.

`ModelFile` does both: empty model first, then one weight at a time from the file.

## How it works with Inference Engine

1. **Read the description only.** `ModelLoader.Load` would read everything into managed arrays. Instead we call Inference Engine's private `ModelLoader.LoadModelDescription` by reflection. It builds the `Model` without weight data and gives the file offset of each weight.
2. **Worker with empty weights.** We create the worker while each weight's array is empty, so Inference Engine uploads nothing. The weights that layers read on the CPU (shapes, small constants) are read normally. `CPUFallbackCalculator` tells which ones those are.
3. **Read and upload in turn.** Two native staging arrays take turns: while one weight uploads, `AsyncReadManager` reads the next one from the file. The upload goes through [GpuUpload](gpu-upload.md). Each weight gets its own `ComputeTensorData`, which the worker's tensor adopts.
4. **Optional RAM copy.** `KeepWeightsInMemory` keeps the weights in one native block, so the next worker or a [StreamingWorker](streaming-worker.md) does not read the disk again.

The file stays open until the `ModelFile` is disposed.

## Limits

- It depends on the private `LoadModelDescription`. A new Inference Engine version can rename or change it.
- A model changed by these features (custom layers) cannot be saved again. Call `LoadModel()` to get the plain model.
