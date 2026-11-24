# GpuUpload: upload through two small staging buffers

## What it does

It copies data from CPU memory into GPU buffers, using only two small (32 MB) staging buffers that it reuses. RAM use stays at the size of those two buffers, however much data goes up.

## Why we need it

`ComputeBuffer.SetData` copies the data into an upload buffer that Unity frees only when the commands that use it are submitted, normally at the end of the frame. A loader that uploads a whole model without frames in between (batch mode, or one long loading step) keeps an upload copy of every weight in RAM until the end. For Phi-3 float16 that was about as much RAM as the weights themselves. A fence (a finished `AsyncGPUReadback`) does not free that memory; a submit does (`GL.Flush()` or the end of the frame). A repro is in `G:\Unity\BugRepros\SetDataStagingMemory`.

## Similar approaches

Game engines and GPU frameworks upload through an **upload ring buffer**: a fixed CPU-visible buffer that the CPU fills, the GPU copies from, and the CPU fills again once a fence says the GPU is done. CUDA code does the same with a few pinned staging buffers.

## How it works with Unity

For each 32 MB part of the data:

1. Wait for the fence of the staging buffer we want to use, so the GPU is done reading it.
2. Write the part into it with `BeginWrite` / `EndWrite`. The buffers are `ComputeBufferMode.SubUpdates`, so the CPU writes directly into GPU-visible memory and nothing is staged again.
3. Dispatch a small copy kernel (`Compute.Shaders.CopyInts`) that copies the part to its place in the destination buffer.
4. Set a fence for that staging buffer: an `AsyncGPUReadback` of a few bytes of the destination, which completes after the copy.

The two staging buffers take turns, so the CPU fills one while the GPU copies from the other.

`ModelFile.CreateWorker` and `WorkerUtils.CreateWorkerWithWeightsOnGpu` use it. [StreamingWorker](streaming-worker.md) still uses `SetData`.

## Possible simplification

The SetData repro found that `SetData` followed by `GL.Flush()` keeps RAM flat too. That could replace this class. The cost of a flush per weight on load time is not measured yet.
