# StreamingWorker: weights go to the GPU layer by layer

## What it does

A normal worker puts all weights on the GPU and keeps them there. A `StreamingWorker` (from `ModelFile.CreateStreamingWorker`) keeps the weights in the file, or in a RAM copy, and uploads each one right before the first layer that reads it. It frees it after the last layer that reads it. The GPU holds the weights of only a few layers at a time, so a model bigger than VRAM can run. The cost: every run uploads all the weights again, so it is much slower.

## Similar approaches

- **diffusers `enable_sequential_cpu_offload`** (built on **accelerate** hooks): each submodule's weights move to the GPU just before it runs and back to the CPU after.
- **llama.cpp** with only some layers on the GPU (`-ngl`) is a related idea. It keeps the other layers on the CPU instead of streaming them.

## How it works with Inference Engine

1. **Plan.** Before the first run, it walks the model's layers once and notes, for each GPU layer, which weights to upload before it and which weights to free after it. Layers that run on the CPU are skipped. `CPUFallbackCalculator` tells which ones those are.
2. **Empty worker.** The worker is created with no GPU weights, like [ModelFile](model-file.md) does.
3. **Run in steps.** `ScheduleIterable()` wraps Inference Engine's `Worker.ScheduleIterable()`, which runs one layer per step. Before each step it uploads that step's weights; after the step it takes the data of the weights to free off their tensors.
4. **Don't run too far ahead.** After every `bytesAhead` bytes of uploads, it sets a fence (an `AsyncGPUReadback` of 4 bytes) and waits for the previous one. Only then does it dispose the weights freed before that fence, because the GPU may still be reading them. This bounds the GPU memory to about twice `bytesAhead` of weights.

diffusers' `ModelRunner` uses it when weight streaming is on.

## Limits

- Don't call `Worker.Schedule()` directly: its layers would read weights that are not on the GPU.
- It uploads with `ComputeBuffer.SetData`. If a run goes through without frames in between, the upload memory of all streamed weights can pile up in RAM until the frame ends (see [GpuUpload](gpu-upload.md)). This is not measured for `StreamingWorker` yet. `GpuUpload`, or a `GL.Flush()` at each fence, would avoid it.
