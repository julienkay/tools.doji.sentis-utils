# FittedModelStorage: a buffer pool that fits

## What it does

A worker keeps its tensors in a "storage", which also recycles GPU buffers: when a tensor is no longer needed, its buffer goes into a pool, and a later tensor can take it. `FittedModelStorage` replaces Inference Engine's `ModelStorage` with a pool that wastes less memory over many runs, and it lets a tensor pass through the worker without a copy (used for the KV cache of language models).

## Why we need it

Inference Engine's pool gives a tensor the smallest free buffer that is big enough, of any size. With a language model that runs again and again, a small tensor (a KV cache entry that the worker keeps between runs) could take a huge buffer (for example one that held a weight converted to float32), and every run then allocated new large buffers. GPU memory grew run after run.

## Similar approaches

**PyTorch's CUDA caching allocator** keeps freed GPU blocks for reuse instead of returning them to the driver, rounds sizes into size classes so a block fits many requests, and splits large blocks. `torch.cuda.empty_cache()` gives unused blocks back.

## How it works with Inference Engine

`IModelStorage` is internal, and `Worker(Model, IBackend, IModelStorage)` is an internal constructor. sentis-utils compiles into Inference Engine (see the [index](README.md)), so `ModelFile.CreateWorker` can pass a `FittedModelStorage`. The rules:

- **Fit.** A tensor takes a free buffer only if the buffer is at most 2 times the tensor's size (`MaxWaste`).
- **Size classes.** A new buffer above 256 KiB is rounded up to the next step of a quarter of its power of two, at most 4 MiB. A tensor that grows a little each run can then take the buffer of the step before.
- **Free what nobody used.** When a run starts, the pool frees the buffers that no tensor took during the last run. So the pool follows what the model needs now, not its all-time peak.
- **Give an input** (`WorkerUtils.TryGiveInput`). The storage becomes the owner of an input tensor and frees it right after the last layer that reads it. Normally the caller owns inputs, so they stay alive for the whole run.
- **Take an output** (`WorkerUtils.TakeOutput`). The caller takes an output tensor out of the storage and owns it. No copy is made.

Give and take together let a KV cache go in and out of the worker while existing only once in GPU memory.

- **Packed types.** It allocates `Short` (float16) and `Byte` tensors with their packed size: two float16 or four uint8 values in each 32-bit element. Inference Engine's own storage can't allocate them.

## Limits

- Only workers from `ModelFile.CreateWorker` use it. `StreamingWorker` and `WorkerUtils` workers still use Inference Engine's storage.
- On D3D12, Unity places every buffer under 32 MiB in 64 MiB heaps and does not give those heaps back. Tensors that get bigger every run leave holes in them. The pool can't fix that; see the KV cache page in tools.doji.transformers.
