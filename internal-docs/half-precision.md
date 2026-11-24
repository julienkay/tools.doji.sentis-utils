# HalfPrecision: float16 inputs and outputs

## What it does

`HalfPrecision.Apply` (or `ModelFile.UseHalfPrecision`) makes chosen float inputs and outputs of a model float16. The layers inside still compute in float32. Only the tensors that go in and out, and stay alive between runs, take half the memory. The main use is the KV cache of a language model: the model reads it as inputs and writes the new one as outputs, and it is the biggest thing that grows with the prompt length.

tools.doji.transformers turns this on for the KV cache when a model has float16 weights and runs on the GPU.

## Similar approaches

- **transformers** keeps the KV cache in the model's dtype, so a float16 model has a float16 cache.
- **vLLM** (`kv_cache_dtype="auto"`) and **llama.cpp** (float16 KV by default) do the same. Both can go further, to 8-bit KV.

## How it works with Inference Engine

Inference Engine stores float16 as `DataType.Short`, packed two values per 32-bit element. It can cast float16 to float32, but has no cast from float32 to float16. So:

1. **Inputs.** Each chosen input is marked `Short`. A custom `CastHalfToFloatLayer` (which calls the backend's own float16-to-float32 cast) is inserted right before the first layer that reads it. The other layers are rewired to read the cast.
2. **Outputs.** A custom `CastFloatToHalfLayer` is inserted right after the layer that writes each chosen output. It runs our `CastFloatToHalf` compute shader (`f32tof16`, two values packed per element). The model output then points at the float16 result.
3. **Placement matters.** Each cast sits next to the layer that uses it, so only one layer's float32 copy exists at a time. A first version put all casts at the start and end, so all float32 copies were alive at once, and it used more memory than float32 did.
4. **Storage.** Inference Engine's own storage can't allocate `Short` tensors, so this needs a worker from `ModelFile.CreateWorker`, which uses [FittedModelStorage](fitted-model-storage.md).

## Limits

- float16 outputs need the GPU backend.
- The changed model can't be saved with `ModelWriter`.
