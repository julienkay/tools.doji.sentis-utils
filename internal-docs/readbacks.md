# Readbacks

`Readbacks.Avoid(model)` stops a worker on the GPU backend from downloading small values, such as a sequence length, from the GPU. `ModelFile` applies it when it opens a file.

## The problem

On the GPU backend, Inference Engine computes shapes and other small values on the CPU. A language model computes its total length from the shape of `attention_mask`, and two layers read that length:

```
Shape(attention_mask) -> Select -> total      CPU
Sub(total, input length)                      GPU layer: total moves to the GPU
Range(0, total, 1)                            needs total on the CPU: download
```

When a GPU layer reads a CPU tensor, Inference Engine moves the tensor to the GPU and frees the CPU copy (`ComputeTensorData.Pin`). The later CPU reader must then download it. A download waits until the GPU has done all the work submitted before it, so the main thread stops until the GPU catches up. Here this happened at the start of every decode step.

## The fix

Compute the value twice. Before the layer that computes the value, the pass adds a copy of that layer:

```
Shape(attention_mask) -> Select -> total'     CPU: the copy, for the CPU readers
Shape(attention_mask) -> Select -> total      for the GPU readers
Sub(total, input length)                      GPU: total moves to the GPU
Range(0, total', 1)                           reads total' on the CPU: no download
```

The copied layers are small CPU layers, so the extra work is negligible. The outputs of the model are the same.

## Similar to

- PyTorch keeps tensor sizes on the CPU, so asking for a size needs no sync with the GPU. Only `.item()` on a GPU tensor syncs.

## How

1. **Find.** `CPUFallbackCalculator.Calculate` tells which layers run on the CPU. A tensor needs a copy when a GPU layer reads it, and a later layer reads its values on the CPU. That later layer is either a GPU layer with a CPU-read input (`Layer.IsInputCPURead`), or a CPU layer that reads the data, not only the shape (`IsInputNoDataDependency`). `Readbacks.Find` lists these tensors.
2. **Copy.** If a CPU layer with one output computes the tensor, a shallow copy of that layer, with a new output, goes right before it. The copy reads the inputs first, before a GPU layer can move them.
3. **Redirect.** The CPU readers read the copy. Only GPU readers read the original.
4. **Repeat.** A copy can need copies of its own inputs, so the pass runs again until it finds nothing, at most 8 times.

## Limits

- Model inputs, constants, tensors that a GPU layer computes and layers with more than one output get no copy.
- `CPUFallbackCalculator` compares output indices, not layer positions. A copy has a new, higher index, so Inference Engine can flush the command buffer before it. The flush comes before the GPU work of the run, so it costs little.
- A CPU layer that reads only the shape of a GPU tensor (for example `Shape` of a key in each attention block) still flushes the command buffer. It does not download.

## Used by

tools.doji.transformers: through `ModelFile`, and `PreTrainedModel` calls `Readbacks.Avoid` on a `Model` in memory. Without it, `GenerationPacing.SpreadStepsOverFrames` had a long frame at the start of each decode step.
