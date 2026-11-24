# ScheduleInParts

`commandBuffer.ScheduleInParts(worker, ...)` (in `WorkerScheduling`) runs one run of a worker over several calls of `MoveNext`, for example one per frame. Each part schedules layers until a time budget is used, or until a callback says the part ends, and then executes the command buffer once. An app can spread a run that takes longer than a frame over frames, so that no frame gets its whole cost.

## Similar to

- Time-sliced work in game engines: a job gets a fixed budget per frame and continues in the next one.

## How

- It wraps Inference Engine's `commandBuffer.ScheduleWorkerIterable(worker)`. That call sets the command buffer of the worker's GPU backend (`GPUComputeBackend.SetCommandBuffer`), so the backend no longer has an internal command buffer.
- `Worker.ScheduleIterable` yields after every GPU layer. With an internal command buffer it executes the buffer after every layer. A language model has thousands of small layers, and each execute has a fixed cost (D3D12 command submission), so that costs more than spreading the run saves. With an external command buffer it executes nothing: the layers collect in the command buffer.
- After each layer, `ScheduleInParts` asks whether the part ends. At the end of a part it executes and clears the command buffer (`Graphics.ExecuteCommandBuffer`).
- The layers run as in `Worker.Schedule()`, so the outputs are the same. CPU fallback layers that read GPU outputs still flush first: `ExecuteCommandBufferAndClear` executes the backend's current command buffer, which is the external one. A worker of `ModelFile.CreateWorker` frees given inputs after the layers that read them, as in `Schedule()`.
- On the CPU backend it iterates `Worker.ScheduleIterable` and ignores the command buffer.
- It uses only public Inference Engine API.

## Limits

- The worker keeps the external command buffer for good: Inference Engine has no way back to the internal one. After a later `Worker.Schedule()` or `commandBuffer.ScheduleWorker(worker)`, the caller must execute and clear the command buffer before reading outputs. `Worker.CopyOutput` no longer executes it either.
- Between parts, don't set inputs, schedule or read outputs of the worker.
- `StreamingWorker` has its own `ScheduleIterable`, which uploads weights before each layer. Don't use `ScheduleInParts` on a `StreamingWorker.Worker`.
- If the enumerator is dropped before the end, the run is incomplete. The next run of the worker starts fresh.
- Each part schedules at least one GPU layer. With `ScheduleInParts(double)`, the clock of a part starts after its first layer.

## Used by

tools.doji.transformers: `GenerationPacing.SpreadStepsOverFrames` spreads each decode step of a language model over frames. The model owns a command buffer and executes it after each full run too.
