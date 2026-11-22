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
- `WorkerUtils.DisposeAndMoveWeightsToCpu`, which disposes such a worker and moves the weights back into the model, so the model can create a worker again.

### Changed

- `Ops.TakeOwnership` and `Ops.WaveOwnership` throw an `ObjectDisposedException` for a disposed tensor.

## [0.1.1] - 2026-02-17

- Compatibility update for Sentis v2.5.0

## [0.1.0] - 2025-04-01

- Compatibility update for Sentis v2.1.0


## [0.0.0] - 2024-08-16

- Initial Release
