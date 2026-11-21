# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Ops.NonZero`, which returns the indices of the non-zero elements like ONNX NonZero, with shape (rank, count).
- `Ops.ThrowIfDisposed`, to check a tensor before a worker reads it. A worker that reads a disposed tensor gives wrong values with no exception.
- `WorkerUtils.CreateWorkerWithWeightsOnGpu`, which uploads a model's weights to the GPU in chunks and frees their CPU copy, so the weights are not in RAM and on the GPU at the same time.

### Changed

- `Ops.TakeOwnership` and `Ops.WaveOwnership` throw an `ObjectDisposedException` for a disposed tensor.

## [0.1.1] - 2026-02-17

- Compatibility update for Sentis v2.5.0

## [0.1.0] - 2025-04-01

- Compatibility update for Sentis v2.1.0


## [0.0.0] - 2024-08-16

- Initial Release
