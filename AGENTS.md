# AGENTS.md

`tools.doji.sentis-utils`: utilities and extensions for Unity Inference Engine (formerly Sentis): loading `.sentis` files without managed copies of the weights, a GPU buffer pool, weight streaming, attention slicing and flash attention, quantized embeddings, and float16 inputs and outputs.

Unity 6000.6.4f1 for development; `tools.doji.sentis-utils/package.json` sets the supported Inference Engine version.

## Layout

From the companion project, the repository root is `../..`.

- `tools.doji.sentis-utils/`: the UPM package.
  - `Runtime/`: an asmref into `Unity.InferenceEngine`, so the code compiles into Inference Engine's assembly and can use its internal types (`Layer`, `IModelStorage`, `ComputeFunction`, ...). Namespace `Doji.AI`.
  - `Runtime/Resources/ComputeShaders/`: the compute shaders of the custom layers.
  - `Tests/Editor/`: EditMode tests.
- `projects/SentisUtils/`: the companion project. It references the package with a local `file:` path. Open and test from here.
- `internal-docs/`: maintainer notes, one page per feature: what it does, the industry approach it follows, and how it is built on Inference Engine. Start at `internal-docs/README.md`. Update a page when you change its feature.

## Working with Inference Engine

- Features that change how a model runs edit `Model.layers` before a worker is created and add custom `Layer` subclasses. Such models cannot be saved with `ModelWriter`; their `SerializeFields` throws.
- Code that reaches private Inference Engine members by reflection (`ModelLoader.LoadModelDescription`, `Worker.m_Storage`, `Worker.m_Model`) must fail with a clear error if a new Inference Engine version changes them.
- Never edit files in `Library/PackageCache/`. For a temporary diagnostic there, back the file up first and restore it.

## Measuring

- Follow the `model-perf-run` skill for any heavy model load, benchmark, or RAM/VRAM measurement: the RAM watchdog, one Unity instance at a time, headless runs, nvidia-smi for VRAM.
- Run only one Unity instance at a time on this machine. Never run heavy work in the user's open Editor.

## Changelog

Follow the `doji-changelog` skill. No figures measured on one machine (tokens/s, VRAM, RAM, load times) in the changelog or in commit messages.

## Validation

- Preserve `.meta` files when you move or add Unity assets and scripts.
- Run the EditMode tests in the companion project after a change to a custom layer or to `FittedModelStorage`.
