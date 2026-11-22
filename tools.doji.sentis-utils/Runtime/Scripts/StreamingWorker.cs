using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Doji.AI {

    public sealed unsafe partial class ModelFile {

        /// <summary>
        /// Creates a <see cref="BackendType.GPUCompute"/> worker whose weights stay in the file, or in the copy in
        /// memory, and go to the GPU layer by layer while it runs. Run it with
        /// <see cref="StreamingWorker.ScheduleIterable"/>.
        /// </summary>
        /// <param name="bytesAhead">How many bytes of weights the CPU uploads before it waits for the GPU. The GPU
        /// holds about twice as many bytes of weights.</param>
        public StreamingWorker CreateStreamingWorker(long bytesAhead = 256L << 20) {
            if (_stream == null || !_stream.CanRead) {
                throw new ObjectDisposedException(nameof(ModelFile));
            }
            return new StreamingWorker(this, bytesAhead);
        }

        /// <summary>
        /// A GPU worker that uploads each weight before the first layer that reads it, and frees it after the last
        /// one, so the GPU holds the weights of a few layers at a time. Each run reads all weights again, from the
        /// file or from the copy in memory (<see cref="KeepWeightsInMemory"/>).
        /// </summary>
        public sealed class StreamingWorker : IDisposable {

            private readonly ModelFile _file;
            private readonly ModelStorage _storage = new ModelStorage();
            private readonly long _bytesAhead;

            /// <summary>The weights to upload before each GPU layer, by their position in the constants.</summary>
            private readonly List<int>[] _uploadBefore;
            /// <summary>The weights to free after each GPU layer, by their position in the constants.</summary>
            private readonly List<int>[] _freeAfter;

            /// <summary>Weights freed since the last fence, and weights freed before it.</summary>
            private List<ComputeTensorData> _freed = new List<ComputeTensorData>();
            private List<ComputeTensorData> _beforeFence = new List<ComputeTensorData>();
            private AsyncGPUReadbackRequest? _fence;
            private long _sinceFence;
            private NativeArray<int> _staging;

            /// <summary>
            /// The worker, to set its inputs and get its outputs. Do not schedule it directly: its layers would read
            /// weights that are not on the GPU.
            /// </summary>
            public Worker Worker { get; }

            internal StreamingWorker(ModelFile file, long bytesAhead) {
                _file = file;
                _bytesAhead = bytesAhead;
                Model model = file._model;
                var weights = new List<int>();
                Worker = file.CreatePlaceholderWorker(BackendType.GPUCompute, _storage, weights);
                var position = new Dictionary<int, int>();
                foreach (int i in weights) {
                    position[model.constants[i].index] = i;
                    _storage.GetTensor(model.constants[i].index).ReleaseTensorData()?.Dispose();
                }

                // The worker yields after each layer that runs on the GPU. The layers that run on the CPU before it
                // read no GPU weight.
                var cpuLayers = CPUFallbackCalculator.Calculate(model, BackendType.GPUCompute, out _);
                var first = new Dictionary<int, int>();
                var last = new Dictionary<int, int>();
                int step = 0;
                foreach (var layer in model.layers) {
                    if (cpuLayers.Contains(layer.outputs[0])) {
                        continue;
                    }
                    foreach (int input in layer.inputs) {
                        if (position.TryGetValue(input, out int i)) {
                            first.TryAdd(i, step);
                            last[i] = step;
                        }
                    }
                    step++;
                }
                _uploadBefore = new List<int>[step];
                _freeAfter = new List<int>[step];
                for (int s = 0; s < step; s++) {
                    _uploadBefore[s] = new List<int>();
                    _freeAfter[s] = new List<int>();
                }
                foreach (var (i, s) in first) {
                    _uploadBefore[s].Add(i);
                }
                foreach (var (i, s) in last) {
                    _freeAfter[s].Add(i);
                }
            }

            /// <summary>
            /// Schedules the model in parts, like <see cref="Unity.InferenceEngine.Worker.ScheduleIterable()"/>, and
            /// streams the weights to the GPU meanwhile.
            /// </summary>
            public IEnumerator ScheduleIterable() {
                IEnumerator schedule = Worker.ScheduleIterable();
                for (int step = 0; ; step++) {
                    if (step < _uploadBefore.Length) {
                        foreach (int i in _uploadBefore[step]) {
                            Upload(i);
                        }
                    }
                    if (!schedule.MoveNext()) {
                        yield break;
                    }
                    if (step < _freeAfter.Length) {
                        foreach (int i in _freeAfter[step]) {
                            if (_storage.GetTensor(_file._model.constants[i].index).ReleaseTensorData() is ComputeTensorData data) {
                                _freed.Add(data);
                            }
                        }
                    }
                    yield return null;
                }
            }

            private void Upload(int i) {
                var constant = _file._model.constants[i];
                Tensor tensor = _storage.GetTensor(constant.index);
                if (tensor.dataOnBackend != null) {
                    // An earlier run stopped before the last layer that reads it.
                    return;
                }
                int ints = Padded(constant.lengthBytes) / sizeof(int);
                NativeArray<int> source;
                if (_file._weights != null) {
                    source = View(_file._weights + _file._weightOffsets[i], ints);
                } else {
                    if (!_staging.IsCreated) {
                        _staging = new NativeArray<int>(_file._largest / sizeof(int), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    }
                    _file.Read(_file._offsets[i], _staging.GetUnsafePtr(), constant.lengthBytes);
                    source = _staging;
                }
                var data = new ComputeTensorData(tensor.count);
                tensor.AdoptTensorData(data, disposePrevious: false, disposeIsDelayed: false);
                data.buffer.SetData(source, 0, 0, ints);
                _sinceFence += constant.lengthBytes;
                if (_sinceFence >= _bytesAhead) {
                    Fence(data);
                }
            }

            /// <summary>
            /// Waits until the GPU reaches the previous fence, frees the weights that were freed before it, and sets
            /// a new fence after the upload to <paramref name="data"/>.
            /// </summary>
            private void Fence(ComputeTensorData data) {
                WaitForFence();
                (_beforeFence, _freed) = (_freed, _beforeFence);
                _fence = AsyncGPUReadback.Request(data.buffer, sizeof(int), 0);
                _sinceFence = 0;
            }

            private void WaitForFence() {
                if (_fence.HasValue) {
                    _fence.Value.WaitForCompletion();
                    _fence = null;
                }
                foreach (var data in _beforeFence) {
                    data.Dispose();
                }
                _beforeFence.Clear();
            }

            public void Dispose() {
                Worker.Dispose();
                WaitForFence();
                foreach (var data in _freed) {
                    data.Dispose();
                }
                _freed.Clear();
                if (_staging.IsCreated) {
                    _staging.Dispose();
                }
            }
        }
    }
}
