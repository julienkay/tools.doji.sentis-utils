using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;

namespace Doji.AI {

    /// <summary>
    /// The tensors of a worker, as Inference Engine's ModelStorage keeps them, with a GPU buffer pool that gives a
    /// tensor only a free buffer of at most <see cref="MaxWaste"/> times its size. ModelStorage gives the smallest
    /// free buffer that is large enough, of any size. Between runs, the worker keeps its outputs, for example the KV
    /// cache of a language model. Small outputs can then keep large buffers, for example the buffer of a weight that
    /// was dequantized to float32, and each run allocates more large buffers. Also, when a run starts, this storage
    /// frees the buffers that no tensor took during the last run, for example buffers for a KV cache that was shorter.
    /// An input given with <see cref="GiveInput"/> is freed after the last layer that reads it, and an output taken
    /// with <see cref="TakeTensorOwnership"/> leaves the storage, so a KV cache can go through the worker without a
    /// copy.
    /// </summary>
    internal sealed class FittedModelStorage : IModelStorage {

        /// <summary>
        /// The largest ratio of buffer capacity to tensor size that the pool gives.
        /// </summary>
        public const int MaxWaste = 2;

        private readonly Dictionary<int, Tensor> _inUse = new Dictionary<int, Tensor>();
        private HashSet<int> _notOwned = new HashSet<int>();
        private readonly Dictionary<int, List<int>> _disposeWhenLayerDone = new Dictionary<int, List<int>>();
        private HashSet<int> _unconnected = new HashSet<int>();
        /// <summary>The inputs to free after each layer, if they were given with <see cref="GiveInput"/>.</summary>
        private readonly Dictionary<int, List<int>> _inputsToFreeWhenLayerDone = new Dictionary<int, List<int>>();
        /// <summary>The given inputs that are not freed yet, with the run they were given for.</summary>
        private readonly Dictionary<int, int> _given = new Dictionary<int, int>();
        private int _run;

        /// <summary>Free GPU buffers by capacity.</summary>
        private readonly SortedDictionary<int, Stack<ComputeTensorData>> _gpuPool = new SortedDictionary<int, Stack<ComputeTensorData>>();
        private readonly SortedSet<int> _gpuCapacities = new SortedSet<int>();
        /// <summary>Free GPU buffers that no tensor took since the last run started.</summary>
        private readonly HashSet<ComputeTensorData> _idle = new HashSet<ComputeTensorData>();
        /// <summary>Free GPU buffers of given inputs, which the next run start frees if no tensor took them.</summary>
        private readonly HashSet<ComputeTensorData> _fromInputs = new HashSet<ComputeTensorData>();
        private readonly Dictionary<int, Stack<CPUTensorData>> _cpuPool = new Dictionary<int, Stack<CPUTensorData>>();

        public Tensor GetTensor(int index) => index == -1 ? null : _inUse[index];

        public TensorShape GetTensorShape(int tensorIndex) => _inUse[tensorIndex].shape;

        public DataType GetDataType(int tensorIndex) => _inUse[tensorIndex].dataType;

        public int GetInt(int tensorIndex, int defaultValue) {
            if (GetTensor(tensorIndex) is not Tensor<int> tensor) {
                return defaultValue;
            }
            if (tensor.dataOnBackend is CPUTensorData cpu && cpu.IsReadbackRequestDone()) {
                return cpu.array.Get<int>(0);
            }
            return tensor.dataOnBackend.Download<int>(1)[0];
        }

        public float GetFloat(int tensorIndex, float defaultValue) {
            if (GetTensor(tensorIndex) is not Tensor<float> tensor) {
                return defaultValue;
            }
            if (tensor.dataOnBackend is CPUTensorData cpu && cpu.IsReadbackRequestDone()) {
                return cpu.array.Get<float>(0);
            }
            return tensor.dataOnBackend.Download<float>(1)[0];
        }

        public ReadOnlySpan<int> GetInts(int tensorIndex, ReadOnlySpan<int> defaultValue) {
            if (GetTensor(tensorIndex) is not Tensor<int> tensor) {
                return defaultValue;
            }
            if (tensor.dataOnBackend.maxCapacity == 0) {
                return ReadOnlySpan<int>.Empty;
            }
            if (tensor.dataOnBackend is CPUTensorData cpu && cpu.IsReadbackRequestDone()) {
                return cpu.array.AsReadOnlySpan<int>(tensor.shape.length);
            }
            return tensor.dataOnBackend.Download<int>(tensor.shape.length).AsReadOnlySpan();
        }

        public ReadOnlySpan<float> GetFloats(int tensorIndex, ReadOnlySpan<float> defaultValue) {
            if (GetTensor(tensorIndex) is not Tensor<float> tensor) {
                return defaultValue;
            }
            if (tensor.dataOnBackend.maxCapacity == 0) {
                return ReadOnlySpan<float>.Empty;
            }
            if (tensor.dataOnBackend is CPUTensorData cpu && cpu.IsReadbackRequestDone()) {
                return cpu.array.AsReadOnlySpan<float>(tensor.shape.length);
            }
            return tensor.dataOnBackend.Download<float>(tensor.shape.length).AsReadOnlySpan();
        }

        public Tensor AllocateTensor(TensorShape shape, DataType dataType, BackendType backendType) {
            Tensor tensor = dataType switch {
                DataType.Float => new Tensor<float>(shape, data: null),
                DataType.Int => new Tensor<int>(shape, data: null),
                _ => throw new NotImplementedException(),
            };
            int count = shape.length;
            tensor.dataOnBackend = backendType switch {
                BackendType.GPUCompute => AdoptGpu(count) ?? NewGpu(count),
                BackendType.CPU => AdoptCpu(count) ?? new CPUTensorData(count),
                _ => null,
            };
            return tensor;
        }

        public Tensor AllocateTensorAndStore(int index, TensorShape shape, DataType dataType, BackendType backendType) {
            if (index == -1) {
                return null;
            }
            Tensor tensor = AllocateTensor(shape, dataType, backendType);
            Store(index, tensor);
            return tensor;
        }

        public void Store(int index, Tensor result) {
            if (index != -1) {
                _inUse[index] = result;
            }
        }

        public void SetInput(int index, Tensor X) {
            FreeGiven(index);
            _inUse[index] = X;
        }

        /// <summary>
        /// Sets an input that the storage owns: the next run frees it after the last layer that reads it.
        /// </summary>
        public void GiveInput(int index, Tensor X) {
            FreeGiven(index);
            _inUse[index] = X;
            _given[index] = _run + 1;
        }

        private void FreeGiven(int index) {
            if (_given.Remove(index) && _inUse.TryGetValue(index, out var tensor)) {
                if (tensor?.dataOnBackend is ComputeTensorData data) {
                    _fromInputs.Add(data);
                }
                Dispose(tensor);
                _inUse[index] = null;
            }
        }

        public void PrepareStorage(Model model) {
            _disposeWhenLayerDone.Clear();
            _inputsToFreeWhenLayerDone.Clear();
            _given.Clear();
            _inUse.Clear();
            _notOwned = new HashSet<int>(model.inputs.Select(i => i.index));

            var constants = new HashSet<int>();
            foreach (var constant in model.constants) {
                var array = constant.shape.length == 0 ? null : new NativeTensorArrayFromManagedArray(constant.array, constant.shape.length);
                _inUse[constant.index] = AllocatorUtils.AllocTensor(constant.dataType, constant.shape, new CPUTensorData(array));
                constants.Add(constant.index);
            }

            // The last layer that reads each tensor, as ModelStorage finds it. Constants stay until the storage is disposed.
            _unconnected = new HashSet<int>(model.inputs.Select(i => i.index));
            var lastReader = new Dictionary<int, int>();
            foreach (var layer in model.layers) {
                foreach (var input in layer.inputs) {
                    if (input == -1 || constants.Contains(input)) {
                        continue;
                    }
                    _unconnected.Remove(input);
                    lastReader[input] = layer.outputs[0];
                }
                foreach (var output in layer.outputs) {
                    _unconnected.Add(output);
                }
            }
            foreach (var output in model.outputs) {
                if (!constants.Contains(output.index)) {
                    _unconnected.Add(output.index);
                }
            }
            foreach (var (tensor, layer) in lastReader) {
                if (_unconnected.Contains(tensor) || _notOwned.Contains(tensor)) {
                    continue;
                }
                if (!_disposeWhenLayerDone.TryGetValue(layer, out var list)) {
                    _disposeWhenLayerDone[layer] = list = new List<int>();
                }
                list.Add(tensor);
            }
            foreach (var input in model.inputs) {
                _unconnected.Remove(input.index);
                if (lastReader.TryGetValue(input.index, out int layer)) {
                    if (!_inputsToFreeWhenLayerDone.TryGetValue(layer, out var list)) {
                        _inputsToFreeWhenLayerDone[layer] = list = new List<int>();
                    }
                    list.Add(input.index);
                }
            }
        }

        public void DisposeOnExecute() {
            FreeIdleBuffers();
            _run++;
            // Given inputs that the last run did not read.
            foreach (var index in _given.Where(g => g.Value < _run).Select(g => g.Key).ToArray()) {
                FreeGiven(index);
            }
            foreach (var index in _unconnected) {
                if (_inUse.TryGetValue(index, out var tensor)) {
                    Dispose(tensor);
                    _inUse[index] = null;
                }
            }
        }

        public void DisposeAfterLayer(Layer layer) {
            if (_inputsToFreeWhenLayerDone.TryGetValue(layer.outputs[0], out var inputs)) {
                foreach (var index in inputs) {
                    FreeGiven(index);
                }
            }
            if (!_disposeWhenLayerDone.TryGetValue(layer.outputs[0], out var list)) {
                return;
            }
            foreach (var index in list) {
                Dispose(_inUse[index]);
                _inUse[index] = null;
            }
        }

        public void Dispose(Tensor tensor) {
            if (tensor == null) {
                return;
            }
            switch (tensor.dataOnBackend) {
                case ComputeTensorData gpu:
                    ReleaseGpu(gpu);
                    break;
                case CPUTensorData cpu:
                    if (!_cpuPool.TryGetValue(cpu.maxCapacity, out var stack)) {
                        _cpuPool[cpu.maxCapacity] = stack = new Stack<CPUTensorData>();
                    }
                    stack.Push(cpu);
                    break;
                case { } data:
                    data.Dispose();
                    break;
            }
            tensor.dataOnBackend = null;
        }

        public Tensor PeekTensor(int index) {
            if (!_inUse.TryGetValue(index, out var tensor)) {
                D.LogWarning("ModelStorage missing variable: " + index);
            }
            return tensor;
        }

        public Tensor TakeTensorOwnership(int index) {
            if (!_inUse.Remove(index, out var tensor)) {
                D.LogWarning("ModelStorage missing variable: " + index);
            }
            _given.Remove(index);
            return tensor;
        }

        public void Dispose() {
            foreach (var stack in _gpuPool.Values) {
                foreach (var data in stack) {
                    data.Dispose();
                }
            }
            _gpuPool.Clear();
            _gpuCapacities.Clear();
            _idle.Clear();
            _fromInputs.Clear();
            foreach (var stack in _cpuPool.Values) {
                foreach (var data in stack) {
                    data.Dispose();
                }
            }
            _cpuPool.Clear();
            foreach (var input in _notOwned) {
                if (!_given.ContainsKey(input)) {
                    _inUse.Remove(input);
                }
            }
            _given.Clear();
            foreach (var tensor in _inUse.Values) {
                tensor?.Dispose();
            }
            _inUse.Clear();
            _unconnected.Clear();
            _notOwned.Clear();
            _disposeWhenLayerDone.Clear();
        }

        /// <summary>
        /// Takes the smallest free GPU buffer of at least <paramref name="count"/> and at most
        /// <see cref="MaxWaste"/> times <paramref name="count"/> elements, or returns null.
        /// </summary>
        private ComputeTensorData AdoptGpu(int count) {
            if (_gpuCapacities.Count == 0 || count <= 0) {
                return null;
            }
            long max = Math.Min((long)count * MaxWaste, int.MaxValue);
            var fitting = _gpuCapacities.GetViewBetween(count, (int)max);
            if (fitting.Count == 0) {
                return null;
            }
            int capacity = fitting.Min;
            var stack = _gpuPool[capacity];
            var data = stack.Pop();
            if (stack.Count == 0) {
                _gpuPool.Remove(capacity);
                _gpuCapacities.Remove(capacity);
            }
            _idle.Remove(data);
            _fromInputs.Remove(data);
            return data;
        }

        /// <summary>
        /// Frees the GPU buffers that no tensor took since the last call, and marks the others as idle.
        /// </summary>
        private void FreeIdleBuffers() {
            _idle.UnionWith(_fromInputs);
            _fromInputs.Clear();
            if (_idle.Count > 0) {
                foreach (int capacity in _gpuCapacities.ToArray()) {
                    var stack = _gpuPool[capacity];
                    var kept = stack.Where(data => !_idle.Contains(data)).Reverse().ToArray();
                    if (kept.Length == stack.Count) {
                        continue;
                    }
                    foreach (var data in stack) {
                        if (_idle.Contains(data)) {
                            data.Dispose();
                        }
                    }
                    if (kept.Length == 0) {
                        _gpuPool.Remove(capacity);
                        _gpuCapacities.Remove(capacity);
                    } else {
                        _gpuPool[capacity] = new Stack<ComputeTensorData>(kept);
                    }
                }
            }
            _idle.Clear();
            foreach (var stack in _gpuPool.Values) {
                _idle.UnionWith(stack);
            }
        }

        /// <summary>
        /// Allocates a GPU buffer of at least <paramref name="count"/> elements. Above 256 KiB, the size is rounded
        /// up to the step after it, in steps of a quarter of its power of two and at most 4 MiB, like a caching
        /// allocator's size classes. A growing tensor, such as a KV cache, can then take the buffer of the step before.
        /// </summary>
        private ComputeTensorData NewGpu(int count) {
            long bytes = 4L * count;
            if (bytes > 256 << 10) {
                long step = Math.Min(4L << 20, (1L << (63 - LeadingZeros(bytes))) / 4);
                bytes = (bytes / step + 1) * step;
            }
            int capacity = (int)Math.Min(bytes / 4, int.MaxValue);
            return new ComputeTensorData(capacity);
        }

        private static int LeadingZeros(long x) {
            int n = 0;
            for (long bit = 1L << 62; bit > 0 && (x & bit) == 0; bit >>= 1) {
                n++;
            }
            return n + 1;
        }

        private void ReleaseGpu(ComputeTensorData data) {
            int capacity = data.maxCapacity;
            if (!_gpuPool.TryGetValue(capacity, out var stack)) {
                _gpuPool[capacity] = stack = new Stack<ComputeTensorData>();
                _gpuCapacities.Add(capacity);
            }
            stack.Push(data);
        }

        /// <summary>
        /// Takes a free CPU buffer of exactly <paramref name="count"/> elements, or returns null.
        /// </summary>
        private CPUTensorData AdoptCpu(int count) {
            if (!_cpuPool.TryGetValue(count, out var stack) || stack.Count == 0) {
                return null;
            }
            return stack.Pop();
        }
    }
}
