using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.InferenceEngine;
using Unity.IO.LowLevel.Unsafe;

namespace Doji.AI {

    /// <summary>
    /// An open .sentis file whose weights stay in the file until a worker needs them. On
    /// <see cref="BackendType.GPUCompute"/> and <see cref="BackendType.CPU"/>, <see cref="CreateWorker"/> reads each
    /// weight into native memory, and uploads it to the GPU, so the weights are never in managed memory. The file stays open until the handle is disposed, so on
    /// Windows it cannot be deleted or moved meanwhile.
    /// </summary>
    public sealed unsafe partial class ModelFile : IDisposable {

        /// <summary>
        /// The bytes uploaded before waiting for the GPU, as in <see cref="WorkerUtils.CreateWorkerWithWeightsOnGpu"/>.
        /// </summary>
        private const long UploadChunkBytes = 256L << 20;

        /// <summary>
        /// ModelLoader's private LoadModelDescription(byte[], Model), which returns the offset of each constant in
        /// its weights chunk.
        /// </summary>
        private static readonly MethodInfo LoadModelDescription = typeof(ModelLoader).GetMethod("LoadModelDescription",
            BindingFlags.NonPublic | BindingFlags.Static, null, new[] { typeof(byte[]), typeof(Model) }, null);

        /// <summary>
        /// The model without the weights that go to the GPU. Layers read its other constants on the CPU.
        /// </summary>
        private readonly Model _model;
        private readonly FileStream _stream;
        /// <summary>The file offset of each constant, by its position in <see cref="Model.constants"/>.</summary>
        private readonly long[] _offsets;
        private readonly int _largest;

        /// <summary>The copy of the weights in native memory, or null.</summary>
        private byte* _weights;
        /// <summary>The offset of each constant in <see cref="_weights"/>, by its position in <see cref="Model.constants"/>.</summary>
        private long[] _weightOffsets;

        public string Path { get; }
        public IReadOnlyList<Model.Input> Inputs => _model.inputs;
        public IReadOnlyList<Model.Output> Outputs => _model.outputs;

        private ModelFile(string path) {
            Path = path;
            _model = new Model();
            _stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            try {
                byte[] description = ReadSized(_stream);
                var chunkOffsets = (List<(int constant, int offset)>[])LoadModelDescription.Invoke(null, new object[] { description, _model });
                _offsets = new long[_model.constants.Count];
                foreach (var chunk in chunkOffsets) {
                    long storageStart = FindStorage(_stream);
                    foreach (var (constant, offset) in chunk) {
                        _offsets[constant] = storageStart + offset;
                    }
                }

                var readOnCpu = CPUFallbackCalculator.Calculate(_model, BackendType.GPUCompute, out _);
                for (int i = 0; i < _model.constants.Count; i++) {
                    var constant = _model.constants[i];
                    _largest = Math.Max(_largest, Padded(constant.lengthBytes));
                    if (constant.shape.length != 0 && readOnCpu.Contains(constant.index)) {
                        var bytes = new byte[Padded(constant.lengthBytes)];
                        fixed (byte* pointer = bytes) {
                            Read(_offsets[i], pointer, constant.lengthBytes);
                        }
                        constant.array = new ArraySegment<byte>(bytes, 0, constant.lengthBytes);
                    }
                }
            } catch {
                _stream.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Opens the .sentis file at <paramref name="path"/> and reads its model description.
        /// </summary>
        public static ModelFile Open(string path) => new ModelFile(path);

        /// <summary>
        /// Loads the whole model with its weights, for example to change or save it.
        /// </summary>
        public Model LoadModel() => ModelLoader.Load(Path);

        /// <summary>
        /// Whether a copy of the weights stays in native memory, so <see cref="CreateWorker"/> uploads them from
        /// RAM instead of reading the file. Setting it reads or frees the copy.
        /// </summary>
        public bool KeepWeightsInMemory {
            get => _weights != null;
            set {
                if (value == KeepWeightsInMemory) {
                    return;
                }
                if (!value) {
                    FreeWeights();
                    return;
                }
                _weightOffsets = new long[_model.constants.Count];
                long size = 0;
                for (int i = 0; i < _model.constants.Count; i++) {
                    if (OnGpu(_model.constants[i])) {
                        _weightOffsets[i] = size;
                        size += Padded(_model.constants[i].lengthBytes);
                    }
                }
                _weights = (byte*)UnsafeUtility.Malloc(Math.Max(size, 1), 16, Allocator.Persistent);
                for (int i = 0; i < _model.constants.Count; i++) {
                    if (OnGpu(_model.constants[i])) {
                        Read(_offsets[i], _weights + _weightOffsets[i], _model.constants[i].lengthBytes);
                    }
                }
            }
        }

        /// <summary>
        /// Creates a worker. On <see cref="BackendType.GPUCompute"/>, the weights go from the file, or the copy in
        /// memory, to the GPU. On <see cref="BackendType.CPU"/>, they go into native memory. Other backends load the
        /// whole model.
        /// </summary>
        public Worker CreateWorker(BackendType backendType) {
            if (_stream == null || !_stream.CanRead) {
                throw new ObjectDisposedException(nameof(ModelFile));
            }
            if (backendType != BackendType.GPUCompute && backendType != BackendType.CPU) {
                return new Worker(LoadModel(), backendType);
            }

            var storage = new ModelStorage();
            var weights = new List<int>();
            Worker worker = CreatePlaceholderWorker(backendType, storage, weights);
            try {
                if (backendType == BackendType.GPUCompute) {
                    Upload(storage, weights);
                } else {
                    foreach (int i in weights) {
                        Tensor tensor = storage.GetTensor(_model.constants[i].index);
                        var data = new CPUTensorData(tensor.count);
                        tensor.AdoptTensorData(data, disposePrevious: true, disposeIsDelayed: false);
                        Copy(i, data.array.RawPtr);
                    }
                }
            } catch {
                worker.Dispose();
                throw;
            }
            return worker;
        }

        /// <summary>
        /// Creates a worker whose weights that are not read on the CPU are a placeholder, and adds their positions in
        /// the constants to <paramref name="weights"/>. Replace each placeholder before a layer reads it.
        /// </summary>
        private Worker CreatePlaceholderWorker(BackendType backendType, ModelStorage storage, List<int> weights) {
            // The storage wraps the CPU data of each constant when the worker is created.
            var placeholder = new byte[_largest];
            for (int i = 0; i < _model.constants.Count; i++) {
                if (OnGpu(_model.constants[i])) {
                    _model.constants[i].array = new ArraySegment<byte>(placeholder, 0, _model.constants[i].lengthBytes);
                    weights.Add(i);
                }
            }
            try {
                return new Worker(_model, BackendFactory.CreateBackend(backendType), storage);
            } finally {
                foreach (int i in weights) {
                    _model.constants[i].array = default;
                }
            }
        }

        /// <summary>
        /// Uploads the weights to the GPU. From the file, it reads the next weight while it uploads one.
        /// </summary>
        private void Upload(ModelStorage storage, List<int> weights) {
            var staging = new NativeArray<int>[2];
            var commands = new NativeArray<ReadCommand>(2, Allocator.Persistent);
            ReadHandle pending = default;
            try {
                if (_weights == null) {
                    for (int b = 0; b < staging.Length; b++) {
                        staging[b] = new NativeArray<int>(_largest / sizeof(int), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    }
                    if (weights.Count > 0) {
                        pending = StartRead(weights[0], staging[0].GetUnsafePtr(), commands, 0);
                    }
                }
                long uploaded = 0;
                for (int k = 0; k < weights.Count; k++) {
                    int i = weights[k];
                    var constant = _model.constants[i];
                    int ints = Padded(constant.lengthBytes) / sizeof(int);
                    NativeArray<int> source;
                    if (_weights != null) {
                        source = View(_weights + _weightOffsets[i], ints);
                    } else {
                        Finish(ref pending, i);
                        if (k + 1 < weights.Count) {
                            int next = (k + 1) % 2;
                            pending = StartRead(weights[k + 1], staging[next].GetUnsafePtr(), commands, next);
                        }
                        source = staging[k % 2];
                    }
                    Tensor tensor = storage.GetTensor(constant.index);
                    var data = new ComputeTensorData(tensor.count);
                    tensor.AdoptTensorData(data, disposePrevious: true, disposeIsDelayed: false);
                    data.buffer.SetData(source, 0, 0, ints);
                    uploaded += constant.lengthBytes;
                    if (uploaded >= UploadChunkBytes) {
                        // Reading back one value waits until the GPU has every upload before it.
                        data.buffer.GetData(new int[1], 0, 0, 1);
                        uploaded = 0;
                    }
                }
            } finally {
                if (pending.IsValid()) {
                    pending.JobHandle.Complete();
                    pending.Dispose();
                }
                commands.Dispose();
                foreach (var buffer in staging) {
                    if (buffer.IsCreated) {
                        buffer.Dispose();
                    }
                }
            }
        }

        /// <summary>
        /// Copies the constant at position <paramref name="i"/> to <paramref name="destination"/>, from the copy in
        /// memory or from the file.
        /// </summary>
        private void Copy(int i, void* destination) {
            if (_weights != null) {
                UnsafeUtility.MemCpy(destination, _weights + _weightOffsets[i], _model.constants[i].lengthBytes);
                return;
            }
            Read(_offsets[i], destination, _model.constants[i].lengthBytes);
        }

        /// <summary>
        /// Starts to read the constant at position <paramref name="i"/> from the file, like <see cref="Read"/>. The
        /// command stays in <paramref name="commands"/> until the read is finished.
        /// </summary>
        private ReadHandle StartRead(int i, void* buffer, NativeArray<ReadCommand> commands, int slot) {
            var command = (ReadCommand*)commands.GetUnsafePtr() + slot;
            *command = new ReadCommand { Offset = _offsets[i], Size = _model.constants[i].lengthBytes, Buffer = buffer };
            return AsyncReadManager.Read(Path, command, 1);
        }

        private void Finish(ref ReadHandle handle, int i) {
            handle.JobHandle.Complete();
            ReadStatus status = handle.Status;
            handle.Dispose();
            handle = default;
            if (status != ReadStatus.Complete) {
                throw new IOException($"Could not read {_model.constants[i].lengthBytes} bytes at {_offsets[i]} of {Path}: {status}");
            }
        }

        /// <summary>
        /// A NativeArray over native memory that this class owns.
        /// </summary>
        private static NativeArray<int> View(void* pointer, int length) {
            var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<int>(pointer, length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
            return array;
        }

        public void Dispose() {
            FreeWeights();
            _stream.Dispose();
        }

        private void FreeWeights() {
            if (_weights != null) {
                UnsafeUtility.Free(_weights, Allocator.Persistent);
                _weights = null;
                _weightOffsets = null;
            }
        }

        private static bool OnGpu(Constant constant) => constant.shape.length != 0 && constant.array.Array == null;

        private static int Padded(int bytes) => (bytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);

        /// <summary>
        /// Reads with AsyncReadManager, because a FileStream copies through a managed array to read into native memory.
        /// </summary>
        private void Read(long offset, void* buffer, int bytes) {
            var command = new ReadCommand { Offset = offset, Size = bytes, Buffer = buffer };
            ReadHandle handle = AsyncReadManager.Read(Path, &command, 1);
            handle.JobHandle.Complete();
            ReadStatus status = handle.Status;
            handle.Dispose();
            if (status != ReadStatus.Complete) {
                throw new IOException($"Could not read {bytes} bytes at {offset} of {Path}: {status}");
            }
        }

        /// <summary>
        /// Reads a block that starts with its size as an int, and returns it with the size.
        /// </summary>
        private static byte[] ReadSized(Stream stream) {
            var size = new byte[sizeof(int)];
            ReadExactly(stream, size, 0, size.Length);
            var block = new byte[sizeof(int) + BitConverter.ToInt32(size)];
            Buffer.BlockCopy(size, 0, block, 0, size.Length);
            ReadExactly(stream, block, sizeof(int), block.Length - sizeof(int));
            return block;
        }

        /// <summary>
        /// Returns the file offset of the storage of the weights chunk at the position of the stream, and moves the
        /// stream past the chunk. A chunk is its size as an int, then a flatbuffer Buffer whose first field is the
        /// storage vector. The flatbuffer offsets count from the size.
        /// </summary>
        private static long FindStorage(Stream stream) {
            long start = stream.Position;
            var head = new byte[Math.Min(4096, stream.Length - start)];
            ReadExactly(stream, head, 0, head.Length);
            int size = BitConverter.ToInt32(head, 0);
            int table = sizeof(int) + BitConverter.ToInt32(head, sizeof(int));
            int vtable = table - BitConverter.ToInt32(head, table);
            int field = BitConverter.ToUInt16(head, vtable + 4);
            int vectorRef = table + field;
            int vector = vectorRef + BitConverter.ToInt32(head, vectorRef);
            int length = BitConverter.ToInt32(head, vector);
            if (vector + sizeof(int) + (long)length > sizeof(int) + (long)size) {
                throw new InvalidDataException($"The weights chunk at {start} has a storage of {length} bytes, more than its {size} bytes.");
            }
            stream.Position = start + sizeof(int) + size;
            return start + vector + sizeof(int);
        }

        private static void ReadExactly(Stream stream, byte[] buffer, int offset, int count) {
            while (count > 0) {
                int read = stream.Read(buffer, offset, count);
                if (read == 0) {
                    throw new EndOfStreamException();
                }
                offset += read;
                count -= read;
            }
        }
    }
}
