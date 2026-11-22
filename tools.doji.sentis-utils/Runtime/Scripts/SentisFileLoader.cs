using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.InferenceEngine;
using Unity.IO.LowLevel.Unsafe;

namespace Doji.AI {

    /// <summary>
    /// Loads a .sentis file whose weights stay in the file until a worker needs them. On
    /// <see cref="BackendType.GPUCompute"/>, a worker reads each weight from the file into native memory and uploads
    /// it, so the weights are never in managed memory.
    /// </summary>
    public static class SentisFileLoader {

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

        private class ModelFile {
            public string Path;
            /// <summary>The file offset of each constant, by its position in <see cref="Model.constants"/>.</summary>
            public long[] Offsets;
        }

        private static readonly ConditionalWeakTable<Model, ModelFile> Files = new();

        /// <summary>
        /// Loads the model of the .sentis file at <paramref name="path"/>. It holds only the constants that layers read
        /// on the CPU; <see cref="CreateWorker"/> reads the other weights from the file, which must stay in place.
        /// </summary>
        public static Model Load(string path) {
            var model = new Model();
            var file = new ModelFile { Path = path };
            using (var stream = File.OpenRead(path)) {
                byte[] description = ReadSized(stream);
                var chunkOffsets = (List<(int constant, int offset)>[])LoadModelDescription.Invoke(null, new object[] { description, model });
                file.Offsets = new long[model.constants.Count];
                foreach (var chunk in chunkOffsets) {
                    long storageStart = FindStorage(stream);
                    foreach (var (constant, offset) in chunk) {
                        file.Offsets[constant] = storageStart + offset;
                    }
                }
            }

            var readOnCpu = CPUFallbackCalculator.Calculate(model, BackendType.GPUCompute, out _);
            for (int i = 0; i < model.constants.Count; i++) {
                var constant = model.constants[i];
                if (constant.shape.length != 0 && readOnCpu.Contains(constant.index)) {
                    var bytes = new byte[Padded(constant.lengthBytes)];
                    ReadInto(path, file.Offsets[i], bytes, constant.lengthBytes);
                    constant.array = new ArraySegment<byte>(bytes, 0, constant.lengthBytes);
                }
            }
            Files.Add(model, file);
            return model;
        }

        /// <summary>
        /// Whether <paramref name="model"/> was loaded by <see cref="Load"/>.
        /// </summary>
        public static bool IsFromFile(Model model) => Files.TryGetValue(model, out _);

        /// <summary>
        /// Creates a worker for a model of <see cref="Load"/>, reading its weights from the file.
        /// </summary>
        public static Worker CreateWorker(Model model, BackendType backendType) {
            if (!Files.TryGetValue(model, out var file)) {
                throw new ArgumentException($"The model was not loaded by {nameof(SentisFileLoader)}.{nameof(Load)}.", nameof(model));
            }
            if (backendType != BackendType.GPUCompute) {
                return new Worker(ModelLoader.Load(file.Path), backendType);
            }

            int largest = 0;
            foreach (var constant in model.constants) {
                largest = Math.Max(largest, Padded(constant.lengthBytes));
            }
            // The worker's storage wraps the CPU data of each constant; this placeholder stands in for the weights
            // that go to the GPU, which are replaced before they are used.
            var placeholder = new byte[largest];
            foreach (var constant in model.constants) {
                if (OnGpu(constant)) {
                    constant.array = new ArraySegment<byte>(placeholder, 0, constant.lengthBytes);
                }
            }

            var storage = new ModelStorage();
            var worker = new Worker(model, BackendFactory.CreateBackend(backendType), storage);
            using var staging = new NativeArray<int>(largest / sizeof(int), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            long uploaded = 0;
            for (int i = 0; i < model.constants.Count; i++) {
                var constant = model.constants[i];
                if (constant.array.Array != placeholder) {
                    continue;
                }
                Tensor tensor = storage.GetTensor(constant.index);
                var data = new ComputeTensorData(tensor.count);
                tensor.AdoptTensorData(data, disposePrevious: true, disposeIsDelayed: false);
                ReadInto(file.Path, file.Offsets[i], staging, constant.lengthBytes);
                data.buffer.SetData(staging, 0, 0, Padded(constant.lengthBytes) / sizeof(int));
                constant.array = default;
                uploaded += constant.lengthBytes;
                if (uploaded >= UploadChunkBytes) {
                    // Reading back one value waits until the GPU has every upload before it.
                    data.buffer.GetData(new int[1], 0, 0, 1);
                    uploaded = 0;
                }
            }
            return worker;
        }

        private static bool OnGpu(Constant constant) => constant.shape.length != 0 && constant.array.Array == null;

        private static int Padded(int bytes) => (bytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);

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

        private static unsafe void ReadInto(string path, long offset, byte[] buffer, int bytes) {
            fixed (byte* pointer = buffer) {
                Read(path, offset, pointer, bytes);
            }
        }

        private static unsafe void ReadInto(string path, long offset, NativeArray<int> buffer, int bytes) {
            Read(path, offset, buffer.GetUnsafePtr(), bytes);
        }

        private static unsafe void Read(string path, long offset, void* buffer, long bytes) {
            var command = new ReadCommand { Offset = offset, Size = bytes, Buffer = buffer };
            ReadHandle handle = AsyncReadManager.Read(path, &command, 1);
            handle.JobHandle.Complete();
            ReadStatus status = handle.Status;
            handle.Dispose();
            if (status != ReadStatus.Complete) {
                throw new IOException($"Could not read {bytes} bytes at {offset} of {path}: {status}");
            }
        }
    }
}
