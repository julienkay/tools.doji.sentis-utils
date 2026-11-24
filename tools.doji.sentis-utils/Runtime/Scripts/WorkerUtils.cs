using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.InferenceEngine;
using UnityEngine;

namespace Doji.AI {
    public static class WorkerUtils {
        public static void Schedule(this Worker worker, Dictionary<string, Tensor> inputs) {
            foreach (var input in inputs) {
                worker.SetInput(input.Key, input.Value);
            }
            worker.Schedule();
        }
        public static System.Collections.IEnumerator ScheduleIterable(this Worker worker, Dictionary<string, Tensor> inputs) {
            foreach (var input in inputs) {
                worker.SetInput(input.Key, input.Value);
            }
            return worker.ScheduleIterable();
        }

        /// <summary>
        /// The model and storage of each worker that <see cref="CreateWorkerWithWeightsOnGpu"/> made.
        /// </summary>
        private static readonly ConditionalWeakTable<Worker, Tuple<Model, ModelStorage>> GpuWorkers = new();

        /// <summary>
        /// Creates a worker. On <see cref="BackendType.GPUCompute"/>, it uploads the model's weights now and frees
        /// their CPU copy, so the weights are not in RAM and on the GPU at the same time. Constants that layers
        /// read on the CPU stay on the CPU. Until <see cref="DisposeAndMoveWeightsToCpu"/>, the model cannot create
        /// another worker or be saved.
        /// </summary>
        public static Worker CreateWorkerWithWeightsOnGpu(Model model, BackendType backendType) {
            if (backendType != BackendType.GPUCompute) {
                return new Worker(model, backendType);
            }

            var readOnCpu = CPUFallbackCalculator.Calculate(model, backendType, out _);
            foreach (var constant in model.constants) {
                if (StaysOnCpu(constant, readOnCpu) && constant.array.Array != null) {
                    // A loaded .sentis model keeps its weights in large shared arrays. A copy lets those be freed.
                    constant.array = new ArraySegment<byte>(constant.array.ToArray());
                }
            }

            var storage = new ModelStorage();
            var worker = new Worker(model, BackendFactory.CreateBackend(backendType), storage);
            using (var upload = new GpuUpload()) {
                foreach (var constant in model.constants) {
                    if (StaysOnCpu(constant, readOnCpu)) {
                        continue;
                    }
                    Tensor tensor = storage.GetTensor(constant.index);
                    var data = new ComputeTensorData(tensor.count);
                    Upload(upload, constant.array, data.buffer);
                    tensor.AdoptTensorData(data, disposePrevious: true, disposeIsDelayed: false);
                    constant.array = default;
                }
            }
            // The weights stay pinned until a finalizer releases them (NativeTensorArrayFromManagedArray.Dispose
            // does nothing), so they are freed by the second collection.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GpuWorkers.Add(worker, Tuple.Create(model, storage));
            return worker;
        }

        /// <summary>
        /// Disposes a worker of <see cref="CreateWorkerWithWeightsOnGpu"/> and moves the model's weights from the
        /// GPU back into the model, so the model can create a worker again. Other workers are only disposed.
        /// </summary>
        public static void DisposeAndMoveWeightsToCpu(this Worker worker) {
            if (GpuWorkers.TryGetValue(worker, out var gpu)) {
                foreach (var constant in gpu.Item1.constants) {
                    if (constant.array.Array != null || constant.shape.length == 0) {
                        continue;
                    }
                    var buffer = ComputeTensorData.Pin(gpu.Item2.GetTensor(constant.index)).buffer;
                    var ints = new int[buffer.count];
                    buffer.GetData(ints);
                    var bytes = new byte[Math.Max(ints.Length * sizeof(int), constant.lengthBytes)];
                    Buffer.BlockCopy(ints, 0, bytes, 0, ints.Length * sizeof(int));
                    constant.array = new ArraySegment<byte>(bytes, 0, constant.lengthBytes);
                }
                GpuWorkers.Remove(worker);
            }
            worker.Dispose();
        }

        private static readonly System.Reflection.FieldInfo WorkerModel = typeof(Worker).GetField("m_Model",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        private static readonly System.Reflection.FieldInfo WorkerStorage = typeof(Worker).GetField("m_Storage",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        /// <summary>
        /// Sets input <paramref name="index"/> and, if the worker supports it, gives the tensor to the worker: the
        /// next <see cref="Worker.Schedule()"/> frees it after the last layer that reads it, so it does not stay in
        /// memory while the rest of the model runs. Workers of <see cref="ModelFile.CreateWorker"/> support it.
        /// </summary>
        /// <returns>
        /// True if the worker owns the tensor now: do not use or dispose it. False if the input is set and the
        /// caller still owns the tensor.
        /// </returns>
        public static bool TryGiveInput(this Worker worker, int index, Tensor input) {
            worker.SetInput(index, input);
            if (WorkerStorage.GetValue(worker) is not FittedModelStorage storage) {
                return false;
            }
            storage.GiveInput(index, input);
            return true;
        }

        /// <summary>
        /// Takes output <paramref name="index"/> from the worker without a copy. The caller owns and disposes the
        /// tensor, and the next run of the worker writes a new one.
        /// </summary>
        public static Tensor TakeOutput(this Worker worker, int index) {
            var model = (Model)WorkerModel.GetValue(worker);
            var storage = (IModelStorage)WorkerStorage.GetValue(worker);
            return storage.TakeTensorOwnership(model.outputs[index].index);
        }

        /// <summary>
        /// Uploads <paramref name="bytes"/> to the start of <paramref name="destination"/>, in 32-bit values.
        /// </summary>
        private static unsafe void Upload(GpuUpload upload, ArraySegment<byte> bytes, ComputeBuffer destination) {
            if (bytes.Count % sizeof(int) != 0) {
                var padded = new byte[(bytes.Count + sizeof(int) - 1) / sizeof(int) * sizeof(int)];
                Buffer.BlockCopy(bytes.Array, bytes.Offset, padded, 0, bytes.Count);
                bytes = new ArraySegment<byte>(padded);
            }
            fixed (byte* source = &bytes.Array[bytes.Offset]) {
                upload.Upload(source, destination, bytes.Count / sizeof(int));
            }
        }

        private static bool StaysOnCpu(Constant constant, HashSet<int> readOnCpu) {
            return constant.shape.length == 0 || readOnCpu.Contains(constant.index);
        }
    }
}
