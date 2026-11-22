using System;
using System.Collections.Generic;
using Unity.InferenceEngine;

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
        /// The bytes uploaded before waiting for the GPU. The driver keeps a copy of each upload until the GPU has
        /// it, so without waiting a large model would need its weights in RAM twice.
        /// </summary>
        private const long UploadChunkBytes = 256L << 20;

        /// <summary>
        /// Creates a worker. On <see cref="BackendType.GPUCompute"/>, it uploads the model's weights now and frees
        /// their CPU copy, so the weights are not in RAM and on the GPU at the same time. Constants that layers
        /// read on the CPU stay on the CPU. Afterwards the model cannot create another worker or be saved.
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
            long uploaded = 0;
            foreach (var constant in model.constants) {
                if (StaysOnCpu(constant, readOnCpu)) {
                    continue;
                }
                var data = ComputeTensorData.Pin(storage.GetTensor(constant.index));
                constant.array = default;
                uploaded += constant.lengthBytes;
                if (uploaded >= UploadChunkBytes) {
                    // Reading back one value waits until the GPU has every upload before it.
                    data.buffer.GetData(new int[1], 0, 0, 1);
                    uploaded = 0;
                }
            }
            // The weights stay pinned until a finalizer releases them (NativeTensorArrayFromManagedArray.Dispose
            // does nothing), so they are freed by the second collection.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            return worker;
        }

        private static bool StaysOnCpu(Constant constant, HashSet<int> readOnCpu) {
            return constant.shape.length == 0 || readOnCpu.Contains(constant.index);
        }
    }
}
