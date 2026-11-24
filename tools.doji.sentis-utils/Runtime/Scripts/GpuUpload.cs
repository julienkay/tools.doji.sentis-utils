using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace Doji.AI {

    /// <summary>
    /// Uploads data to GPU buffers through two small staging buffers that the CPU writes, and a kernel that copies
    /// each part to its buffer. <see cref="ComputeBuffer.SetData(Array)"/> stages each upload in memory that the
    /// graphics driver frees only some frames later, so uploading a whole model without frames between needs RAM for
    /// all of its weights at once. This needs RAM only for the staging buffers.
    /// </summary>
    internal sealed unsafe class GpuUpload : IDisposable {

        /// <summary>The 32-bit values of each staging buffer.</summary>
        private const int ChunkInts = 8 << 20;
        private const int Threads = 256;

        private static ComputeShader _shader;
        private static int _kernel;
        private static readonly int SrcId = Shader.PropertyToID("Src");
        private static readonly int DstId = Shader.PropertyToID("Dst");
        private static readonly int CountId = Shader.PropertyToID("Count");
        private static readonly int DstOffsetId = Shader.PropertyToID("DstOffset");

        private readonly ComputeBuffer[] _staging = new ComputeBuffer[2];
        /// <summary>For each staging buffer, a readback that completes when the GPU has copied its data.</summary>
        private readonly AsyncGPUReadbackRequest?[] _copied = new AsyncGPUReadbackRequest?[2];
        private int _next;

        public GpuUpload() {
            if (_shader == null) {
                _shader = Resources.Load<ComputeShader>("ComputeShaders/Compute.Shaders.CopyInts");
                _kernel = _shader.FindKernel("CopyInts");
            }
        }

        /// <summary>
        /// Copies <paramref name="ints"/> 32-bit values from <paramref name="source"/> to the start of
        /// <paramref name="destination"/>. The source can be reused when this returns.
        /// </summary>
        public void Upload(void* source, ComputeBuffer destination, int ints) {
            for (int offset = 0; offset < ints; offset += ChunkInts) {
                int count = Math.Min(ChunkInts, ints - offset);
                int slot = _next;
                _next = 1 - _next;
                Wait(slot);
                _staging[slot] ??= new ComputeBuffer(ChunkInts, sizeof(int), ComputeBufferType.Structured, ComputeBufferMode.SubUpdates);
                var staging = _staging[slot].BeginWrite<int>(0, count);
                UnsafeUtility.MemCpy(staging.GetUnsafePtr(), (int*)source + offset, (long)count * sizeof(int));
                _staging[slot].EndWrite<int>(count);

                _shader.SetBuffer(_kernel, SrcId, _staging[slot]);
                _shader.SetBuffer(_kernel, DstId, destination);
                _shader.SetInt(CountId, count);
                _shader.SetInt(DstOffsetId, offset);
                _shader.Dispatch(_kernel, (count + Threads - 1) / Threads, 1, 1);
                _copied[slot] = AsyncGPUReadback.Request(destination, sizeof(int), offset * sizeof(int));
            }
        }

        private void Wait(int slot) {
            if (_copied[slot].HasValue) {
                _copied[slot].Value.WaitForCompletion();
                _copied[slot] = null;
            }
        }

        public void Dispose() {
            for (int slot = 0; slot < _staging.Length; slot++) {
                Wait(slot);
                _staging[slot]?.Release();
                _staging[slot] = null;
            }
        }
    }
}
