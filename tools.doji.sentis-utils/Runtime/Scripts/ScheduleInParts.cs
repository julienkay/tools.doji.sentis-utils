using System;
using System.Collections;
using System.Diagnostics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Doji.AI {

    public static class WorkerScheduling {

        /// <summary>
        /// Schedules the run of the worker in parts: each <c>MoveNext</c> of the returned enumerator schedules
        /// layers until <paramref name="maxMillisecondsPerPart"/> of CPU time has passed, then executes
        /// <paramref name="commandBuffer"/> once. Call <c>MoveNext</c> once per frame to spread one run over several
        /// frames, so that no frame gets the whole cost. The run is done when <c>MoveNext</c> returns false.
        /// </summary>
        /// <remarks>
        /// <para>
        /// On <see cref="BackendType.GPUCompute"/>, it runs <c>commandBuffer.ScheduleWorkerIterable(worker)</c>, so the
        /// worker records its GPU work into <paramref name="commandBuffer"/>, and this method executes it at the end
        /// of each part. The worker keeps using <paramref name="commandBuffer"/> afterwards: after a later
        /// <see cref="Worker.Schedule()"/> of this worker, or <c>commandBuffer.ScheduleWorker(worker)</c>, execute and
        /// clear <paramref name="commandBuffer"/> yourself before reading outputs.
        /// </para>
        /// <para>
        /// The outputs are the same as <see cref="Worker.Schedule()"/>. Each part schedules at least one layer.
        /// Don't set inputs, schedule or read outputs of the worker until the run is done. On other backends,
        /// <paramref name="commandBuffer"/> is not used and can be null.
        /// </para>
        /// </remarks>
        public static IEnumerator ScheduleInParts(this CommandBuffer commandBuffer, Worker worker, double maxMillisecondsPerPart) {
            var clock = new Stopwatch();
            return commandBuffer.ScheduleInParts(worker, () => {
                if (!clock.IsRunning) {
                    clock.Start();
                }
                if (clock.Elapsed.TotalMilliseconds < maxMillisecondsPerPart) {
                    return false;
                }
                clock.Reset();
                return true;
            });
        }

        /// <summary>
        /// Schedules the run of the worker in parts, as <see cref="ScheduleInParts(CommandBuffer, Worker, double)"/>
        /// does, and asks <paramref name="endPart"/> after each layer whether the part ends there. For example, a
        /// frame budget shared with other work.
        /// </summary>
        public static IEnumerator ScheduleInParts(this CommandBuffer commandBuffer, Worker worker, Func<bool> endPart) {
            if (worker == null) {
                throw new ArgumentNullException(nameof(worker));
            }
            if (endPart == null) {
                throw new ArgumentNullException(nameof(endPart));
            }
            bool gpu = worker.backendType == BackendType.GPUCompute;
            if (gpu && commandBuffer == null) {
                throw new ArgumentNullException(nameof(commandBuffer));
            }
            return Run(gpu ? commandBuffer : null, gpu ? commandBuffer.ScheduleWorkerIterable(worker) : worker.ScheduleIterable(), endPart);
        }

        private static IEnumerator Run(CommandBuffer commandBuffer, IEnumerator schedule, Func<bool> endPart) {
            while (schedule.MoveNext()) {
                if (endPart()) {
                    Execute(commandBuffer);
                    yield return null;
                }
            }
            Execute(commandBuffer);
        }

        private static void Execute(CommandBuffer commandBuffer) {
            if (commandBuffer != null) {
                Graphics.ExecuteCommandBuffer(commandBuffer);
                commandBuffer.Clear();
            }
        }
    }
}
