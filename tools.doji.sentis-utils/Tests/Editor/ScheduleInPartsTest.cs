using System.Collections;
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Doji.AI.Editor.Tests {

    public class ScheduleInPartsTest {

        private const int Width = 16;

        /// <summary>
        /// A chain of small layers with weights, and two outputs.
        /// </summary>
        private static Model CreateModel() {
            var graph = new FunctionalGraph();
            var x = graph.AddInput<float>(new DynamicTensorShape(-1, Width));
            var random = new System.Random(3);
            var weights = new float[Width * Width];
            for (int i = 0; i < weights.Length; i++) {
                weights[i] = (float)(random.NextDouble() - 0.5) / Width;
            }
            var w = Functional.Constant(new TensorShape(Width, Width), weights);
            var y = x;
            for (int i = 0; i < 40; i++) {
                y = Functional.Tanh(Functional.MatMul(y, w) * 1.5f + 0.1f) + y;
            }
            return graph.Compile(y, y.Reshape(new[] { -1 }));
        }

        [TestCase(BackendType.GPUCompute)]
        [TestCase(BackendType.CPU)]
        public void Outputs_MatchSchedule(BackendType backend) {
            var model = CreateModel();
            var data = new float[3 * Width];
            for (int i = 0; i < data.Length; i++) {
                data[i] = i % 7 - 3;
            }
            using var input = new Tensor<float>(new TensorShape(3, Width), data);

            using var reference = new Worker(model, backend);
            reference.Schedule(input);
            float[] expected0 = Read(reference.PeekOutput(0));
            float[] expected1 = Read(reference.PeekOutput(1));

            using var worker = new Worker(model, backend);
            using var commandBuffer = new CommandBuffer();
            worker.SetInput(0, input);
            IEnumerator run = commandBuffer.ScheduleInParts(worker, 0);
            int parts = 1;
            while (run.MoveNext()) {
                parts++;
            }
            // A budget of 0 ms makes each layer a part.
            Assert.That(parts, Is.GreaterThan(40));
            Assert.That(Read(worker.PeekOutput(0)), Is.EqualTo(expected0));
            Assert.That(Read(worker.PeekOutput(1)), Is.EqualTo(expected1));

            // A large budget runs the model in one part.
            worker.SetInput(0, input);
            run = commandBuffer.ScheduleInParts(worker, 1e6);
            Assert.That(run.MoveNext(), Is.False);
            Assert.That(Read(worker.PeekOutput(0)), Is.EqualTo(expected0));

            // The worker keeps the command buffer: a later Schedule records into it.
            if (backend == BackendType.GPUCompute) {
                worker.SetInput(0, input);
                commandBuffer.ScheduleWorker(worker);
                Graphics.ExecuteCommandBuffer(commandBuffer);
                commandBuffer.Clear();
                Assert.That(Read(worker.PeekOutput(1)), Is.EqualTo(expected1));
            }
        }

        /// <summary>
        /// A worker of <see cref="ModelFile.CreateWorker"/> frees a given input after the layer that reads it,
        /// in a run in parts too.
        /// </summary>
        [Test]
        public void FittedModelStorage_GivenInput_MatchesSchedule() {
            var model = CreateModel();
            string path = Path.Combine(Path.GetTempPath(), "ScheduleInPartsTest.sentis");
            ModelWriter.Save(path, model);
            var data = new float[2 * Width];
            for (int i = 0; i < data.Length; i++) {
                data[i] = 0.25f * i - 2;
            }
            float[] expected;
            using (var file = ModelFile.Open(path)) {
                using var reference = file.CreateWorker(BackendType.GPUCompute);
                using var input = new Tensor<float>(new TensorShape(2, Width), data);
                reference.Schedule(input);
                expected = Read(reference.PeekOutput(0));

                using var worker = file.CreateWorker(BackendType.GPUCompute);
                using var commandBuffer = new CommandBuffer();
                var given = new Tensor<float>(new TensorShape(2, Width), data);
                Assert.That(worker.TryGiveInput(0, given), Is.True);
                IEnumerator run = commandBuffer.ScheduleInParts(worker, 0);
                while (run.MoveNext()) { }
                Assert.That(Read(worker.PeekOutput(0)), Is.EqualTo(expected));
            }
            File.Delete(path);
        }

        private static float[] Read(Tensor tensor) {
            using var cpu = ((Tensor<float>)tensor).ReadbackAndClone();
            return cpu.DownloadToArray();
        }
    }
}
