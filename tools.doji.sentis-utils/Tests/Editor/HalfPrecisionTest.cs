using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace Doji.AI.Editor.Tests {

    public class HalfPrecisionTest {

        /// <summary>
        /// A model that reads a float16 input as float32, and writes a float32 result as a float16 output.
        /// Five values check the unpaired last value of an odd length.
        /// </summary>
        [Test]
        public void Apply_CastsInputsAndOutputs() {
            var graph = new FunctionalGraph();
            var x = graph.AddInput<float>(new DynamicTensorShape(-1));
            Model model = graph.Compile(x * 2f, x + 1f);
            string path = Path.Combine(Path.GetTempPath(), "HalfPrecisionTest.sentis");
            ModelWriter.Save(path, model);
            using var file = ModelFile.Open(path);
            int changed = file.UseHalfPrecision(new[] { model.inputs[0].name }, new[] { model.outputs[0].name });
            Assert.That(changed, Is.EqualTo(2));
            Assert.That(file.Inputs[0].dataType, Is.EqualTo(DataType.Short));

            float[] values = { 0.5f, -1.25f, 3f, 1000f, 0.1f };
            var halves = new short[values.Length];
            for (int i = 0; i < values.Length; i++) {
                halves[i] = (short)Mathf.FloatToHalf(values[i]);
            }
            using var input = new Tensor<short>(new TensorShape(values.Length), halves);
            using var worker = file.CreateWorker(BackendType.GPUCompute);
            worker.Schedule(input);

            using var doubled = (worker.PeekOutput(0) as Tensor<short>).ReadbackAndClone();
            using var added = (worker.PeekOutput(1) as Tensor<float>).ReadbackAndClone();
            short[] actual = doubled.DownloadToArray();
            float[] plusOne = added.DownloadToArray();
            for (int i = 0; i < values.Length; i++) {
                float expected = Mathf.HalfToFloat((ushort)halves[i]);
                Assert.That(Mathf.HalfToFloat((ushort)actual[i]), Is.EqualTo(2 * expected), $"output 0 at {i}");
                Assert.That(plusOne[i], Is.EqualTo(expected + 1), $"output 1 at {i}");
            }
            worker.Dispose();
            file.Dispose();
            File.Delete(path);
        }
    }
}
