using System.Linq;
using System.Runtime.Serialization;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Editor.Tests {

    public class ReadbacksTest {

        // The layer types are internal to Inference Engine. These ones have no fields besides their inputs and outputs.
        private static Layer NewLayer(string name, int output, params int[] inputs) {
            var type = typeof(Model).Assembly.GetType("Unity.InferenceEngine.Layers." + name);
            var layer = (Layer)FormatterServices.GetUninitializedObject(type);
            layer.outputs = new[] { output };
            layer.inputs = inputs;
            return layer;
        }

        /// <summary>
        /// limit = s + s runs on the CPU, because Range reads it there. A GPU layer reads limit before Range does.
        /// </summary>
        private static Model CreateModel() {
            var model = new Model();
            model.inputs.Add(new Model.Input { name = "s", index = 0, dataType = DataType.Float, shape = new DynamicTensorShape(new TensorShape()) });
            model.constants.Add(new Constant(1, new TensorShape(), new[] { 0f }));
            model.constants.Add(new Constant(2, new TensorShape(), new[] { 1f }));
            model.layers.Add(NewLayer("Add", 3, 0, 0));
            model.layers.Add(NewLayer("Mul", 4, 3, 3));
            model.layers.Add(NewLayer("Range", 5, 1, 3, 2));
            model.AddOutput("square", 4);
            model.AddOutput("range", 5);
            return model;
        }

        [Test]
        public void Avoid_CopiesTheCpuTensor() {
            var model = CreateModel();
            Assert.That(Readbacks.Find(model), Is.EqualTo(new[] { 3 }));
            Assert.That(Readbacks.Avoid(model), Is.EqualTo(1));
            Assert.That(Readbacks.Find(model), Is.Empty);
            Assert.That(model.layers.ConvertAll(l => l.opName), Is.EqualTo(new[] { "Add", "Add", "Mul", "Range" }));
        }

        [TestCase(BackendType.GPUCompute)]
        [TestCase(BackendType.CPU)]
        public void Outputs_StayTheSame(BackendType backend) {
            var model = CreateModel();
            Readbacks.Avoid(model);
            using var input = new Tensor<float>(new TensorShape(), new[] { 2.5f });
            using var worker = new Worker(model, backend);
            worker.Schedule(input);
            using var square = (worker.PeekOutput("square") as Tensor<float>).ReadbackAndClone();
            using var range = (worker.PeekOutput("range") as Tensor<float>).ReadbackAndClone();
            Assert.That(square.DownloadToArray(), Is.EqualTo(new[] { 25f }));
            Assert.That(range.DownloadToArray(), Is.EqualTo(new[] { 0f, 1f, 2f, 3f, 4f }));
        }
    }
}
