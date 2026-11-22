using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.Editor.Tests {

    public class QuantizedGatherTest {

        private static readonly BackendType[] Backends = { BackendType.CPU, BackendType.GPUCompute };

        private string _path;

        /// <summary>
        /// A .sentis file with a uint8 embedding table, gathered with token ids.
        /// </summary>
        [OneTimeSetUp]
        public void SaveEmbedding() {
            const int vocabulary = 10, size = 8;
            var table = new float[vocabulary * size];
            for (int i = 0; i < table.Length; i++) {
                table[i] = (i * 37 % 101) / 50f - 1;
            }
            var graph = new FunctionalGraph();
            var ids = graph.AddInput<int>(new DynamicTensorShape(-1, -1));
            using var weights = new Tensor<float>(new TensorShape(vocabulary, size), table);
            Model model = graph.Compile(Functional.Constant(weights).IndexSelect(0, ids));
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref model);
            _path = Path.Combine(Path.GetTempPath(), "QuantizedGatherTest.sentis");
            ModelWriter.Save(_path, model);
        }

        [OneTimeTearDown]
        public void DeleteEmbedding() {
            File.Delete(_path);
        }

        [Test]
        public void Fuse_ReplacesTheDequantizeAndTheGather() {
            Model model = ModelLoader.Load(_path);
            Assert.That(QuantizedGather.Fuse(model), Is.EqualTo(1));
            Assert.That(model.layers.ConvertAll(l => l.opName), Is.EqualTo(new[] { "GatherDequantizeUint8" }));
        }

        /// <summary>
        /// <see cref="ModelFile"/> fuses the Gather, which must give the embeddings of the model as it is saved.
        /// </summary>
        [TestCaseSource(nameof(Backends))]
        public void ModelFile_GivesTheSameEmbeddings(BackendType backend) {
            using var ids = new Tensor<int>(new TensorShape(2, 3), new[] { 0, 9, 4, -1, 3, 3 });
            using var expectedWorker = new Worker(ModelLoader.Load(_path), backend);
            using var file = ModelFile.Open(_path);
            using var actualWorker = file.CreateWorker(backend);
            expectedWorker.Schedule(ids);
            actualWorker.Schedule(ids);
            using var expected = (expectedWorker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            using var actual = (actualWorker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            Assert.That(actual.shape, Is.EqualTo(new TensorShape(2, 3, 8)));
            Assert.That(actual.DownloadToArray(), Is.EqualTo(expected.DownloadToArray()));
        }
    }
}
