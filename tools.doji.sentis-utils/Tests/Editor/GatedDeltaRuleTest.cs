using System;
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using Random = System.Random;

namespace Doji.AI.Editor.Tests {

    public class GatedDeltaRuleTest {

        private static readonly BackendType[] Backends = { BackendType.CPU, BackendType.GPUCompute };

        private const int Batch = 2, Heads = 3, KeySize = 20, ValueSize = 40;
        private const float Scale = 0.3f;

        private static Model CreateModel() {
            var graph = new FunctionalGraph();
            var q = graph.AddInput<float>(new DynamicTensorShape(Batch, -1, Heads, KeySize));
            var k = graph.AddInput<float>(new DynamicTensorShape(Batch, -1, Heads, KeySize));
            var v = graph.AddInput<float>(new DynamicTensorShape(Batch, -1, Heads, ValueSize));
            var g = graph.AddInput<float>(new DynamicTensorShape(Batch, -1, Heads));
            var beta = graph.AddInput<float>(new DynamicTensorShape(Batch, -1, Heads));
            var state = graph.AddInput<float>(new TensorShape(Batch, Heads, KeySize, ValueSize));
            var (output, finalState) = LinearAttention.GatedDeltaRule(q, k, v, g, beta, state, Scale);
            return graph.Compile(output, finalState);
        }

        private static float[] Values(int count, int seed, float min, float max) {
            var random = new Random(seed);
            var values = new float[count];
            for (int i = 0; i < count; i++) {
                values[i] = min + (float)random.NextDouble() * (max - min);
            }
            return values;
        }

        private sealed class Case {
            public int tokens;
            public float[] q, k, v, g, beta, state;

            public Case(int tokens) {
                this.tokens = tokens;
                int rows = Batch * tokens * Heads;
                q = Values(rows * KeySize, 1, -1, 1);
                k = Values(rows * KeySize, 2, -0.3f, 0.3f);
                v = Values(rows * ValueSize, 3, -1, 1);
                g = Values(rows, 4, -1, 0);
                beta = Values(rows, 5, 0, 1);
                state = Values(Batch * Heads * KeySize * ValueSize, 6, -1, 1);
            }

            /// <summary>
            /// The recurrence of transformers' torch_recurrent_gated_delta_rule, in double precision.
            /// </summary>
            public (float[] output, float[] state) Expected() {
                var output = new float[Batch * tokens * Heads * ValueSize];
                var s = Array.ConvertAll(state, x => (double)x);
                for (int b = 0; b < Batch; b++) {
                    for (int h = 0; h < Heads; h++) {
                        int sBase = (b * Heads + h) * KeySize * ValueSize;
                        for (int t = 0; t < tokens; t++) {
                            int row = (b * tokens + t) * Heads + h;
                            double decay = Math.Exp(g[row]);
                            for (int j = 0; j < ValueSize; j++) {
                                double memory = 0;
                                for (int i = 0; i < KeySize; i++) {
                                    s[sBase + i * ValueSize + j] *= decay;
                                    memory += s[sBase + i * ValueSize + j] * k[row * KeySize + i];
                                }
                                double delta = (v[row * ValueSize + j] - memory) * beta[row];
                                double o = 0;
                                for (int i = 0; i < KeySize; i++) {
                                    s[sBase + i * ValueSize + j] += k[row * KeySize + i] * delta;
                                    o += s[sBase + i * ValueSize + j] * q[row * KeySize + i] * Scale;
                                }
                                output[row * ValueSize + j] = (float)o;
                            }
                        }
                    }
                }
                return (output, Array.ConvertAll(s, x => (float)x));
            }

            public (float[] output, float[] state) Run(Model model, BackendType backend) {
                using var worker = new Worker(model, backend);
                using var tq = new Tensor<float>(new TensorShape(Batch, tokens, Heads, KeySize), q);
                using var tk = new Tensor<float>(new TensorShape(Batch, tokens, Heads, KeySize), k);
                using var tv = new Tensor<float>(new TensorShape(Batch, tokens, Heads, ValueSize), v);
                using var tg = new Tensor<float>(new TensorShape(Batch, tokens, Heads), g);
                using var tb = new Tensor<float>(new TensorShape(Batch, tokens, Heads), beta);
                using var ts = new Tensor<float>(new TensorShape(Batch, Heads, KeySize, ValueSize), state);
                worker.Schedule(tq, tk, tv, tg, tb, ts);
                using var output = (worker.PeekOutput(0) as Tensor<float>).ReadbackAndClone();
                using var finalState = (worker.PeekOutput(1) as Tensor<float>).ReadbackAndClone();
                return (output.DownloadToArray(), finalState.DownloadToArray());
            }
        }

        [Test]
        public void Run_GivesTheRecurrence([ValueSource(nameof(Backends))] BackendType backend, [Values(1, 7)] int tokens) {
            var test = new Case(tokens);
            var (expectedOutput, expectedState) = test.Expected();
            var (output, state) = test.Run(CreateModel(), backend);
            Assert.That(output, Is.EqualTo(expectedOutput).Within(1e-4f));
            Assert.That(state, Is.EqualTo(expectedState).Within(1e-4f));
        }

        /// <summary>
        /// Running the tokens one at a time, with the state of each run in the next, gives the same as one run.
        /// </summary>
        [TestCaseSource(nameof(Backends))]
        public void State_ContinuesTheSequence(BackendType backend) {
            var model = CreateModel();
            var all = new Case(4);
            var (expectedOutput, expectedState) = all.Run(model, backend);
            float[] state = all.state;
            var output = new float[expectedOutput.Length];
            for (int t = 0; t < 4; t++) {
                var step = new Case(1) { state = state };
                step.q = Row(all.q, t, KeySize);
                step.k = Row(all.k, t, KeySize);
                step.v = Row(all.v, t, ValueSize);
                step.g = Row(all.g, t, 1);
                step.beta = Row(all.beta, t, 1);
                float[] stepOutput;
                (stepOutput, state) = step.Run(model, backend);
                for (int b = 0; b < Batch; b++) {
                    Array.Copy(stepOutput, b * Heads * ValueSize, output, ((b * 4 + t) * Heads) * ValueSize, Heads * ValueSize);
                }
            }
            Assert.That(output, Is.EqualTo(expectedOutput).Within(1e-5f));
            Assert.That(state, Is.EqualTo(expectedState).Within(1e-5f));
        }

        // The values of token t of each batch of a (Batch, 4, Heads, size) tensor.
        private static float[] Row(float[] values, int t, int size) {
            var row = new float[Batch * Heads * size];
            for (int b = 0; b < Batch; b++) {
                Array.Copy(values, ((b * 4 + t) * Heads) * size, row, b * Heads * size, Heads * size);
            }
            return row;
        }

        [Test]
        public void Save_KeepsTheLayer() {
            string path = Path.Combine(Path.GetTempPath(), "GatedDeltaRuleTest.sentis");
            try {
                ModelWriter.Save(path, CreateModel());
                Model loaded = ModelLoader.Load(path);
                Assert.That(loaded.layers.ConvertAll(l => l.opName), Does.Contain("GatedDeltaRule"));
                var test = new Case(3);
                var (expectedOutput, _) = test.Expected();
                var (output, _) = test.Run(loaded, BackendType.CPU);
                Assert.That(output, Is.EqualTo(expectedOutput).Within(1e-4f));
            } finally {
                File.Delete(path);
            }
        }
    }
}
