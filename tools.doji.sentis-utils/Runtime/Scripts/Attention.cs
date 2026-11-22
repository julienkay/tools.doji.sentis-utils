using System;
using System.Collections.Generic;
using SentisFlatBuffer;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Google.FlatBuffers;
using Unity.Profiling;
using Layers = Unity.InferenceEngine.Layers;

namespace Doji.AI {

    /// <summary>
    /// Attention slicing, like diffusers' <c>enable_attention_slicing</c>: each attention of a model computes its
    /// scores for a few queries at a time, so it never holds the scores of all queries. The result is the same.
    /// </summary>
    public static class Attention {

        /// <summary>
        /// The bytes of attention scores that <see cref="Slice"/> computes at a time by default.
        /// </summary>
        public const long DefaultSliceBytes = 256L << 20;

        /// <summary>
        /// Replaces each attention of <paramref name="model"/>, MatMul, then optionally ScalarMad and a mask Add,
        /// Softmax and MatMul, with one layer that computes at most <paramref name="maxScoreBytes"/> bytes of scores
        /// at a time. On a model that is already sliced, it changes the bytes. Create the workers of the model after
        /// this.
        /// </summary>
        /// <remarks>
        /// This method edits <paramref name="model"/> in place. Do not save the sliced model with
        /// <see cref="ModelWriter"/>: it throws a <see cref="NotSupportedException"/>.
        /// </remarks>
        /// <returns>The number of sliced attentions.</returns>
        public static int Slice(Model model, long maxScoreBytes = DefaultSliceBytes) {
            if (maxScoreBytes <= 0) {
                throw new ArgumentOutOfRangeException(nameof(maxScoreBytes));
            }
            int sliced = 0;
            foreach (var layer in model.layers) {
                if (layer is AttentionLayer attention) {
                    attention.maxScoreBytes = maxScoreBytes;
                    sliced++;
                }
            }
            if (sliced > 0) {
                return sliced;
            }

            var producer = new Dictionary<int, Layer>();
            var consumers = new Dictionary<int, int>();
            foreach (var layer in model.layers) {
                foreach (int output in layer.outputs) {
                    producer[output] = layer;
                }
                foreach (int input in layer.inputs) {
                    consumers[input] = consumers.GetValueOrDefault(input) + 1;
                }
            }
            foreach (var output in model.outputs) {
                consumers[output.index] = consumers.GetValueOrDefault(output.index) + 1;
            }
            // Only the result of the last layer of an attention may be read by other layers.
            bool Internal(int tensor) => consumers.GetValueOrDefault(tensor) == 1;
            Layer Producer(int tensor) => producer.GetValueOrDefault(tensor);

            var replaced = new Dictionary<Layer, Layer>();
            var removed = new HashSet<Layer>();
            foreach (var layer in model.layers) {
                if (layer is not Layers.Softmax softmax || softmax.axis != -1 || !Internal(softmax.outputs[0])) {
                    continue;
                }
                Layers.MatMul values = null;
                foreach (var other in model.layers) {
                    if (other is Layers.MatMul m && m.inputs[0] == softmax.outputs[0]) {
                        values = m;
                        break;
                    }
                }
                if (values == null || !Internal(softmax.inputs[0])) {
                    continue;
                }
                var parts = new List<Layer> { softmax, values };
                int mask = -1;
                int scores = softmax.inputs[0];
                if (Producer(scores) is Layers.Add add && Internal(scores)) {
                    // The scores are the input that a MatMul computes, maybe through a ScalarMad.
                    int side = IsScores(Producer(add.inputs[0])) ? 0 : IsScores(Producer(add.inputs[1])) ? 1 : -1;
                    if (side < 0) {
                        continue;
                    }
                    parts.Add(add);
                    mask = add.inputs[1 - side];
                    scores = add.inputs[side];
                }
                float scale = 1, bias = 0;
                if (Producer(scores) is Layers.ScalarMad mad && mad.dataType == DataType.Float && Internal(scores)) {
                    parts.Add(mad);
                    scale = mad.sFloat;
                    bias = mad.bFloat;
                    scores = mad.inputs[0];
                }
                if (Producer(scores) is not Layers.MatMul keys || !Internal(scores)) {
                    continue;
                }
                parts.Add(keys);
                var fused = new AttentionLayer(values.outputs[0], keys.inputs[0], keys.inputs[1], values.inputs[1], mask, scale, bias, maxScoreBytes);
                replaced[values] = fused;
                foreach (var part in parts) {
                    if (part != values) {
                        removed.Add(part);
                    }
                }
                sliced++;

                bool IsScores(Layer l) => l is Layers.MatMul || l is Layers.ScalarMad s && Producer(s.inputs[0]) is Layers.MatMul;
            }

            var layers = new List<Layer>(model.layers.Count);
            foreach (var layer in model.layers) {
                if (replaced.TryGetValue(layer, out var fused)) {
                    layers.Add(fused);
                } else if (!removed.Contains(layer)) {
                    layers.Add(layer);
                }
            }
            model.layers = layers;
            return sliced;
        }
    }

    /// <summary>
    /// softmax(scale * (q × k) + bias + mask) × v, for a few queries at a time. The key input is already
    /// transposed, as the first MatMul of an attention reads it.
    /// </summary>
    internal sealed class AttentionLayer : Layer {

        public float scale;
        public float bias;
        public long maxScoreBytes;

        private static readonly ProfilerMarker Marker = new ProfilerMarker("Doji.AI.Attention");

        public AttentionLayer(int output, int query, int key, int value, int mask, float scale, float bias, long maxScoreBytes)
            : base(new[] { output }, new[] { query, key, value, mask }) {
            this.scale = scale;
            this.bias = bias;
            this.maxScoreBytes = maxScoreBytes;
        }

        public override ProfilerMarker profilerMarker => Marker;
        public override string opName => "Attention";
        internal override string category => "Custom";
        internal override string[] GetInputNames() => new[] { "query", "key", "value", "mask" };
        internal override string[] GetOutputNames() => new[] { "output" };

        internal override void SerializeFields(FlatBufferBuilder builder, List<Offset<EValue>> values) {
            throw new NotSupportedException("A model with sliced attention cannot be saved. Save it before Attention.Slice.");
        }

        internal override void InferPartial(PartialInferenceContext ctx) {
            var query = ctx.GetPartialTensor(inputs[0]);
            var key = ctx.GetPartialTensor(inputs[1]);
            var value = ctx.GetPartialTensor(inputs[2]);
            ctx.AddPartialTensor(outputs[0], PartialTensor.Create(DataType.Float, query.shape.MatMul(key.shape).MatMul(value.shape)));
        }

        internal override void Execute(ExecutionContext ctx) {
            var query = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var key = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var value = ctx.storage.GetTensor(inputs[2]) as Tensor<float>;
            var mask = inputs[3] == -1 ? null : ctx.storage.GetTensor(inputs[3]) as Tensor<float>;
            TensorShape scoresShape = query.shape.MatMul(key.shape);
            var output = ctx.storage.AllocateTensorAndStore(outputs[0], scoresShape.MatMul(value.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (output.shape.HasZeroDims()) {
                return;
            }

            int rows = scoresShape[-2];
            long rowBytes = (long)scoresShape.length / Math.Max(rows, 1) * sizeof(float);
            int chunk = (int)Math.Clamp(maxScoreBytes / Math.Max(rowBytes, 1), 1, rows);
            if (chunk == rows) {
                Compute(ctx, query, key, value, mask, output);
                return;
            }
            for (int start = 0; start < rows; start += chunk) {
                int count = Math.Min(chunk, rows - start);
                var queryRows = Rows(ctx, query, start, count);
                // A mask with one row applies to all queries.
                var maskRows = mask != null && mask.shape.rank >= 2 && mask.shape[-2] == rows ? Rows(ctx, mask, start, count) : mask;
                var outputRows = Allocate(ctx, Resize(output.shape, count));
                Compute(ctx, queryRows, key, value, maskRows, outputRows);
                ctx.backend.SliceSet(outputRows, output, output.shape.rank - 2, start, 1);
                ctx.storage.Dispose(queryRows);
                if (maskRows != mask) {
                    ctx.storage.Dispose(maskRows);
                }
                ctx.storage.Dispose(outputRows);
            }
        }

        /// <summary>
        /// The attention of all rows of <paramref name="query"/>, with the same operations as the layers it replaces.
        /// </summary>
        private void Compute(ExecutionContext ctx, Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> mask, Tensor<float> output) {
            var scores = Allocate(ctx, query.shape.MatMul(key.shape));
            ctx.backend.MatMul(query, key, scores);
            if (scale != 1 || bias != 0) {
                var scaled = Allocate(ctx, scores.shape);
                ctx.backend.ScalarMad(scores, scaled, scale, bias);
                ctx.storage.Dispose(scores);
                scores = scaled;
            }
            if (mask != null) {
                var masked = Allocate(ctx, scores.shape.Broadcast(mask.shape));
                ctx.backend.Add(scores, mask, masked);
                ctx.storage.Dispose(scores);
                scores = masked;
            }
            var probabilities = Allocate(ctx, scores.shape);
            ctx.backend.Softmax(scores, probabilities, -1);
            ctx.storage.Dispose(scores);
            ctx.backend.MatMul(probabilities, value, output);
            ctx.storage.Dispose(probabilities);
        }

        /// <summary>
        /// Rows <paramref name="start"/> to <paramref name="start"/> + <paramref name="count"/> of the second to last
        /// axis of <paramref name="tensor"/>.
        /// </summary>
        private static Tensor<float> Rows(ExecutionContext ctx, Tensor<float> tensor, int start, int count) {
            var rows = Allocate(ctx, Resize(tensor.shape, count));
            ctx.backend.Slice(tensor, rows, new[] { start }, new[] { tensor.shape.rank - 2 }, new[] { 1 });
            return rows;
        }

        private static TensorShape Resize(TensorShape shape, int rows) {
            shape[-2] = rows;
            return shape;
        }

        private static Tensor<float> Allocate(ExecutionContext ctx, TensorShape shape) {
            return ctx.storage.AllocateTensor(shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
        }
    }
}
