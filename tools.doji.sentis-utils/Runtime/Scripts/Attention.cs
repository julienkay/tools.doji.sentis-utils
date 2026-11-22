using System;
using System.Collections.Generic;
using SentisFlatBuffer;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Google.FlatBuffers;
using Unity.Profiling;
using UnityEngine;
using static Unity.InferenceEngine.ComputeTensorData;
using Layers = Unity.InferenceEngine.Layers;

namespace Doji.AI {

    /// <summary>
    /// Attention that needs less memory: each attention of a model becomes one layer that never holds the scores of
    /// all queries. <see cref="Slice"/> computes them for a few queries at a time, like diffusers'
    /// <c>enable_attention_slicing</c>. <see cref="UseFlashAttention"/> computes them in one GPU kernel that keeps them
    /// out of memory, like flash attention. The result is the same, apart from rounding.
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
            var attentions = Fuse(model);
            foreach (var attention in attentions) {
                attention.maxScoreBytes = maxScoreBytes;
                attention.flash = false;
            }
            return attentions.Count;
        }

        /// <summary>
        /// Replaces each attention of <paramref name="model"/> as <see cref="Slice"/> does, with one layer that runs
        /// a flash attention kernel on <see cref="BackendType.GPUCompute"/>. It needs a head size of at most 128 and
        /// no broadcasting between the query, key and value; other attentions, and other backends, are sliced with
        /// <see cref="DefaultSliceBytes"/>.
        /// </summary>
        /// <returns>The number of replaced attentions.</returns>
        public static int UseFlashAttention(Model model) {
            var attentions = Fuse(model);
            foreach (var attention in attentions) {
                attention.flash = true;
            }
            return attentions.Count;
        }

        /// <summary>
        /// Replaces each attention with an <see cref="AttentionLayer"/>, or returns the ones it replaced before.
        /// </summary>
        private static List<AttentionLayer> Fuse(Model model) {
            var attentions = new List<AttentionLayer>();
            foreach (var layer in model.layers) {
                if (layer is AttentionLayer attention) {
                    attentions.Add(attention);
                }
            }
            if (attentions.Count > 0) {
                return attentions;
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
                var fused = new AttentionLayer(values.outputs[0], keys.inputs[0], keys.inputs[1], values.inputs[1], mask, scale, bias, DefaultSliceBytes);
                replaced[values] = fused;
                attentions.Add(fused);
                foreach (var part in parts) {
                    if (part != values) {
                        removed.Add(part);
                    }
                }

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
            return attentions;
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
        /// <summary>Whether to run the flash attention kernel where it can.</summary>
        public bool flash;

        private static readonly ComputeFunction FlashFunction = new ComputeFunction(
            Resources.Load<ComputeShader>("ComputeShaders/Compute.Shaders.FlashAttention"), "FlashAttention");
        private const int FlashQueries = 64, FlashThreads = 256, FlashMaxHeadSize = 128;
        private static readonly int QID = Shader.PropertyToID("Qptr"), KID = Shader.PropertyToID("Kptr"),
            VID = Shader.PropertyToID("Vptr"), MID = Shader.PropertyToID("Mptr"), OID = Shader.PropertyToID("Optr"),
            NqID = Shader.PropertyToID("Nq"), NkID = Shader.PropertyToID("Nk"), DID = Shader.PropertyToID("D"),
            DVID = Shader.PropertyToID("DV"), ScaleID = Shader.PropertyToID("scale"), BiasID = Shader.PropertyToID("bias"),
            HasMaskID = Shader.PropertyToID("hasMask"), MaskBatchStrideID = Shader.PropertyToID("maskBatchStride"),
            MaskRowStrideID = Shader.PropertyToID("maskRowStride");

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

            if (flash && ctx.backend is GPUComputeBackend gpu && TryFlash(gpu, query, key, value, mask, output)) {
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
        /// Dispatches the flash attention kernel, if it supports the shapes: query, key and value with the same
        /// batch and head dimensions, a head size of at most 128, and a mask that is the same for all of them or
        /// differs for each.
        /// </summary>
        private bool TryFlash(GPUComputeBackend gpu, Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> mask, Tensor<float> output) {
            TensorShape q = query.shape, k = key.shape, v = value.shape;
            int rank = q.rank;
            if (rank < 3 || k.rank != rank || v.rank != rank) {
                return false;
            }
            int batch = 1;
            for (int axis = 0; axis < rank - 2; axis++) {
                if (k[axis] != q[axis] || v[axis] != q[axis]) {
                    return false;
                }
                batch *= q[axis];
            }
            int queries = q[-2], size = q[-1], keys = k[-1], valueSize = v[-1];
            if (size > FlashMaxHeadSize || valueSize > FlashMaxHeadSize || k[-2] != size || v[-2] != keys || batch > 65535) {
                return false;
            }
            int maskBatchStride = 0, maskRowStride = 0;
            if (mask != null) {
                TensorShape m = mask.shape;
                if (m.rank < 1 || m[-1] != keys || m.rank > rank) {
                    return false;
                }
                maskRowStride = m.rank >= 2 && m[-2] == queries ? keys : m.rank >= 2 && m[-2] != 1 ? -1 : 0;
                int maskBatch = 1;
                for (int axis = 0; axis < m.rank - 2; axis++) {
                    maskBatch *= m[axis];
                }
                maskBatchStride = maskBatch == batch && m.rank == rank ? (m.rank >= 2 ? m[-2] : 1) * keys : maskBatch == 1 ? 0 : -1;
                if (maskRowStride < 0 || maskBatchStride < 0) {
                    return false;
                }
            }

            var fn = FlashFunction;
            var cb = gpu.GetCommandBuffer();
            cb.SetComputeIntParam(fn.shader, NqID, queries);
            cb.SetComputeIntParam(fn.shader, NkID, keys);
            cb.SetComputeIntParam(fn.shader, DID, size);
            cb.SetComputeIntParam(fn.shader, DVID, valueSize);
            cb.SetComputeFloatParam(fn.shader, ScaleID, scale);
            cb.SetComputeFloatParam(fn.shader, BiasID, bias);
            cb.SetComputeIntParam(fn.shader, HasMaskID, mask != null ? 1 : 0);
            cb.SetComputeIntParam(fn.shader, MaskBatchStrideID, maskBatchStride);
            cb.SetComputeIntParam(fn.shader, MaskRowStrideID, maskRowStride);
            cb.SetTensorAsBuffer(fn, QID, Pin(query));
            cb.SetTensorAsBuffer(fn, KID, Pin(key));
            cb.SetTensorAsBuffer(fn, VID, Pin(value));
            // The kernel reads the mask only when there is one, but every buffer must be set.
            cb.SetTensorAsBuffer(fn, MID, Pin(mask ?? query));
            cb.SetTensorAsBuffer(fn, OID, Pin(output));
            cb.Dispatch(fn, (queries + FlashQueries - 1) / FlashQueries * FlashThreads, batch, 1);
            return true;
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
