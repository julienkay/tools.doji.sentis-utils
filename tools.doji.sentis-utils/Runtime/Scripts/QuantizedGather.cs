using System;
using System.Collections.Generic;
using SentisFlatBuffer;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Google.FlatBuffers;
using Unity.Profiling;
using Layers = Unity.InferenceEngine.Layers;

namespace Doji.AI {

    /// <summary>
    /// Gathers the rows of a uint8 weight before it dequantizes them. A quantized embedding table is a
    /// DequantizeUint8 layer followed by a Gather, which turns the whole table into float32 to read a few rows of it:
    /// for the embeddings of Qwen3 4B, that is 1.6 GB. The result is the same.
    /// </summary>
    public static class QuantizedGather {

        /// <summary>
        /// Replaces each DequantizeUint8 layer whose only use is a Gather on axis 0 with one layer that dequantizes
        /// only the gathered rows. Each row of the weight must have a multiple of 4 values, as Inference Engine
        /// packs 4 uint8 values in each int.
        /// </summary>
        /// <returns>The number of replaced layers.</returns>
        public static int Fuse(Model model) {
            var uses = new Dictionary<int, int>();
            foreach (var layer in model.layers) {
                foreach (int input in layer.inputs) {
                    uses[input] = uses.TryGetValue(input, out int n) ? n + 1 : 1;
                }
            }
            foreach (var output in model.outputs) {
                uses[output.index] = uses.TryGetValue(output.index, out int n) ? n + 1 : 1;
            }
            var shapes = new Dictionary<int, TensorShape>();
            foreach (var constant in model.constants) {
                shapes[constant.index] = constant.shape;
            }

            var dequantized = new Dictionary<int, Layers.DequantizeUint8>();
            var layers = new List<Layer>(model.layers.Count);
            int fused = 0;
            foreach (var layer in model.layers) {
                if (layer is Layers.DequantizeUint8 dequantize && uses.GetValueOrDefault(dequantize.outputs[0]) == 1 &&
                    shapes.TryGetValue(dequantize.inputs[0], out var shape) && shape.rank >= 2 && shape.Length(1) % 4 == 0) {
                    // Wait for its use: a Gather replaces both, any other layer keeps it.
                    dequantized[dequantize.outputs[0]] = dequantize;
                    continue;
                }
                if (layer is Layers.Gather gather && gather.axis == 0 && dequantized.Remove(gather.inputs[0], out var source)) {
                    layers.Add(new GatherDequantizeUint8(gather.outputs[0], source.inputs[0], gather.inputs[1], source.scale, source.zeroPoint));
                    fused++;
                    continue;
                }
                foreach (int input in layer.inputs) {
                    if (dequantized.Remove(input, out var kept)) {
                        layers.Add(kept);
                    }
                }
                layers.Add(layer);
            }
            // The ones that are an output of the model.
            layers.AddRange(dequantized.Values);
            if (fused > 0) {
                model.layers = layers;
            }
            return fused;
        }
    }

    /// <summary>
    /// Gather on axis 0 of a uint8 weight, then y = (x - zeroPoint) * scale of the gathered rows.
    /// </summary>
    internal sealed class GatherDequantizeUint8 : Layer {

        public float scale;
        public byte zeroPoint;

        private static readonly ProfilerMarker Marker = new ProfilerMarker("Doji.AI.GatherDequantizeUint8");

        public GatherDequantizeUint8(int output, int weight, int indices, float scale, byte zeroPoint)
            : base(new[] { output }, new[] { weight, indices }) {
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }

        public override ProfilerMarker profilerMarker => Marker;
        public override string opName => "GatherDequantizeUint8";
        internal override string category => "Custom";
        internal override string[] GetInputNames() => new[] { "weight", "indices" };
        internal override string[] GetOutputNames() => new[] { "output" };

        internal override void SerializeFields(FlatBufferBuilder builder, List<Offset<EValue>> values) {
            throw new NotSupportedException("A model with a fused quantized Gather cannot be saved. Save it before QuantizedGather.Fuse.");
        }

        internal override void InferPartial(PartialInferenceContext ctx) {
            var weight = ctx.GetPartialTensor(inputs[0]);
            var indices = ctx.GetPartialTensor(inputs[1]);
            var gathered = Layers.Gather.InferPartial(weight, indices, 0);
            ctx.AddPartialTensor(outputs[0], PartialTensor.Create(DataType.Float, gathered.shape));
        }

        internal override void Execute(ExecutionContext ctx) {
            var weight = ctx.storage.GetTensor(inputs[0]) as Tensor<byte>;
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            TensorShape shape = ShapeInference.Gather(weight.shape, indices.shape, 0);
            var output = ctx.storage.AllocateTensorAndStore(outputs[0], shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (output.shape.HasZeroDims()) {
                return;
            }
            // Each row of 4 * n uint8 values is n ints, so a Gather of int rows copies the packed rows. The views
            // share the data of their tensors and are not disposed.
            int rowInts = weight.shape.Length(1) / 4;
            var words = new Tensor<int>(new TensorShape(weight.shape[0], rowInts), weight.dataOnBackend);
            var rows = ctx.storage.AllocateTensor(new TensorShape(indices.shape.length, rowInts), DataType.Int, ctx.backend.backendType);
            ctx.backend.Gather(words, indices, rows, 0);
            ctx.backend.DequantizeLinear(new Tensor<byte>(shape, rows.dataOnBackend), output, scale, zeroPoint);
            ctx.storage.Dispose(rows);
        }
    }
}
