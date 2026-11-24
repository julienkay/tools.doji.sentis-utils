using System;
using System.Collections.Generic;
using SentisFlatBuffer;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Google.FlatBuffers;
using Unity.InferenceEngine.Layers;
using Unity.Profiling;
using UnityEngine;
using static Unity.InferenceEngine.ComputeTensorData;

namespace Doji.AI {

    /// <summary>
    /// Inputs and outputs of a model in float16 (<see cref="DataType.Short"/>), which take half the memory of float32.
    /// For example, the KV cache of a language model, which the model reads as inputs and writes as outputs, can stay
    /// in float16 between runs, as transformers and llama.cpp keep the cache of a float16 model in float16.
    /// </summary>
    public static class HalfPrecision {

        /// <summary>
        /// Makes the named float inputs and outputs of <paramref name="model"/> float16. The model casts each such
        /// input to float32 before the layers that read it, and casts each such output to float16 after the layer
        /// that writes it, so the float32 copy of a tensor exists only while its layers run. Create the workers of
        /// the model after this, with <see cref="ModelFile.CreateWorker"/>: the storage of a plain
        /// <see cref="Worker"/> cannot hold float16 tensors. The float16 outputs need
        /// <see cref="BackendType.GPUCompute"/>.
        /// </summary>
        /// <returns>The number of inputs and outputs it changed.</returns>
        public static int Apply(Model model, ICollection<string> inputs, ICollection<string> outputs) {
            int next = NextIndex(model);
            int changed = 0;
            // Input index to the index of its float32 cast.
            var renamed = new Dictionary<int, int>();
            for (int i = 0; i < model.inputs.Count; i++) {
                var input = model.inputs[i];
                if (!inputs.Contains(input.name) || input.dataType != DataType.Float) {
                    continue;
                }
                renamed[input.index] = next++;
                input.dataType = DataType.Short;
                model.inputs[i] = input;
                changed++;
            }
            // Float32 tensor index to the index of its float16 output.
            var halves = new Dictionary<int, int>();
            for (int i = 0; i < model.outputs.Count; i++) {
                var output = model.outputs[i];
                if (!outputs.Contains(output.name)) {
                    continue;
                }
                // An output that is an input it changed is read through the cast of that input.
                int source = renamed.TryGetValue(output.index, out int cast) ? cast : output.index;
                if (!halves.TryGetValue(source, out int half)) {
                    halves[source] = half = next++;
                }
                output.index = half;
                model.outputs[i] = output;
                changed++;
            }

            // Each input cast goes right before the first layer that reads it, and each output cast right after the
            // layer that writes its tensor, so float32 copies don't live longer than the layers that use them.
            var layers = new List<Layer>(model.layers.Count + changed);
            var placed = new HashSet<int>();
            void CastToHalf(int source) {
                if (halves.TryGetValue(source, out int half) && placed.Add(half)) {
                    layers.Add(new CastFloatToHalfLayer(half, source));
                }
            }
            foreach (var layer in model.layers) {
                for (int j = 0; j < layer.inputs.Length; j++) {
                    if (layer.inputs[j] < 0 || !renamed.TryGetValue(layer.inputs[j], out int cast)) {
                        continue;
                    }
                    if (placed.Add(cast)) {
                        layers.Add(new CastHalfToFloatLayer(cast, layer.inputs[j]));
                        CastToHalf(cast);
                    }
                    layer.inputs[j] = cast;
                }
                layers.Add(layer);
                foreach (int output in layer.outputs) {
                    CastToHalf(output);
                }
            }
            // Inputs that no layer reads but an output returns.
            foreach (var (input, cast) in renamed) {
                if (halves.ContainsKey(cast) && placed.Add(cast)) {
                    layers.Add(new CastHalfToFloatLayer(cast, input));
                    CastToHalf(cast);
                }
            }
            foreach (var source in halves.Keys) {
                CastToHalf(source);
            }
            model.layers = layers;
            return changed;
        }

        private static int NextIndex(Model model) {
            int max = -1;
            foreach (var input in model.inputs) {
                max = Math.Max(max, input.index);
            }
            foreach (var constant in model.constants) {
                max = Math.Max(max, constant.index);
            }
            foreach (var layer in model.layers) {
                foreach (int index in layer.outputs) {
                    max = Math.Max(max, index);
                }
                foreach (int index in layer.inputs) {
                    max = Math.Max(max, index);
                }
            }
            return max + 1;
        }
    }

    /// <summary>
    /// Casts a float16 (<see cref="DataType.Short"/>) tensor to float, with the cast of the backend.
    /// </summary>
    internal sealed class CastHalfToFloatLayer : Layer {

        private static readonly ProfilerMarker Marker = new ProfilerMarker("Doji.AI.CastHalfToFloat");

        public CastHalfToFloatLayer(int output, int input) : base(new[] { output }, new[] { input }) { }

        public override ProfilerMarker profilerMarker => Marker;
        public override string opName => "CastHalfToFloat";
        internal override string category => "Custom";
        internal override string[] GetInputNames() => new[] { "input" };
        internal override string[] GetOutputNames() => new[] { "output" };

        internal override void SerializeFields(FlatBufferBuilder builder, List<Offset<EValue>> values) {
            throw new NotSupportedException("A model with float16 inputs or outputs cannot be saved. Save it before HalfPrecision.Apply.");
        }

        internal override void InferPartial(PartialInferenceContext ctx) {
            var input = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], PartialTensor.Create(DataType.Float, input.shape));
        }

        internal override void Execute(ExecutionContext ctx) {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<short>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims()) {
                return;
            }
            ctx.backend.Cast(X, O);
        }
    }

    /// <summary>
    /// Casts a float tensor to float16, packed two values in each 32-bit value as <see cref="DataType.Short"/>.
    /// </summary>
    internal sealed class CastFloatToHalfLayer : Layer {

        private static ComputeFunction _function;
        private static readonly int XID = Shader.PropertyToID("Xptr"), OID = Shader.PropertyToID("Optr"),
            LengthID = Shader.PropertyToID("Length"), CountID = Shader.PropertyToID("Count"),
            GroupsXID = Shader.PropertyToID("GroupsX");
        private const int Threads = 256, MaxGroups = 65535;
        private static readonly ProfilerMarker Marker = new ProfilerMarker("Doji.AI.CastFloatToHalf");

        public CastFloatToHalfLayer(int output, int input) : base(new[] { output }, new[] { input }) { }

        public override ProfilerMarker profilerMarker => Marker;
        public override string opName => "CastFloatToHalf";
        internal override string category => "Custom";
        internal override string[] GetInputNames() => new[] { "input" };
        internal override string[] GetOutputNames() => new[] { "output" };

        internal override void SerializeFields(FlatBufferBuilder builder, List<Offset<EValue>> values) {
            throw new NotSupportedException("A model with float16 inputs or outputs cannot be saved. Save it before HalfPrecision.Apply.");
        }

        internal override void InferPartial(PartialInferenceContext ctx) {
            var input = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], PartialTensor.Create(DataType.Short, input.shape));
        }

        internal override void Execute(ExecutionContext ctx) {
            if (ctx.storage is not FittedModelStorage) {
                throw new NotSupportedException("Float16 outputs need a worker from ModelFile.CreateWorker.");
            }
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Short, ctx.backend.backendType);
            if (O.shape.HasZeroDims()) {
                return;
            }
            if (ctx.backend is not GPUComputeBackend gpu) {
                throw new NotSupportedException("Float16 outputs need BackendType.GPUCompute.");
            }
            _function ??= new ComputeFunction(Resources.Load<ComputeShader>("ComputeShaders/Compute.Shaders.CastFloatToHalf"), "CastFloatToHalf");
            var fn = _function;
            var cb = gpu.GetCommandBuffer();
            int groups = (O.count + Threads - 1) / Threads;
            int groupsX = Math.Min(groups, MaxGroups);
            cb.SetTensorAsBuffer(fn, XID, Pin(X));
            cb.SetTensorAsBuffer(fn, OID, Pin(O));
            cb.SetComputeIntParam(fn.shader, LengthID, X.shape.length);
            cb.SetComputeIntParam(fn.shader, CountID, O.count);
            cb.SetComputeIntParam(fn.shader, GroupsXID, groupsX);
            cb.Dispatch(fn, groupsX * Threads, (groups + groupsX - 1) / groupsX, 1);
        }
    }
}
