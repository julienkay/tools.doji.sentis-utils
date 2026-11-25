using System;
using Unity.InferenceEngine;
using UnityEngine;
using static Unity.InferenceEngine.ComputeTensorData;

// The layer is in the namespace of the Inference Engine layers: the source generator of Inference Engine then adds it
// to the model loader, the model writer and the graph, so that a .sentis file can hold it.
namespace Unity.InferenceEngine.Layers {

    /// <summary>
    /// The gated delta rule of Gated DeltaNet (Qwen3.5, Qwen3-Next), one token after the other. For each batch and
    /// head, with the state S (key size x value size) and each token t:
    /// <c>S = S * exp(g[t])</c>, <c>delta = (v[t] - Sᵀ k[t]) * beta[t]</c>, <c>S = S + k[t] deltaᵀ</c> and
    /// <c>o[t] = Sᵀ (q[t] * scale)</c>.
    /// </summary>
    /// <remarks>
    /// The inputs are as in transformers' <c>torch_recurrent_gated_delta_rule</c>: query and key (batch, tokens,
    /// heads, key size), value (batch, tokens, heads, value size), g and beta (batch, tokens, heads), and the state
    /// (batch, heads, key size, value size). The query and the key are not normalized here. The outputs are the
    /// output, the shape of the value, and the state after the last token.
    /// </remarks>
    [Operator(category = "Custom")]
    [Inputs(names = new[] { "query", "key", "value", "g", "beta", "initialState" })]
    [Outputs(names = new[] { "output", "finalState" })]
    partial class GatedDeltaRule : Layer {

        public float scale;

        /// <summary>The largest key size of the GPU kernel.</summary>
        internal const int MaxKeySize = 128;
        private const int Threads = 32;

        private static ComputeFunction s_Function;
        private static readonly int QID = Shader.PropertyToID("Qptr"), KID = Shader.PropertyToID("Kptr"),
            VID = Shader.PropertyToID("Vptr"), GID = Shader.PropertyToID("Gptr"), BID = Shader.PropertyToID("Bptr"),
            SID = Shader.PropertyToID("Sptr"), OID = Shader.PropertyToID("Optr"), SOID = Shader.PropertyToID("SOptr"),
            TID = Shader.PropertyToID("T"), HID = Shader.PropertyToID("H"), DkID = Shader.PropertyToID("Dk"),
            DvID = Shader.PropertyToID("Dv"), ScaleID = Shader.PropertyToID("scale");

        internal override int OutputCount => 2;

        internal static PartialTensor[] InferPartial(PartialTensor query, PartialTensor key, PartialTensor value, PartialTensor g,
            PartialTensor beta, PartialTensor initialState, float scale) {
            var output = new DynamicTensorShape(value.shape);
            var state = new DynamicTensorShape(initialState.shape);
            return new[] { PartialTensor.Create(DataType.Float, output), PartialTensor.Create(DataType.Float, state) };
        }

        internal override void Execute(ExecutionContext ctx) {
            var query = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var key = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var value = ctx.storage.GetTensor(inputs[2]) as Tensor<float>;
            var g = ctx.storage.GetTensor(inputs[3]) as Tensor<float>;
            var beta = ctx.storage.GetTensor(inputs[4]) as Tensor<float>;
            var state = ctx.storage.GetTensor(inputs[5]) as Tensor<float>;
            TensorShape v = value.shape, s = state.shape;
            if (v.rank != 4 || s.rank != 4 || query.shape.rank != 4) {
                throw new ArgumentException($"GatedDeltaRule needs inputs of rank 4, not query {query.shape}, value {v} and state {s}.");
            }
            int batch = v[0], tokens = v[1], heads = v[2], keySize = s[2], valueSize = v[3];
            if (key.shape != query.shape || query.shape[3] != keySize || s[0] != batch || s[1] != heads || s[3] != valueSize
                || g.shape.length != batch * tokens * heads || beta.shape.length != batch * tokens * heads) {
                throw new ArgumentException($"GatedDeltaRule: the shapes do not match: query {query.shape}, key {key.shape}, " +
                    $"value {v}, g {g.shape}, beta {beta.shape}, state {s}.");
            }
            var output = ctx.storage.AllocateTensorAndStore(outputs[0], v, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            var finalState = ctx.storage.AllocateTensorAndStore(outputs[1], s, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (s.HasZeroDims()) {
                return;
            }
            if (ctx.backend is GPUComputeBackend gpu) {
                if (keySize > MaxKeySize) {
                    throw new NotSupportedException($"GatedDeltaRule supports a key size of at most {MaxKeySize} on the GPU, not {keySize}.");
                }
                s_Function ??= new ComputeFunction(Resources.Load<ComputeShader>("ComputeShaders/Compute.Shaders.GatedDeltaRule"), "GatedDeltaRule");
                var fn = s_Function;
                var cb = gpu.GetCommandBuffer();
                cb.SetComputeIntParam(fn.shader, TID, tokens);
                cb.SetComputeIntParam(fn.shader, HID, heads);
                cb.SetComputeIntParam(fn.shader, DkID, keySize);
                cb.SetComputeIntParam(fn.shader, DvID, valueSize);
                cb.SetComputeFloatParam(fn.shader, ScaleID, scale);
                cb.SetTensorAsBuffer(fn, QID, Pin(query));
                cb.SetTensorAsBuffer(fn, KID, Pin(key));
                cb.SetTensorAsBuffer(fn, VID, Pin(value));
                cb.SetTensorAsBuffer(fn, GID, Pin(g));
                cb.SetTensorAsBuffer(fn, BID, Pin(beta));
                cb.SetTensorAsBuffer(fn, SID, Pin(state));
                cb.SetTensorAsBuffer(fn, OID, Pin(output));
                cb.SetTensorAsBuffer(fn, SOID, Pin(finalState));
                cb.Dispatch(fn, (valueSize + Threads - 1) / Threads * Threads, batch * heads, 1);
                return;
            }
            if (ctx.backend is CPUBackend) {
                ExecuteOnCpu(query, key, value, g, beta, state, output, finalState, batch, tokens, heads, keySize, valueSize);
                return;
            }
            throw new NotSupportedException($"GatedDeltaRule does not support the {ctx.backend.backendType} backend.");
        }

        private unsafe void ExecuteOnCpu(Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> g, Tensor<float> beta,
            Tensor<float> state, Tensor<float> output, Tensor<float> finalState, int batch, int tokens, int heads, int keySize, int valueSize) {
            float* Data(Tensor<float> tensor) {
                var data = CPUTensorData.Pin(tensor);
                data.CompleteAllPendingOperations();
                return (float*)data.rawPtr;
            }
            float* q = Data(query), k = Data(key), v = Data(value), gp = Data(g), bp = Data(beta), s0 = Data(state);
            float* o = Data(output), s = Data(finalState);
            int stateSize = keySize * valueSize;
            Buffer.MemoryCopy(s0, s, (long)batch * heads * stateSize * sizeof(float), (long)batch * heads * stateSize * sizeof(float));
            var memory = new float[valueSize];
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < heads; h++) {
                    float* S = s + (b * heads + h) * stateSize;
                    for (int t = 0; t < tokens; t++) {
                        int row = (b * tokens + t) * heads + h;
                        float* kt = k + row * keySize, qt = q + row * keySize, vt = v + row * valueSize, ot = o + row * valueSize;
                        float decay = MathF.Exp(gp[row]), bt = bp[row];
                        Array.Clear(memory, 0, valueSize);
                        for (int i = 0; i < keySize; i++) {
                            for (int j = 0; j < valueSize; j++) {
                                S[i * valueSize + j] *= decay;
                                memory[j] += S[i * valueSize + j] * kt[i];
                            }
                        }
                        for (int j = 0; j < valueSize; j++) {
                            memory[j] = (vt[j] - memory[j]) * bt;
                            ot[j] = 0;
                        }
                        for (int i = 0; i < keySize; i++) {
                            float qi = qt[i] * scale;
                            for (int j = 0; j < valueSize; j++) {
                                S[i * valueSize + j] += kt[i] * memory[j];
                                ot[j] += S[i * valueSize + j] * qi;
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace Doji.AI {

    /// <summary>
    /// Linear attention operators for models built with the functional API of Inference Engine.
    /// </summary>
    public static class LinearAttention {

        /// <summary>
        /// The gated delta rule of Gated DeltaNet, as transformers' <c>torch_recurrent_gated_delta_rule</c> computes
        /// it without the normalization of <paramref name="query"/> and <paramref name="key"/>. A model with this
        /// layer can be saved to a <c>.sentis</c> file.
        /// </summary>
        /// <param name="query">(batch, tokens, heads, key size).</param>
        /// <param name="key">(batch, tokens, heads, key size).</param>
        /// <param name="value">(batch, tokens, heads, value size).</param>
        /// <param name="g">The log of the decay of the state, (batch, tokens, heads).</param>
        /// <param name="beta">The update strength, (batch, tokens, heads).</param>
        /// <param name="initialState">(batch, heads, key size, value size).</param>
        /// <param name="scale">The factor of the query, for example 1 / sqrt(key size).</param>
        /// <returns>The output, the shape of <paramref name="value"/>, and the state after the last token.</returns>
        public static (FunctionalTensor output, FunctionalTensor finalState) GatedDeltaRule(FunctionalTensor query, FunctionalTensor key,
            FunctionalTensor value, FunctionalTensor g, FunctionalTensor beta, FunctionalTensor initialState, float scale) {
            var outputs = FunctionalLayer.GatedDeltaRule(query, key, value, g, beta, initialState, scale);
            return (outputs[0], outputs[1]);
        }
    }
}
