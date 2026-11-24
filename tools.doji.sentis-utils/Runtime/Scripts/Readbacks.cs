using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.InferenceEngine;

namespace Doji.AI {

    /// <summary>
    /// Removes the GPU readbacks of small CPU tensors from a model, for example of a sequence length that a model
    /// computes from the shape of an input.
    /// </summary>
    /// <remarks>
    /// On <see cref="BackendType.GPUCompute"/>, a worker runs the layers that compute shapes and other small values on
    /// the CPU. When a GPU layer reads such a tensor, the tensor moves to the GPU. When a later layer reads its values
    /// on the CPU, the worker downloads it, and the download waits until the GPU has done all the work before it.
    /// <see cref="Avoid"/> gives each such tensor a CPU copy: a second copy of the layer that computes it, which the
    /// layers that read it on the CPU use.
    /// </remarks>
    public static class Readbacks {

        private static readonly MethodInfo Clone = typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic);

        /// <summary>
        /// Gives each tensor of <paramref name="model"/> that a GPU layer reads before a layer reads it on the CPU a
        /// CPU copy, when a CPU layer with one output computes it. The outputs of the model stay the same.
        /// </summary>
        /// <returns>The number of copies it added.</returns>
        public static int Avoid(Model model) {
            int added = 0;
            // A copy can need copies of its own inputs, which the next pass finds.
            for (int pass = 0; pass < 8; pass++) {
                int count = AvoidOnce(model);
                if (count == 0) {
                    break;
                }
                added += count;
            }
            return added;
        }

        /// <summary>
        /// The tensors of <paramref name="model"/> that a worker on <see cref="BackendType.GPUCompute"/> downloads from
        /// the GPU in each run, because a GPU layer reads them before a layer reads them on the CPU.
        /// </summary>
        public static List<int> Find(Model model) {
            var cpu = CPUFallbackCalculator.Calculate(model, BackendType.GPUCompute, out _);
            var firstGpuRead = FirstGpuReads(model, cpu);
            var found = new List<int>();
            var seen = new HashSet<int>();
            for (int i = 0; i < model.layers.Count; i++) {
                var layer = model.layers[i];
                for (int j = 0; j < layer.inputs.Length; j++) {
                    int input = layer.inputs[j];
                    if (input >= 0 && ReadsOnCpu(layer, j, cpu) && firstGpuRead.TryGetValue(input, out int first) && first < i && seen.Add(input)) {
                        found.Add(input);
                    }
                }
            }
            return found;
        }

        private static int AvoidOnce(Model model) {
            var cpu = CPUFallbackCalculator.Calculate(model, BackendType.GPUCompute, out _);
            var producers = new Dictionary<int, Layer>();
            foreach (var layer in model.layers) {
                foreach (int output in layer.outputs) {
                    producers[output] = layer;
                }
            }
            var copies = new Dictionary<int, int>();
            int next = HalfPrecision.NextIndex(model);
            foreach (int tensor in Find(model)) {
                if (producers.TryGetValue(tensor, out var producer) && producer.outputs.Length == 1 && cpu.Contains(tensor)) {
                    copies[tensor] = next++;
                }
            }
            if (copies.Count == 0) {
                return 0;
            }

            var layers = new List<Layer>(model.layers.Count + copies.Count);
            foreach (var layer in model.layers) {
                // Each copy goes right before the layer it copies, so it reads the inputs of that layer first.
                if (copies.TryGetValue(layer.outputs[0], out int copyIndex)) {
                    var copy = (Layer)Clone.Invoke(layer, null);
                    copy.outputs = new[] { copyIndex };
                    copy.inputs = Redirect(layer, cpu, copies, (int[])layer.inputs.Clone());
                    layers.Add(copy);
                    layers.Add(layer);
                    continue;
                }
                layer.inputs = Redirect(layer, cpu, copies, layer.inputs);
                layers.Add(layer);
            }
            model.layers = layers;
            return copies.Count;
        }

        // Reads of copied tensors on the CPU read the copies.
        private static int[] Redirect(Layer layer, HashSet<int> cpu, Dictionary<int, int> copies, int[] inputs) {
            for (int j = 0; j < inputs.Length; j++) {
                if (inputs[j] >= 0 && copies.TryGetValue(inputs[j], out int copy) && ReadsOnCpu(layer, j, cpu)) {
                    inputs[j] = copy;
                }
            }
            return inputs;
        }

        // Whether the layer reads the values of input j on the CPU: a GPU layer reads its CPU-read inputs there, and
        // a CPU layer all inputs except those it reads only the shape of.
        private static bool ReadsOnCpu(Layer layer, int j, HashSet<int> cpu) {
            return cpu.Contains(layer.outputs[0]) ? !layer.IsInputNoDataDependency(j) : layer.IsInputCPURead(j);
        }

        // The first GPU layer that reads each tensor on the GPU.
        private static Dictionary<int, int> FirstGpuReads(Model model, HashSet<int> cpu) {
            var first = new Dictionary<int, int>();
            for (int i = 0; i < model.layers.Count; i++) {
                var layer = model.layers[i];
                if (cpu.Contains(layer.outputs[0])) {
                    continue;
                }
                for (int j = 0; j < layer.inputs.Length; j++) {
                    int input = layer.inputs[j];
                    if (input >= 0 && !layer.IsInputCPURead(j) && !layer.IsInputNoDataDependency(j) && !first.ContainsKey(input)) {
                        first[input] = i;
                    }
                }
            }
            return first;
        }
    }
}
