using NUnit.Framework;
using System;
using Unity.InferenceEngine;

namespace Doji.AI.Editor.Tests {

    public class OpsTest {

        [TestCase(BackendType.CPU)]
        [TestCase(BackendType.GPUCompute)]
        public void NonZero_1D_ReturnsTheIndicesOfNonZeroValues(BackendType backend) {
            using Ops ops = new Ops(backend);
            float[] data = new float[] { 0, 1, 0, 2, 3, 0, 0, -1 };
            using Tensor<float> X = new Tensor<float>(new TensorShape(data.Length), data);
            Tensor<int> nonzero = ops.NonZero(X);
            Assert.That(nonzero.shape, Is.EqualTo(new TensorShape(1, 4)));
            CollectionAssert.AreEqual(new int[] { 1, 3, 4, 7 }, nonzero.DownloadToArray());
        }

        [Test]
        public void NonZero_2D_ReturnsOneRowPerDimension() {
            using Ops ops = new Ops(BackendType.CPU);
            float[] data = new float[] { 1, 0, 0, 0, 2, 3 };
            using Tensor<float> X = new Tensor<float>(new TensorShape(2, 3), data);
            Tensor<int> nonzero = ops.NonZero(X);
            Assert.That(nonzero.shape, Is.EqualTo(new TensorShape(2, 3)));
            // rows: [0, 1, 1], columns: [0, 1, 2]
            CollectionAssert.AreEqual(new int[] { 0, 1, 1, 0, 1, 2 }, nonzero.DownloadToArray());
        }

        [Test]
        public void NonZero_AllZero_ReturnsAnEmptyTensor() {
            using Ops ops = new Ops(BackendType.CPU);
            using Tensor<float> X = new Tensor<float>(new TensorShape(3), new float[3]);
            Tensor<int> nonzero = ops.NonZero(X);
            Assert.That(nonzero.shape, Is.EqualTo(new TensorShape(1, 0)));
        }

        [Test]
        public void ThrowIfDisposed_Throws_ForADisposedOpsResult() {
            using Ops ops = new Ops(BackendType.CPU);
            Tensor<float> zeros = ops.Zeros<float>(new TensorShape(2));
            Assert.DoesNotThrow(() => Ops.ThrowIfDisposed(zeros));
            zeros.Dispose();
            Assert.Throws<ObjectDisposedException>(() => Ops.ThrowIfDisposed(zeros));
        }

        [Test]
        public void TakeOwnership_Throws_WhenTheTensorWasDisposed() {
            using Ops ops = new Ops(BackendType.CPU);
            Tensor<float> zeros = ops.Zeros<float>(new TensorShape(2));
            zeros.Dispose();
            Assert.Throws<ObjectDisposedException>(() => ops.TakeOwnership(zeros));
        }

        [Test]
        public void TakeOwnership_KeepsTheTensorAliveAfterTheFlush() {
            using Ops ops = new Ops(BackendType.CPU);
            Tensor<float> zeros = ops.Zeros<float>(new TensorShape(2));
            using Tensor owned = ops.TakeOwnership(zeros);
            ops.FlushTensors();
            CollectionAssert.AreEqual(new float[] { 0, 0 }, zeros.DownloadToArray());
        }
    }
}
