import * as d from 'typegpu/data';
import tgpu, { type ExperimentalTgpuRoot } from 'typegpu/experimental';

export const layerLayout = tgpu.bindGroupLayout({
  weights: { storage: (n) => d.arrayOf(d.vec4f, n) },
  biases: { storage: (n) => d.arrayOf(d.f32, n) },
});

export function createNetworkLayer(
  root: ExperimentalTgpuRoot,
  weightData: Float32Array,
  biasData: Float32Array,
) {
  const weightsBuffer = root.device.createBuffer({
    size: weightData.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  {
    const mapping = new Float32Array(weightsBuffer.getMappedRange());
    mapping.set(weightData);
    weightsBuffer.unmap();
  }

  const biasesBuffer = root.device.createBuffer({
    size: biasData.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  {
    const mapping = new Float32Array(biasesBuffer.getMappedRange());
    mapping.set(biasData);
    biasesBuffer.unmap();
  }

  return {
    bindGroup: layerLayout.populate({
      weights: weightsBuffer,
      biases: biasesBuffer,
    }),
  };
}
