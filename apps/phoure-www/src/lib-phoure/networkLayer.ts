import { arrayOf, f32, vec4f } from 'typegpu/data';
import tgpu, { type ExperimentalTgpuRoot } from 'typegpu/experimental';

export const layerLayout = tgpu.bindGroupLayout({
  weights: { storage: (n) => arrayOf(vec4f, n) },
  biases: { storage: (n) => arrayOf(f32, n) },
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
    bindGroup: root.createBindGroup(layerLayout, {
      weights: weightsBuffer,
      biases: biasesBuffer,
    }),
  };
}
