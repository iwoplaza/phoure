import { tgpu, d, std } from 'typegpu';
import { fullScreenTriangle } from '../../lib/shaders/fullScreenQuad';

const layout = tgpu.bindGroupLayout({
  sampler: { sampler: 'filtering' },
  texture: { texture: d.texture2d(d.f32) },
});

export const resampleLinear = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  'use gpu';
  // The shared triangle has bottom-left UVs; textures use a top-left origin.
  return std.textureSample(
    layout.$.texture,
    layout.$.sampler,
    d.vec2f(input.uv.x, 1 - input.uv.y),
  );
});

type Options = {
  device: GPUDevice;
  targetFormat: GPUTextureFormat;
  sourceTexture: GPUTextureView;
  targetTexture: GPUTextureView;
};

export const ResampleStep = ({
  device,
  targetFormat,
  sourceTexture,
  targetTexture,
}: Options) => {
  const root = tgpu.initFromDevice({ device });
  const sampler = device.createSampler({
    minFilter: 'linear',
    magFilter: 'linear',
  });
  const bindGroup = root.createBindGroup(layout, {
    sampler,
    texture: sourceTexture,
  });
  const pipeline = root
    .createRenderPipeline({
      vertex: fullScreenTriangle,
      fragment: resampleLinear,
      targets: { format: targetFormat },
    })
    .with(bindGroup);
  return {
    perform(commandEncoder: GPUCommandEncoder) {
      pipeline
        .with(commandEncoder)
        .withColorAttachment({ view: targetTexture })
        .draw(3);
    },
  };
};
