import { tgpu, d, std, type TgpuRoot } from 'typegpu';
import { convertRgbToY } from '@typegpu/color';
import type { GBuffer } from '../gBuffer';

const layout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
  output: { storage: d.arrayOf(d.f32), access: 'mutable' },
});

/** The original 3 × 3 vertical luminance derivative: [-1, 0, 1] per column. */
export const edgeDetectionCompute = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})((input) => {
  'use gpu';
  const size = std.textureDimensions(layout.$.source);
  if (input.gid.x >= size.x || input.gid.y >= size.y) return;
  let result = d.f32(0);
  for (let y = -1; y <= 1; y++) {
    for (let x = -1; x <= 1; x++) {
      const coord = std.clamp(
        d.vec2i(input.gid.xy) + d.vec2i(x, y),
        d.vec2i(),
        d.vec2i(size) - d.vec2i(1),
      );
      result +=
        d.f32(x) *
        convertRgbToY(std.textureLoad(layout.$.source, coord, 0).rgb);
    }
  }
  layout.$.output[input.gid.y * size.x + input.gid.x] = result;
});

export const EdgeDetectionStep = ({
  root,
  gBuffer,
  menderResultBuffer,
}: {
  root: TgpuRoot;
  gBuffer: GBuffer;
  menderResultBuffer: GPUBuffer;
}) => {
  const pipeline = root
    .createComputePipeline({ compute: edgeDetectionCompute })
    .with(
      root.createBindGroup(layout, {
        source: gBuffer.upscaledView,
        output: menderResultBuffer,
      }),
    );
  return {
    perform(commandEncoder: GPUCommandEncoder) {
      pipeline
        .with(commandEncoder)
        .dispatchWorkgroups(
          Math.ceil(gBuffer.size[0] / 8),
          Math.ceil(gBuffer.size[1] / 8),
        );
    },
  };
};
