import tgpu, { type TgpuRoot } from 'typegpu';
import { textureLoad } from 'typegpu/std';
import * as d from 'typegpu/data';

import type { GBuffer } from '../gBuffer';
import { fullScreenTriangle } from '../shaders/fullScreenQuad';

type Options = {
  root: TgpuRoot;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  gBuffer: GBuffer;
};

const layout = tgpu.bindGroupLayout({
  sourceTexture: { texture: 'float' },
});

const mainFragFn = tgpu['~unstable'].fragmentFn({
  in: { pos: d.builtin.position, uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  const coord = d.vec2u(input.pos.xy);
  const color = textureLoad(layout.$.sourceTexture, coord, 0);

  // no post-processing for now

  return d.vec4f(color.xyz, 1.0);
});

export const PostProcessingStep = ({
  root,
  context,
  presentationFormat,
  gBuffer,
}: Options) => {
  const passColorAttachment = {
    // view is acquired and set in render loop.
    view: undefined as unknown as GPUTextureView,
    clearValue: [0, 0, 0, 1],
    loadOp: 'clear' as const,
    storeOp: 'store' as const,
  };

  const postProcessingPipeline = root['~unstable']
    .withVertex(fullScreenTriangle, {})
    .withFragment(mainFragFn, { format: presentationFormat })
    .createPipeline();

  return {
    perform() {
      // Updating color attachment
      const textureView = context.getCurrentTexture().createView();
      passColorAttachment.view = textureView;

      const externalBindGroup = root.createBindGroup(layout, {
        sourceTexture: gBuffer.outRawRenderView,
      });

      postProcessingPipeline
        .with(layout, externalBindGroup)
        .withColorAttachment(passColorAttachment)
        .draw(3);
    },
  };
};
