import * as d from 'typegpu/data';
import tgpu, { type TgpuRoot } from 'typegpu';

import type { GBuffer } from '../gBuffer';
import { fullScreenQuadVertexFn } from '../shaders/fullScreenQuad';

type Options = {
  root: TgpuRoot;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  gBuffer: GBuffer;
};

const layout = tgpu
  .bindGroupLayout({
    sourceTexture: { texture: 'float' },
  })
  .$name('Post Processing - Bind Group Layout');

const mainFragFn = tgpu['~unstable']
  .fragmentFn({ pos: d.builtin.position, uv: d.vec2f }, d.vec4f)
  .does(`(@builtin(position) coord_f: vec4f) -> @location(0) vec4f {
    var coord = vec2u(floor(coord_f.xy));

    let color = textureLoad(
      sourceTexture,
      coord,
      0
    );

    // no post-processing for now

    return vec4f(color.rgb, 1.0);
  }`)
  .$uses({ sourceTexture: layout.bound.sourceTexture });

export const PostProcessingStep = ({
  root,
  context,
  presentationFormat,
  gBuffer,
}: Options) => {
  const passColorAttachment: GPURenderPassColorAttachment = {
    // view is acquired and set in render loop.
    view: undefined as unknown as GPUTextureView,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear',
    storeOp: 'store',
  };

  const pipeline = root['~unstable']
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(mainFragFn, { format: presentationFormat })
    .createPipeline()
    .$name('Post Processing Pipeline');

  return {
    perform() {
      // Updating color attachment
      const textureView = context.getCurrentTexture().createView();
      passColorAttachment.view = textureView;

      const externalBindGroup = layout.populate({
        sourceTexture: gBuffer.outRawRenderView,
      });

      pipeline
        .with(layout, externalBindGroup)
        .withColorAttachment(passColorAttachment)
        .draw(6);
    },
  };
};
