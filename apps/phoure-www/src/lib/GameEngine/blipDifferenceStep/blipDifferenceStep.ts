import { fullScreenQuadVertexFn } from '@/shaders/fullScreenQuad';
import * as d from 'typegpu/data';
import tgpu, { type ExperimentalTgpuRoot } from 'typegpu/experimental';

type Options = {
  root: ExperimentalTgpuRoot;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  textures: [() => GPUTextureView, () => GPUTextureView];
};

const fragFn = tgpu
  .fragmentFn({ coordFloat: d.builtin.position }, d.vec4f)
  .does(`(@builtin(position) coordFloat: vec4f) -> @location(0) vec4f {
    var coord = vec2u(floor(coordFloat.xy));

    let color_a = textureLoad(
      texture_a,
      coord,
      0
    );

    let color_b = textureLoad(
      texture_b,
      coord,
      0
    );

    return vec4f(abs(color_a.rgb - color_b.rgb), 1.0);
  }`)
  .$uses({});

export const BlipDifferenceStep = ({
  root,
  context,
  presentationFormat,
  textures,
}: Options) => {
  const layout = tgpu
    .bindGroupLayout({
      textureA: { texture: 'float' },
      textureB: { texture: 'float' },
    })
    .$name('Blip Difference - Bind Group Layout');

  const pipeline = root
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(fragFn, { format: presentationFormat })
    .createPipeline();

  const passColorAttachment: GPURenderPassColorAttachment = {
    // view is acquired and set in render loop.
    view: undefined as unknown as GPUTextureView,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear',
    storeOp: 'store',
  };

  return {
    perform() {
      // Updating color attachment
      const textureView = context.getCurrentTexture().createView();
      passColorAttachment.view = textureView;

      const bindGroup = root.createBindGroup(layout, {
        textureA: textures[0](),
        textureB: textures[1](),
      });

      pipeline
        .withColorAttachment(passColorAttachment)
        .with(layout, bindGroup)
        .draw(6);
    },
  };
};
