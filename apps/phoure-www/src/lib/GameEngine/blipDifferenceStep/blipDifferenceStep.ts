import { tgpu, d, std, type TgpuRoot } from 'typegpu';
import { fullScreenTriangle } from '../../shaders/fullScreenQuad';

type Options = {
  root: TgpuRoot;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  textures: [() => GPUTextureView, () => GPUTextureView];
};

const layout = tgpu.bindGroupLayout({
  textureA: { texture: d.texture2d(d.f32) },
  textureB: { texture: d.texture2d(d.f32) },
});

export const differenceFragment = tgpu.fragmentFn({
  in: { coordFloat: d.builtin.position },
  out: d.vec4f,
})((input) => {
  'use gpu';
  const coord = d.vec2u(input.coordFloat.xy);
  const a = std.textureLoad(layout.$.textureA, coord, 0);
  const b = std.textureLoad(layout.$.textureB, coord, 0);
  return d.vec4f(std.abs(a.rgb - b.rgb), 1);
});

export const BlipDifferenceStep = ({
  root,
  context,
  presentationFormat,
  textures,
}: Options) => {
  const pipeline = root.createRenderPipeline({
    vertex: fullScreenTriangle,
    fragment: differenceFragment,
    targets: { format: presentationFormat },
  });
  return {
    perform() {
      const bindGroup = root.createBindGroup(layout, {
        textureA: textures[0](),
        textureB: textures[1](),
      });
      pipeline.with(bindGroup).withColorAttachment({ view: context }).draw(3);
    },
  };
};
