import { tgpu, type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

import { displayModeAtom } from '#src/lib/controlAtoms.ts';
import { fullScreenTriangle } from '#src/lib/shaders/fullScreenQuad.ts';
import { store } from '#src/lib/store.ts';
import type { GBuffer } from '../gBuffer.ts';

const CHANNEL_SPLIT = 0;
const CHANNEL_COLOR = 1;
const CHANNEL_ALBEDO = 2;
const CHANNEL_NORMAL = 3;

export const channelMode = tgpu.accessor(d.u32);

const layout = tgpu.bindGroupLayout({
  blurredTex: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
  auxTex: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
});

export const mainFragFn = tgpu.fragmentFn({
  in: { coord_f: d.builtin.position, uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  'use gpu';
  const coord = d.vec2i(std.floor(input.coord_f.xy));
  const blurred = std.textureLoad(layout.$.blurredTex, coord, 0);
  const aux = std.textureLoad(layout.$.auxTex, coord, 0);
  const normal = d.vec4f((aux.x + 1) * 0.5, (aux.y + 1) * 0.5, 0.5, 1);
  const mode = channelMode.$;
  if (
    mode === CHANNEL_NORMAL ||
    (mode === CHANNEL_SPLIT && input.uv.x < 0.33)
  ) {
    return normal;
  }
  if (
    mode === CHANNEL_ALBEDO ||
    (mode === CHANNEL_SPLIT && input.uv.x < 0.66)
  ) {
    return d.vec4f(aux.z, aux.z, aux.z, 1);
  }
  if (mode === CHANNEL_COLOR || mode === CHANNEL_SPLIT) {
    return d.vec4f(blurred.rgb, 1);
  }
  return d.vec4f();
});

export function makeGBufferDebugger(
  root: TgpuRoot,
  presentationFormat: GPUTextureFormat,
  gBuffer: GBuffer,
) {
  const passColorAttachment = {
    // view is acquired and set in render loop.
    view: undefined as unknown as GPUTextureView,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear' as const,
    storeOp: 'store' as const,
  };

  const channelModeUniform = root.createUniform(d.u32, CHANNEL_SPLIT);

  const pipeline = root
    .with(channelMode, channelModeUniform)
    .createRenderPipeline({
      vertex: fullScreenTriangle,
      fragment: mainFragFn,
      targets: {
        format: presentationFormat,
      },
    })
    .$name('GBuffer Debugger - pipeline')
    .with(
      layout,
      root.createBindGroup(layout, {
        blurredTex: gBuffer.upscaledView,
        auxTex: gBuffer.auxView,
      }),
    );

  return {
    perform(ctx: GPUCanvasContext) {
      const textureView = ctx.getCurrentTexture().createView();
      passColorAttachment.view = textureView;

      const mode = store.get(displayModeAtom);
      let channelMode = CHANNEL_SPLIT;
      if (mode === 'g-buffer-color') {
        channelMode = CHANNEL_COLOR;
      } else if (mode === 'g-buffer-albedo') {
        channelMode = CHANNEL_ALBEDO;
      } else if (mode === 'g-buffer-normal') {
        channelMode = CHANNEL_NORMAL;
      }

      channelModeUniform.write(channelMode);
      pipeline.withColorAttachment(passColorAttachment).draw(3);
    },
  };
}
