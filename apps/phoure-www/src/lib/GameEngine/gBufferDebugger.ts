import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';

import { displayModeAtom } from 'src/lib/controlAtoms.ts';
import { fullScreenQuadVertexFn } from 'src/lib/shaders/fullScreenQuad.ts';
import { store } from 'src/lib/store.ts';
import type { GBuffer } from '../gBuffer.ts';

const CHANNEL_SPLIT = 0;
const CHANNEL_COLOR = 1;
const CHANNEL_ALBEDO = 2;
const CHANNEL_NORMAL = 3;

const channelMode = tgpu['~unstable'].accessor(d.u32);

const layout = tgpu.bindGroupLayout({
  blurredTex: { texture: 'unfilterable-float' },
  auxTex: { texture: 'unfilterable-float' },
});

const mainFragFn = tgpu['~unstable']
  .fragmentFn({
    in: { coord_f: d.builtin.position, uv: d.vec2f },
    out: d.vec4f,
  })(/* wgsl */ `{
    let coord = vec2<i32>(floor(in.coord_f.xy));
    let channel_mode = channelMode;

    let blurred = textureLoad(
      blurredTex,
      coord,
      0
    );

    let aux = textureLoad(
      auxTex,
      coord,
      0
    );

    let normal = vec4(
      (aux.x + 1.0) * 0.5, // normal.x
      (aux.y + 1.0) * 0.5, // normal.y
      0.5,
      1.0,
    );

    var result: vec4<f32>;

    let c = in.uv;
    if (channel_mode == CHANNEL_SPLIT) {
      if (c.x < 0.33) {
        // NORMALS
        result = normal;
      }
      else if (c.x < 0.66) {
        // ALBEDO_LUMI

        let albedo = aux.z;
        result = vec4(
          albedo,
          albedo,
          albedo,
          1.0,
        );
      }
      else {
        // COLOR

        result = vec4(
          blurred.rgb,
          1.0,
        );
      }
    } else if (channel_mode == CHANNEL_COLOR) {
      result = vec4(
        blurred.rgb,
        1.0,
      );
    } else if (channel_mode == CHANNEL_ALBEDO) {
      let albedo = aux.z;
      result = vec4(
        albedo,
        albedo,
        albedo,
        1.0,
      );
    } else if (channel_mode == CHANNEL_NORMAL) {
      result = normal;
    }

    return result;
  }`)
  .$uses({
    blurredTex: layout.bound.blurredTex,
    auxTex: layout.bound.auxTex,
    channelMode,
    CHANNEL_SPLIT,
    CHANNEL_COLOR,
    CHANNEL_ALBEDO,
    CHANNEL_NORMAL,
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

  const channelModeUniform = root['~unstable'].createUniform(
    d.u32,
    CHANNEL_SPLIT,
  );

  const pipeline = root['~unstable']
    .with(channelMode, channelModeUniform)
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(mainFragFn, {
      format: presentationFormat,
    })
    .createPipeline()
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
      pipeline.withColorAttachment(passColorAttachment).draw(6);
    },
  };
}
