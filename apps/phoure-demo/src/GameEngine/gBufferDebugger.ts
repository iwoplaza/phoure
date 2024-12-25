import * as d from 'typegpu/data';
import tgpu, {
  asUniform,
  type ExperimentalTgpuRoot,
  type TgpuFn,
} from 'typegpu/experimental';

import { displayModeAtom } from '@/controlAtoms';
import { fullScreenQuadVertexFn } from '@/shaders/fullScreenQuad';
import { store } from '@/store';
import type { GBuffer } from '../gBuffer';

const CHANNEL_SPLIT = 0;
const CHANNEL_COLOR = 1;
const CHANNEL_ALBEDO = 2;
const CHANNEL_NORMAL = 3;

const getChannelModeSlot = tgpu.slot<TgpuFn<[], d.U32>>();

const layout = tgpu.bindGroupLayout({
  blurredTex: { texture: 'unfilterable-float' },
  auxTex: { texture: 'unfilterable-float' },
});

const mainFragFn = tgpu
  .fragmentFn({ pos: d.builtin.position, uv: d.vec2f }, d.vec4f)
  .does(/* wgsl */ `(@builtin(position) coord_f: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
    let coord = vec2<i32>(floor(coord_f.xy));
    let channel_mode = getChannelModeSlot();

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

    let c = uv;
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
    getChannelModeSlot,
    CHANNEL_SPLIT,
    CHANNEL_COLOR,
    CHANNEL_ALBEDO,
    CHANNEL_NORMAL,
  });

export function makeGBufferDebugger(
  root: ExperimentalTgpuRoot,
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

  const channelModeBuffer = root
    .createBuffer(d.u32, CHANNEL_SPLIT)
    .$usage('uniform');
  const channelModeUniform = asUniform(channelModeBuffer);

  const myGetChannelMode = tgpu
    .fn([], d.u32)
    .does(`() -> u32 {
      return channelModeUniform;
    }`)
    .$uses({ channelModeUniform });

  const pipeline = root
    .with(getChannelModeSlot, myGetChannelMode)
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(mainFragFn, {
      format: presentationFormat,
    })
    .createPipeline()
    .$name('GBuffer Debugger - pipeline')
    .with(
      layout,
      layout.populate({
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

      channelModeBuffer.write(channelMode);
      pipeline.withColorAttachment(passColorAttachment).draw(6);
    },
  };
}
