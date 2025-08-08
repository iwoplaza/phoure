import { accessViewportSize } from '@typegpu/common';
import { convertRgbToY } from '@typegpu/color';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';

import {
  convolveFn,
  inChannelsQuarter,
  inChannelsSlot,
  kernelRadiusSlot,
  outChannelsSlot,
} from 'src/lib/GameEngine/convolve.ts';
import type { GBuffer } from 'src/lib/gBuffer.ts';
import { fullScreenQuadVertexFn } from 'src/lib/shaders/fullScreenQuad.ts';
import { combinationEntryFn, combinationLayout } from './combineShader.ts';
import { Model7 } from './model7.ts';
import { createNetworkLayer, layerLayout } from './networkLayer.ts';

const blockDim = 8;

const FIRST_DEPTH = 8;
const SECOND_DEPTH = 8;

type Options = {
  root: TgpuRoot;
  gBuffer: GBuffer;
  targetTexture: () => GPUTextureView;
};

const reluSlot = tgpu.slot<boolean>();
const inputFromGBufferSlot = tgpu.slot<boolean>();
const BLOCK_SIZE = 8;

// const convolveLocalFn = wgsl.fn`(local: vec2u, result: ptr<function, array<f32, ${outChannelsSlot}>>) {
//   var weight_idx: u32 = 0;

//   for (var out_c: u32 = 0; out_c < ${outChannelsSlot}; out_c++) {
//     let result_channel = &(*result)[out_c];

//     for (var i: u32 = ${TILE_PADDING}-${kernelRadiusSlot}; i <= ${TILE_PADDING}+${kernelRadiusSlot}; i++) {
//       for (var j: u32 = ${TILE_PADDING}-${kernelRadiusSlot}; j <= ${TILE_PADDING}+${kernelRadiusSlot}; j++) {
//         let tile_slice = &(tile[local.x + i][local.y + j]);

//         for (var in_c: u32 = 0; in_c < ${inChannelsSlot} / 4; in_c++) {
//           let some = tile[local.x + i][local.y + j][in_c];
//           (*result_channel) += dot((*tile_slice)[in_c], conv_weights[weight_idx]);
//           // (*result_channel) = conv_weights[weight_idx].x * f32(i * j) / 10.0;
//           weight_idx++;
//         }
//       }
//     }
//   }
// }`;

const ioLayout = tgpu.bindGroupLayout({
  output_buffer: {
    storage: (n: number) => d.arrayOf(d.f32, n),
    access: 'mutable',
  },
  input_buffer: { storage: (n: number) => d.arrayOf(d.vec4f, n) },
  blurred_tex: { texture: 'unfilterable-float' },
  aux_tex: { texture: 'unfilterable-float' },
});

const { weights, biases } = layerLayout.bound;

const sampleGlobal = tgpu['~unstable'].derived(() => {
  return tgpu.fn([
    d.i32,
    d.i32,
    d.ptrFn(d.arrayOf(d.vec4f, inChannelsQuarter.$)),
  ])((x, y, result) => {
    const canvasSize = accessViewportSize.$;
    const coord = d.vec2u(
      d.u32(std.max(0, std.min(x, d.i32(canvasSize.x) - 1))),
      d.u32(std.max(0, std.min(y, d.i32(canvasSize.y) - 1))),
    );

    if (inputFromGBufferSlot.$) {
      const blurred = std.textureLoad(ioLayout.$.blurred_tex, coord, 0);

      const aux = std.textureLoad(ioLayout.$.aux_tex, coord, 0);

      result[0] = d.vec4f(
        convertRgbToY(blurred.xyz),
        aux.z, // albedo luminance
        aux.x, // normal.x
        aux.y, // normal.y
      );
      result[1] = d.vec4f(
        aux.w, // emission luminance
        0, // zero padding
        0, // zero padding
        0, // zero padding
      );
    } else {
      for (let i = d.u32(0); i < inChannelsQuarter.$; i++) {
        const index =
          (coord.y * d.u32(canvasSize.x) + coord.x) * inChannelsQuarter.$ + i;

        result[i] = ioLayout.$.input_buffer[index];
      }
    }
  });
});

const f32Array = (n: number) => d.arrayOf(d.f32, n);

const applyReLU = tgpu['~unstable'].derived(() => {
  const outChannels = outChannelsSlot.$;

  return tgpu.fn([d.ptrFn(f32Array(outChannels))])((result) => {
    for (let i = 0; i < outChannels; i++) {
      result[i] = std.max(0, result[i]);
    }
  });
});

const readKernel = tgpu.fn([d.u32], d.vec4f)((idx) => weights.$[idx]);

const menderConvolveFn = convolveFn({
  sampleFiller: sampleGlobal,
  kernelReader: readKernel,
});

const entryComputeFn = tgpu['~unstable'].computeFn({
  workgroupSize: [BLOCK_SIZE, BLOCK_SIZE],
  in: {
    gid: d.builtin.globalInvocationId,
  },
})`{
    var result: array<f32, OUT_CHANNELS>;

    for (var i = 0; i < OUT_CHANNELS; i += 1) {
      result[i] = biases[i];
    }

    menderConvolveFn(in.gid.xy, &result);

    if (reluSlot) {
      applyReLU(&result);
    }

    let canvasSize = accessViewportSize;

    let output_buffer_begin =
      (in.gid.y * u32(canvasSize.x) +
      in.gid.x) * OUT_CHANNELS;

    for (var i: u32 = 0; i < OUT_CHANNELS; i++) {
      output_buffer[output_buffer_begin + i] = result[i];
    }
  }`.$uses({
  menderConvolveFn,
  reluSlot,
  applyReLU,
  biases,
  accessViewportSize,
  output_buffer: ioLayout.bound.output_buffer,
  OUT_CHANNELS: outChannelsSlot,
});

export const MenderStep = ({ root, gBuffer, targetTexture }: Options) => {
  // Resource locators

  const viewportSizeUniform = root.createUniform(d.vec2f);

  // Textures

  const firstWorkBuffer = root
    .createBuffer(
      d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1] * FIRST_DEPTH),
    )
    .$usage('storage');

  const secondWorkBuffer = root
    .createBuffer(
      d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1] * SECOND_DEPTH),
    )
    .$usage('storage');

  const mendedResultBuffer = root
    .createBuffer(d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1]))
    .$usage('storage');

  //
  // Weights & Biases
  //

  const convLayers = [
    createNetworkLayer(root, Model7.Conv1Weight, Model7.Conv1Bias),
    createNetworkLayer(root, Model7.Conv2Weight, Model7.Conv2Bias),
    createNetworkLayer(root, Model7.Conv3Weight, Model7.Conv3Bias),
  ];

  const makeLayerPipeline = (options: {
    label: string;
    kernelRadius: number;
    inChannels: number;
    outChannels: number;
    relu: boolean;
    inputFromGBuffer: boolean;
  }) => {
    return (
      root['~unstable']
        // filling slots
        .with(kernelRadiusSlot, options.kernelRadius)
        .with(inChannelsSlot, options.inChannels)
        .with(outChannelsSlot, options.outChannels)
        .with(reluSlot, options.relu)
        .with(inputFromGBufferSlot, options.inputFromGBuffer)
        .with(accessViewportSize, viewportSizeUniform)
        // ---
        .withCompute(entryComputeFn)
        .createPipeline()
        .$name(options.label)
    );
  };

  const pipelines = [
    makeLayerPipeline({
      label: 'Layer #1 Pipeline',
      kernelRadius: 4,
      inChannels: 8,
      outChannels: FIRST_DEPTH,
      relu: true,
      inputFromGBuffer: true,
    }),
    makeLayerPipeline({
      label: 'Layer #2 Pipeline',
      kernelRadius: 2,
      inChannels: FIRST_DEPTH,
      outChannels: SECOND_DEPTH,
      relu: true,
      inputFromGBuffer: false,
    }),
    makeLayerPipeline({
      label: 'Layer #3 Pipeline',
      kernelRadius: 2,
      inChannels: SECOND_DEPTH,
      outChannels: 1,
      relu: false,
      inputFromGBuffer: false,
    }),
  ];

  const ioBindGroups = [
    root.createBindGroup(ioLayout, {
      blurred_tex: gBuffer.upscaledView,
      aux_tex: gBuffer.auxView,
      output_buffer: firstWorkBuffer,
      // biome-ignore lint/suspicious/noExplicitAny: its fine
      input_buffer: secondWorkBuffer as any, // <- UNUSED
    }),
    root.createBindGroup(ioLayout, {
      // biome-ignore lint/suspicious/noExplicitAny: its fine
      input_buffer: firstWorkBuffer as any,
      output_buffer: secondWorkBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
    root.createBindGroup(ioLayout, {
      // biome-ignore lint/suspicious/noExplicitAny: its fine
      input_buffer: secondWorkBuffer as any,
      output_buffer: mendedResultBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
  ];

  // ---
  // Combination pass
  // ---

  const combinationPipeline = root['~unstable']
    .with(accessViewportSize, viewportSizeUniform)
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(combinationEntryFn, { format: 'rgba8unorm' })
    .createPipeline();

  const combinationBindGroup = root.createBindGroup(combinationLayout, {
    blurredTexture: gBuffer.upscaledView,
    mendedBuffer: mendedResultBuffer,
  });

  viewportSizeUniform.write(d.vec2f(...gBuffer.size));

  return {
    perform() {
      for (let i = 0; i < 3; ++i) {
        pipelines[i]
          .with(ioLayout, ioBindGroups[i])
          .with(layerLayout, convLayers[i].bindGroup)
          .dispatchWorkgroups(
            Math.ceil(gBuffer.size[0] / blockDim),
            Math.ceil(gBuffer.size[1] / blockDim),
          );
      }

      // Combining the convolved result with the initial blurry render

      combinationPipeline
        .withColorAttachment({
          view: targetTexture(),

          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        })
        .with(combinationLayout, combinationBindGroup)
        .draw(6);
    },
  };
};
