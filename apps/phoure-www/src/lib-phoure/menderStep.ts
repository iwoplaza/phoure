import { accessViewportSize } from '@typegpu/common';
import { convertRgbToY } from '#src/lib-phoure/color.ts';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

import {
  convolveFn,
  inChannelsQuarter,
  inChannelsSlot,
  kernelRadiusSlot,
  outChannelsSlot,
} from '#src/lib/GameEngine/convolve.ts';
import type { GBuffer } from '#src/lib/gBuffer.ts';
import { fullScreenTriangle } from '#src/lib/shaders/fullScreenQuad.ts';
import {
  combinationEntryFn,
  layout as combinationLayout,
} from './combineShader.ts';
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

export const reluSlot = tgpu.slot<boolean>();
export const inputFromGBufferSlot = tgpu.slot<boolean>();
const BLOCK_SIZE = 8;

const ioLayout = tgpu.bindGroupLayout({
  output_buffer: {
    storage: (n: number) => d.arrayOf(d.f32, n),
    access: 'mutable',
  },
  input_buffer: { storage: (n: number) => d.arrayOf(d.vec4f, n) },
  blurred_tex: {
    texture: d.texture2d(d.f32),
    sampleType: 'unfilterable-float',
  },
  aux_tex: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
});

const sampleGlobal = tgpu.lazy(() => {
  return tgpu.fn([
    d.i32,
    d.i32,
    d.ptrFn(d.arrayOf(d.vec4f, inChannelsQuarter.$)),
  ])((x, y, result) => {
    'use gpu';
    const canvasSize = accessViewportSize.$;
    const coord = d.vec2u(
      d.u32(std.max(0, std.min(x, d.i32(canvasSize.x) - 1))),
      d.u32(std.max(0, std.min(y, d.i32(canvasSize.y) - 1))),
    );

    if (inputFromGBufferSlot.$) {
      const blurred = std.textureLoad(ioLayout.$.blurred_tex, coord, 0);

      const aux = std.textureLoad(ioLayout.$.aux_tex, coord, 0);

      result.$[0] = d.vec4f(
        convertRgbToY(blurred.xyz),
        aux.z, // albedo luminance
        aux.x, // normal.x
        aux.y, // normal.y
      );
      result.$[1] = d.vec4f(
        aux.w, // emission luminance
        0, // zero padding
        0, // zero padding
        0, // zero padding
      );
    } else {
      for (let i = d.u32(0); i < inChannelsQuarter.$; i++) {
        const index =
          (coord.y * d.u32(canvasSize.x) + coord.x) * inChannelsQuarter.$ + i;

        result.$[i] = d.vec4f(ioLayout.$.input_buffer[index]);
      }
    }
  });
});

const f32Array = (n: number) => d.arrayOf(d.f32, n);

const applyReLU = tgpu.lazy(() => {
  const outChannels = outChannelsSlot.$;

  return tgpu.fn([d.ptrFn(f32Array(outChannels))])((result) => {
    'use gpu';
    for (let i = 0; i < outChannels; i++) {
      result.$[i] = std.max(0, result.$[i]);
    }
  });
});

const readKernel = tgpu.fn(
  [d.u32],
  d.vec4f,
)((idx) => {
  'use gpu';
  return layerLayout.$.weights[idx];
});

const menderConvolveFn = convolveFn({
  sampleFiller: sampleGlobal,
  kernelReader: readKernel,
});

const ResultArray = tgpu.lazy(() => d.arrayOf(d.f32, outChannelsSlot.$));

export const entryComputeFn = tgpu.computeFn({
  workgroupSize: [BLOCK_SIZE, BLOCK_SIZE],
  in: { gid: d.builtin.globalInvocationId },
})((input) => {
  'use gpu';
  const canvasSize = accessViewportSize.$;
  if (
    input.gid.x >= d.u32(canvasSize.x) ||
    input.gid.y >= d.u32(canvasSize.y)
  ) {
    return;
  }
  const result = ResultArray.$();
  for (let i = d.u32(0); i < outChannelsSlot.$; i++) {
    result[i] = layerLayout.$.biases[i];
  }
  menderConvolveFn.$(input.gid.xy, d.ref(result));
  if (reluSlot.$) {
    applyReLU.$(d.ref(result));
  }
  const outputBegin =
    (input.gid.y * d.u32(canvasSize.x) + input.gid.x) * outChannelsSlot.$;
  for (let i = d.u32(0); i < outChannelsSlot.$; i++) {
    ioLayout.$.output_buffer[outputBegin + i] = result[i];
  }
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
      root
        // filling slots
        .with(kernelRadiusSlot, options.kernelRadius)
        .with(inChannelsSlot, options.inChannels)
        .with(outChannelsSlot, options.outChannels)
        .with(reluSlot, options.relu)
        .with(inputFromGBufferSlot, options.inputFromGBuffer)
        .with(accessViewportSize, viewportSizeUniform)
        // ---
        .createComputePipeline({ compute: entryComputeFn })
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

  // Reinterpret contiguous f32 storage as vec4f inputs (four scalars per vector).
  // Raw buffers express this shared byte layout without an unsafe TypeScript cast.
  const ioBindGroups = [
    root.createBindGroup(ioLayout, {
      blurred_tex: gBuffer.upscaledView,
      aux_tex: gBuffer.auxView,
      output_buffer: firstWorkBuffer,
      input_buffer: root.unwrap(secondWorkBuffer), // <- UNUSED
    }),
    root.createBindGroup(ioLayout, {
      input_buffer: root.unwrap(firstWorkBuffer),
      output_buffer: secondWorkBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
    root.createBindGroup(ioLayout, {
      input_buffer: root.unwrap(secondWorkBuffer),
      output_buffer: mendedResultBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
  ];

  // ---
  // Combination pass
  // ---

  const combinationPipeline = root
    .with(accessViewportSize, viewportSizeUniform)
    .createRenderPipeline({
      vertex: fullScreenTriangle,
      fragment: combinationEntryFn,
      targets: { format: 'rgba8unorm' },
    });

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
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        })
        .with(combinationLayout, combinationBindGroup)
        .draw(3);
    },
  };
};
