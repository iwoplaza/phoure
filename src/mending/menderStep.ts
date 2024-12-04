import * as d from 'typegpu/data';
import tgpu, {
  asUniform,
  wgsl,
  type ExperimentalTgpuRoot,
  type Wgsl,
} from 'typegpu/experimental';

import { Model7 } from '../model7';
import type { GBuffer } from '../gBuffer';
import { layerLayout, createNetworkLayer } from './networkLayer';
import { fullScreenQuadVertexFn } from '../shaders/fullScreenQuad';
import { convertRgbToY } from '../GameEngine/sdfRenderer/colorUtils';
import { convolveFn } from '../GameEngine/convolve';
import { combinationEntryFn, combinationLayout } from './combineShader';
import { getViewportSizeSlot } from '@/GameEngine/commonSlots';

const blockDim = 8;

const FIRST_DEPTH = 8;
const SECOND_DEPTH = 8;

type Options = {
  root: ExperimentalTgpuRoot;
  gBuffer: GBuffer;
  targetTexture: () => GPUTextureView;
};

const kernelRadiusSlot = wgsl.slot<number>().$name('kernel_radius');
/**
 * Has to be divisible by 4
 */
const inChannelsSlot = wgsl.slot<number>().$name('in_channels');
const outChannelsSlot = wgsl.slot<number>().$name('out_channels');
const reluSlot = wgsl.slot<boolean>().$name('relu');
const inputFromGBufferSlot = wgsl.slot<boolean>().$name('input_from_gbuffer');
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
  output_buffer: { storage: (n) => d.arrayOf(d.f32, n), access: 'mutable' },
  input_buffer: { storage: (n) => d.arrayOf(d.vec4f, n) },
  blurred_tex: { texture: 'unfilterable-float' },
  aux_tex: { texture: 'unfilterable-float' },
});

console.log(ioLayout.bound);

const { weights, biases } = layerLayout.bound;

const sampleGlobal =
  wgsl.fn`(x: i32, y: i32, result: ptr<function, array<vec4f, ${inChannelsSlot} / 4>>) {
  let canvasSize = ${getViewportSizeSlot}();
  let coord = vec2u(
    u32(max(0, min(x, i32(canvasSize.x) - 1))),
    u32(max(0, min(y, i32(canvasSize.y) - 1))),
  );

  if (${inputFromGBufferSlot}) {
    let blurred = textureLoad(
      ${ioLayout.bound.blurred_tex},
      coord,
      0
    );

    var aux = textureLoad(
      ${ioLayout.bound.aux_tex},
      coord,
      0
    );

    (*result)[0] = vec4f(
      ${convertRgbToY}(blurred.rgb),
      aux.z, // albedo luminance
      aux.x, // normal.x
      aux.y, // normal.y
    );
    (*result)[1] = vec4f(
      aux.w, // emission luminance
      0,     // zero padding
      0,     // zero padding
      0,     // zero padding
    );
  }
  else {
    for (var i: u32 = 0; i < ${inChannelsSlot} / 4; i++) {
      let index =
        (coord.y * u32(canvasSize.x) +
        coord.x) * ${inChannelsSlot}/4 +
        i;
      
      (*result)[i] = ${ioLayout.bound.input_buffer}[index];
    }
  }
}`.$name('sample_global');

const applyReLU = wgsl.fn`
  (result: ptr<function, array<f32, ${outChannelsSlot}>>) {
    for (var i = 0u; i < ${outChannelsSlot}; i++) {
      (*result)[i] = max(0, (*result)[i]);
    }
  }
`.$name('apply_relu');

const menderConvolveFn = convolveFn({
  inChannels: inChannelsSlot,
  outChannels: outChannelsSlot,
  kernelRadius: kernelRadiusSlot,
  sampleFiller: (x: Wgsl, y: Wgsl, outSamplePtr: Wgsl) =>
    wgsl`${sampleGlobal}(${x}, ${y}, ${outSamplePtr});`,
  kernelReader: (idx: Wgsl) => wgsl`${weights}[${idx}]`,
});

const entryComputeFn = tgpu
  .computeFn([], { workgroupSize: [BLOCK_SIZE, BLOCK_SIZE] })
  .does(/* wgsl */ `(@builtin(global_invocation_id) gid: vec3u) {
    var result: array<f32, OUT_CHANNELS>;
    for (var i = 0; i < OUT_CHANNELS; i += 1) {
      result[i] = biases[i];
    }

    menderConvolveFn(gid.xy, &result);
  
    if (reluSlot) {
      applyReLU(&result);
    }

    let canvasSize = getViewportSizeSlot();
  
    let output_buffer_begin =
      (gid.y * u32(canvasSize.x) +
      gid.x) * OUT_CHANNELS;
  
    for (var i: u32 = 0; i < OUT_CHANNELS; i++) {
      output_buffer[output_buffer_begin + i] = result[i];
    }
  }`)
  .$uses({
    menderConvolveFn,
    reluSlot,
    applyReLU,
    biases,
    getViewportSizeSlot,
    output_buffer: ioLayout.bound.output_buffer,
    OUT_CHANNELS: outChannelsSlot,
  });

export const MenderStep = ({ root, gBuffer, targetTexture }: Options) => {
  // Resource locators

  const viewportSizeBuffer = root.createBuffer(d.vec2f).$usage('uniform');

  const myGetViewportSize = tgpu
    .fn([], d.vec2f)
    .does(`() -> vec2f {
      return viewportSize;
    }`)
    .$uses({ viewportSize: asUniform(viewportSizeBuffer) });

  // Textures

  const firstWorkBuffer = root
    .createBuffer(
      d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1] * FIRST_DEPTH),
    )
    .$name('First Work Buffer')
    .$usage('storage');

  const secondWorkBuffer = root
    .createBuffer(
      d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1] * SECOND_DEPTH),
    )
    .$name('Second Work Buffer')
    .$usage('storage');

  const mendedResultBuffer = root
    .createBuffer(d.arrayOf(d.f32, gBuffer.size[0] * gBuffer.size[1]))
    .$name('Mender Result Buffer')
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
        .with(getViewportSizeSlot, myGetViewportSize)
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
    ioLayout.populate({
      blurred_tex: gBuffer.upscaledView,
      aux_tex: gBuffer.auxView,
      output_buffer: firstWorkBuffer,
      // biome-ignore lint/suspicious/noExplicitAny: <its fine>
      input_buffer: secondWorkBuffer as any, // <- UNUSED
    }),
    ioLayout.populate({
      // biome-ignore lint/suspicious/noExplicitAny: <its fine>
      input_buffer: firstWorkBuffer as any,
      output_buffer: secondWorkBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
    ioLayout.populate({
      // biome-ignore lint/suspicious/noExplicitAny: <its fine>
      input_buffer: secondWorkBuffer as any,
      output_buffer: mendedResultBuffer,
      blurred_tex: gBuffer.upscaledView, // <- UNUSED
      aux_tex: gBuffer.auxView, // <- UNUSED
    }),
  ];

  // ---
  // Combination pass
  // ---

  const combinationPipeline = root
    .with(getViewportSizeSlot, myGetViewportSize)
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(combinationEntryFn, { format: 'rgba8unorm' })
    .createPipeline();

  const combinationBindGroup = combinationLayout.populate({
    blurredTexture: gBuffer.upscaledView,
    mendedBuffer: mendedResultBuffer,
  });

  viewportSizeBuffer.write(d.vec2f(...gBuffer.size));

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
