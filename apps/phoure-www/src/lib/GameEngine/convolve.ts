import * as d from 'typegpu/data';
import tgpu, { type Eventual, type TgpuFn } from 'typegpu';
import { dot } from 'typegpu/std';

/** Has to be divisible by 4 */
export const inChannelsSlot = tgpu.slot<number>();
export const outChannelsSlot = tgpu.slot<number>();
export const kernelRadiusSlot = tgpu.slot<number>();

export const inChannelsQuarter = tgpu['~unstable'].derived(() => {
  if (inChannelsSlot.$ % 4 !== 0) {
    throw new Error(`'inChannels' has to be divisible by 4`);
  }
  return inChannelsSlot.$ / 4;
});

const SampleArray = tgpu['~unstable'].derived(() =>
  d.arrayOf(d.vec4f, inChannelsQuarter.$),
);

const ResultArray = tgpu['~unstable'].derived(() =>
  d.arrayOf(d.f32, outChannelsSlot.$),
);

export type SampleFiller = TgpuFn<
  (
    x: d.I32,
    y: d.I32,
    outSamples: d.Ptr<'function', typeof SampleArray.$, 'read-write'>,
  ) => d.Void
>;
export type KernelReader = TgpuFn<(idx: d.U32) => d.Vec4f>;

const fillSampleSlot = tgpu.slot<SampleFiller>();
const readKernelSlot = tgpu.slot<KernelReader>();

const _convolveFn = tgpu['~unstable'].derived(() => {
  return tgpu.fn([d.vec2u, d.ptrFn(ResultArray.$)])((coord, outResult) => {
    const sample = SampleArray.$();
    const kernelRadius = d.i32(kernelRadiusSlot.$);
    const kernelRadiusU = d.u32(kernelRadiusSlot.$);

    let coord_idx = d.u32(0);
    for (let i = -kernelRadius; i <= kernelRadius; i++) {
      for (let j = -kernelRadius; j <= kernelRadius; j++) {
        fillSampleSlot.$(d.i32(coord.x) + i, d.i32(coord.y) + j, sample);

        for (let out_c = d.u32(0); out_c < outChannelsSlot.$; out_c++) {
          let weight_idx =
            (coord_idx +
              out_c * (2 * kernelRadiusU + 1) * (2 * kernelRadiusU + 1)) *
            inChannelsQuarter.$;

          for (let in_c = d.u32(0); in_c < inChannelsQuarter.$; in_c++) {
            outResult[out_c] += dot(sample[in_c], readKernelSlot.$(weight_idx));
            weight_idx++;
          }
        }

        coord_idx++;
      }
    }
  });
});

export const convolveFn = ({
  sampleFiller,
  kernelReader,
}: {
  sampleFiller: Eventual<SampleFiller>;
  kernelReader: Eventual<KernelReader>;
}) =>
  _convolveFn
    .with(fillSampleSlot, sampleFiller)
    .with(readKernelSlot, kernelReader);
