import {
  type I32,
  type U32,
  type Vec4f,
  type WgslArray,
  type PtrFn,
  arrayOf,
  f32,
  ptrFn,
  vec2u,
} from 'typegpu/data';
import tgpu, { type Eventual, type TgpuFn } from 'typegpu';

export type SampleFiller = TgpuFn<
  [x: I32, y: I32, outSamplerPtr: PtrFn<WgslArray<Vec4f>>]
>;
export type KernelReader = TgpuFn<[idx: U32], Vec4f>;

/**
 * Has to be divisible by 4
 */
export const inChannelsSlot = tgpu['~unstable']
  .slot<number>()
  .$name('in_channels');
export const outChannelsSlot = tgpu['~unstable']
  .slot<number>()
  .$name('out_channels');
export const kernelRadiusSlot = tgpu['~unstable']
  .slot<number>()
  .$name('kernel_radius');
const sampleFillerSlot = tgpu['~unstable']
  .slot<SampleFiller>()
  .$name('sample_filler');
const kernelReaderSlot = tgpu['~unstable']
  .slot<KernelReader>()
  .$name('kernel_reader');

export const inChannelsQuarter = tgpu['~unstable'].derived(() => {
  if (inChannelsSlot.value % 4 !== 0) {
    throw new Error(`'inChannels' has to be divisible by 4`);
  }
  return inChannelsSlot.value / 4;
});

const _convolveFn = tgpu['~unstable'].derived(() => {
  return tgpu['~unstable']
    .fn([vec2u, ptrFn(arrayOf(f32, outChannelsSlot.value))])
    .does(/* wgsl */ `(coord: vec2u, result: ptr<function, array<f32, outChannels>>) {
      var sample = array<vec4f, inChannelsQuarter>();

      var coord_idx: u32 = 0;
      for (var i: i32 = -i32(kernelRadiusSlot); i <= i32(kernelRadiusSlot); i++) {
        for (var j: i32 = -i32(kernelRadiusSlot); j <= i32(kernelRadiusSlot); j++) {
          fillSample(i32(coord.x) + i, i32(coord.y) + j, &sample);

          for (var out_c: u32 = 0; out_c < outChannels; out_c++) {
            var weight_idx = (coord_idx + out_c * (2 * kernelRadiusSlot + 1) * (2 * kernelRadiusSlot + 1)) * inChannelsQuarter;
            for (var in_c: u32 = 0; in_c < inChannelsQuarter; in_c++) {
              (*result)[out_c] += dot(sample[in_c], readKernel(weight_idx));
              weight_idx++;
            }
          }

          coord_idx++;
        }
      }
    }`)
    .$uses({
      inChannelsQuarter,
      outChannels: outChannelsSlot,
      kernelRadiusSlot,
      fillSample: sampleFillerSlot,
      readKernel: kernelReaderSlot,
    })
    .$name('_convolveFn');
});

export const convolveFn = ({
  sampleFiller,
  kernelReader,
}: {
  sampleFiller: Eventual<SampleFiller>;
  kernelReader: Eventual<KernelReader>;
}) =>
  _convolveFn
    .with(sampleFillerSlot, sampleFiller)
    .with(kernelReaderSlot, kernelReader);
