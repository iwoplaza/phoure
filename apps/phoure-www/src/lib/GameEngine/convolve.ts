import * as d from 'typegpu/data';
import tgpu, { type Eventual, type TgpuFn } from 'typegpu';

export type SampleFiller = TgpuFn<
  (
    x: d.I32,
    y: d.I32,
    outSamplerPtr: d.Ptr<'function', d.WgslArray<d.Vec4f>, 'read-write'>,
  ) => d.Void
>;
export type KernelReader = TgpuFn<(idx: d.U32) => d.Vec4f>;

/** Has to be divisible by 4 */
export const inChannelsSlot = tgpu.slot<number>();
export const outChannelsSlot = tgpu.slot<number>();
export const kernelRadiusSlot = tgpu.slot<number>();
const sampleFillerSlot = tgpu.slot<SampleFiller>();
const kernelReaderSlot = tgpu.slot<KernelReader>();

export const inChannelsQuarter = tgpu['~unstable'].derived(() => {
  if (inChannelsSlot.value % 4 !== 0) {
    throw new Error(`'inChannels' has to be divisible by 4`);
  }
  return inChannelsSlot.value / 4;
});

const _convolveFn = tgpu['~unstable'].derived(() => {
  return tgpu
    .fn([d.vec2u, d.ptrFn(d.arrayOf(d.f32, outChannelsSlot.value))])(
      /* wgsl */ `(coord: vec2u, result: ptr<function, array<f32, outChannels>>) {
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
      }`,
    )
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
