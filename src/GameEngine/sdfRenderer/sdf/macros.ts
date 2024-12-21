import tgpu from 'typegpu/experimental';
import { f32, vec3f } from 'typegpu/data';
import { dot, clamp } from 'typegpu/std';

export const dd = tgpu.fn([vec3f]).does((value) => {
  return dot(value, value);
});

export const clamp01 = tgpu.fn([f32], f32).does((value) => {
  return clamp(value, 0, 1);
});

// export const union = (values: Wgsl[]) => {
//   if (values.length === 1) {
//     return values[0];
//   }

//   return wgsl`${values.map((v, idx) =>
//     idx < values.length - 1 ? wgsl`min(${v}, ` : wgsl`(${v}`,
//   )}${values.map(() => ')')}`;
// };
