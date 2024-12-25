import { vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { dot } from 'typegpu/std';

export const dd = tgpu.fn([vec3f]).does((value) => {
  return dot(value, value);
});

// export const union = (values: Wgsl[]) => {
//   if (values.length === 1) {
//     return values[0];
//   }

//   return wgsl`${values.map((v, idx) =>
//     idx < values.length - 1 ? wgsl`min(${v}, ` : wgsl`(${v}`,
//   )}${values.map(() => ')')}`;
// };
