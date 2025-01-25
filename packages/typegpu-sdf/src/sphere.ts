import { f32, vec2f, vec3f } from 'typegpu/data';
import tgpu from 'typegpu';
import { length, sub } from 'typegpu/std';

export const sphere = tgpu['~unstable']
  .fn([vec3f, vec3f, f32], f32)
  .does((pos, origin, radius) => {
    // TODO: Use 'distance' here, once it exists in `typegpu/std`
    return length(sub(pos, origin)) - radius;
  });

export const circle = tgpu['~unstable']
  .fn([vec2f, vec2f, f32], f32)
  .does((pos, origin, radius) => {
    // TODO: Use 'distance' here, once it exists in `typegpu/std`
    return length(sub(pos, origin)) - radius;
  });
