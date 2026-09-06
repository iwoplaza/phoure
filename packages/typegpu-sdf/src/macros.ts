import { tgpu } from 'typegpu';
import { f32, vec3f } from 'typegpu/data';
import { dot } from 'typegpu/std';

export const dd = tgpu.fn(
  [vec3f],
  f32,
)((value) => {
  'use gpu';
  return dot(value, value);
});
