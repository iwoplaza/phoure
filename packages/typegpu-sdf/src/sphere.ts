import { tgpu } from 'typegpu';
import { f32, vec2f, vec3f } from 'typegpu/data';
import { distance } from 'typegpu/std';

export const sphere = tgpu.fn(
  [vec3f, vec3f, f32],
  f32,
)((pos, origin, radius) => {
  'use gpu';
  return distance(pos, origin) - radius;
});

export const circle = tgpu.fn(
  [vec2f, vec2f, f32],
  f32,
)((pos, origin, radius) => {
  'use gpu';
  return distance(pos, origin) - radius;
});
