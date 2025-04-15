import tgpu from 'typegpu';
import { f32, vec2f, vec3f } from 'typegpu/data';
import { distance } from 'typegpu/std';

export const sphere = tgpu['~unstable'].fn(
  [vec3f, vec3f, f32],
  f32,
)((pos, origin, radius) => {
  return distance(pos, origin) - radius;
});

export const circle = tgpu['~unstable'].fn(
  [vec2f, vec2f, f32],
  f32,
)((pos, origin, radius) => {
  return distance(pos, origin) - radius;
});
