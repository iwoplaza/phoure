import { tgpu } from 'typegpu';
import { f32, vec2f, vec3f } from 'typegpu/data';
import { clamp, dot, length, mul, sub } from 'typegpu/std';

export const lineSegment2 = tgpu.fn(
  [vec2f, vec2f, vec2f],
  f32,
)((p, a, b) => {
  'use gpu';
  const pa = sub(p, a);
  const ba = sub(b, a);
  const h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1);
  return length(sub(pa, mul(h, ba)));
});

export const lineSegment3 = tgpu.fn(
  [vec3f, vec3f, vec3f],
  f32,
)((p, a, b) => {
  'use gpu';
  const pa = sub(p, a);
  const ba = sub(b, a);
  const h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1);
  return length(sub(pa, mul(h, ba)));
});
