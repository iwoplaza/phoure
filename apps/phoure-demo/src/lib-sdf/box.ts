import { f32, vec2f, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { abs, sub, length, max, min } from 'typegpu/std';

export const box2 = tgpu.fn([vec2f, vec2f], f32).does((p, b) => {
  const d = sub(abs(p), b);
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
});

export const box3 = tgpu.fn([vec3f, vec3f], f32).does((p, b) => {
  const q = sub(abs(p), b);
  return length(max(q, vec3f(0, 0, 0))) + min(max(q.x, max(q.y, q.z)), 0.0);
});
