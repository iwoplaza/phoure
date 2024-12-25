import { f32, vec2f, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { clamp, dot, length, mul, sub } from 'typegpu/std';

export const lineSegment2 = tgpu
  .fn([vec2f, vec2f, vec2f], f32)
  .does((p, a, b) => {
    const pa = sub(p, a);
    const ba = sub(b, a);
    const h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1);
    return length(sub(pa, mul(h, ba)));
  });

export const lineSegment3 = tgpu
  .fn([vec3f, vec3f, vec3f], f32)
  .does((p, a, b) => {
    const pa = sub(p, a);
    const ba = sub(b, a);
    const h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1);
    return length(sub(pa, mul(h, ba)));
  });
