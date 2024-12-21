import tgpu from 'typegpu/experimental';
import { f32, vec2f, vec3f } from 'typegpu/data';
import { clamp01 } from './macros';

export const lineSegment2 = tgpu
  .fn([vec2f, vec2f, vec2f], f32)
  .does(`(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp01(dot(pa, ba) / dot(ba,ba));
    return length(pa - ba*h);
  }`)
  .$uses({ clamp01 });

export const lineSegment3 = tgpu
  .fn([vec3f, vec3f, vec3f], f32)
  .does(`(p: vec3f, a: vec3f, b: vec3f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp01(dot(pa, ba) / dot(ba,ba));
    return length(pa - ba*h);
  }`)
  .$uses({ clamp01 });
