import { f32, vec2f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { abs, length, max, min } from 'typegpu/std';

export const extrude = tgpu.fn([f32, f32, f32], f32).does((dxy, dz, h) => {
  const w = vec2f(dxy, abs(dz) - h);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0, 0)));
});
