import { f32 } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { abs, max, min } from 'typegpu/std';

/**
 * polynomial smooth min 2
 * Source: https://iquilezles.org/articles/smin/
 */
export const smin = tgpu.fn([f32, f32, f32], f32).does((a, b, k) => {
  const h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * (1.0 / 4.0);
});
