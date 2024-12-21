import { f32 } from 'typegpu/data';
import tgpu from 'typegpu/experimental';

/**
 * Inflates the passed in field, and makes it rounded as
 * a side-effect.

 * @returns 3d sdf
 */
export const inflate = tgpu
  .fn([f32, f32], f32)
  .does((d, r) => d - r)
  .$name('op_inflate');
