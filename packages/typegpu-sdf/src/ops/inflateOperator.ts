import { tgpu } from 'typegpu';
import { f32 } from 'typegpu/data';

/**
 * Inflates the passed in field, and makes it rounded as
 * a side-effect.

 * @returns 3d sdf
 */
export const inflate = tgpu
  .fn(
    [f32, f32],
    f32,
  )((d, r) => {
    'use gpu';
    return d - r;
  })
  .$name('op_inflate');
