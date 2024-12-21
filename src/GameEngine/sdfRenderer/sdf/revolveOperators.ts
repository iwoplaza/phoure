import { f32, vec2f, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { length } from 'typegpu/std';

/**
 * @returns 2d coordinates
 */
export const revolveX = tgpu
  .fn([vec3f, f32], vec2f)
  .does((p, offset) => vec2f(p.x, length(p.yz) - offset));

/**
 * @returns 2d coordinates
 */
export const revolveY = tgpu
  .fn([vec3f, f32], vec2f)
  .does((p, offset) => vec2f(length(p.xz) - offset, p.y));

/**
 * @returns 2d coordinates
 */
export const revolveZ = tgpu
  .fn([vec3f, f32], vec2f)
  .does((p, offset) => vec2f(p.z, length(p.xy) - offset));
