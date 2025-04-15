import tgpu from 'typegpu';
import { vec2f, vec3f } from 'typegpu/data';

export const repeatXYZ = tgpu['~unstable']
  .fn(
    [vec3f, vec3f],
    vec3f,
  )(`(pos: vec3f, tile_size: vec3f) -> vec3f {
    return round(pos / tile_size) * tile_size;
  }`)
  .$name('op_repeat_xyz');

export const repeatXZ = tgpu['~unstable']
  .fn(
    [vec3f, vec2f],
    vec3f,
  )(`(pos: vec3f, tile_size: vec2f) -> vec3f {
    let chunk_pos = round(pos.xz / tile_size) * tile_size;
    return vec3f(chunk_pos.x, 0, chunk_pos.y);
  }`)
  .$name('op_repeat_xz');
