import { tgpu, d, std } from 'typegpu';

export const repeatXYZ = tgpu
  .fn(
    [d.vec3f, d.vec3f],
    d.vec3f,
  )((pos, tileSize) => {
    'use gpu';
    return std.round(pos / tileSize) * tileSize;
  })
  .$name('op_repeat_xyz');

export const repeatXZ = tgpu
  .fn(
    [d.vec3f, d.vec2f],
    d.vec3f,
  )((pos, tileSize) => {
    'use gpu';
    const chunk = std.round(pos.xz / tileSize) * tileSize;
    return d.vec3f(chunk.x, 0, chunk.y);
  })
  .$name('op_repeat_xz');
