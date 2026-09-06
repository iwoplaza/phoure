import { tgpu, d } from 'typegpu';
import { checkerboardAlbedo } from './renderSampleScene';

export const writeGBufferVertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex },
  out: { position: d.builtin.position, rayDir: d.vec3f },
})((input) => {
  'use gpu';
  const positions = d.arrayOf(
    d.vec2f,
    6,
  )([
    d.vec2f(-1, -1),
    d.vec2f(1, -1),
    d.vec2f(-1, 1),
    d.vec2f(1, -1),
    d.vec2f(-1, 1),
    d.vec2f(1, 1),
  ]);
  const worldPosition = d.vec3f(positions[input.vertexIndex], 0);
  return { position: d.vec4f(worldPosition, 1), rayDir: worldPosition };
});

export const writeGBufferFragment = tgpu.fragmentFn({
  in: { rayDir: d.vec3f },
  out: { albedo: d.vec4f, normal: d.vec4f },
})((input) => {
  'use gpu';
  return {
    normal: d.vec4f(input.rayDir, 1),
    albedo: d.vec4f(checkerboardAlbedo(input.rayDir.xy), 1),
  };
});
