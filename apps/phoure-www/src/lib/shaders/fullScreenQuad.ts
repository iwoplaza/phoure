import { tgpu } from 'typegpu';
import { builtin, vec2f, vec4f } from 'typegpu/data';

export const fullScreenTriangle = tgpu.vertexFn({
  in: { vertexIndex: builtin.vertexIndex },
  out: { pos: builtin.position, uv: vec2f },
})((input) => {
  'use gpu';
  const pos = [vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)];
  const uv = [vec2f(0, 0), vec2f(2, 0), vec2f(0, 2)];

  return {
    pos: vec4f(pos[input.vertexIndex], 0, 1),
    uv: uv[input.vertexIndex],
  };
});
