import tgpu from 'typegpu';
import { builtin, vec2f } from 'typegpu/data';

export const fullScreenQuadVertexFn = tgpu['~unstable'].vertexFn({
  in: { idx: builtin.vertexIndex },
  out: { pos: builtin.position, uv: vec2f },
})(/* wgsl */ `{
  const SCREEN_RECT = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),

    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, 1.0),
  );

  const UVS = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),

    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
  );

  var output: Out;
  output.pos = vec4f(SCREEN_RECT[in.idx], 0.0, 1.0);
  output.uv = UVS[in.idx];
  return output;
}`);
