import tgpu from 'typegpu';
import { builtin, vec2f } from 'typegpu/data';

export const fullScreenQuadVertexFn = tgpu['~unstable']
  .vertexFn({
    in: { idx: builtin.vertexIndex },
    out: { pos: builtin.position, uv: vec2f },
  })
  .does(/* wgsl */ `(input: VertexInput) -> VertexOutput {
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

    var output: VertexOutput;
    output.pos = vec4f(SCREEN_RECT[input.idx], 0.0, 1.0);
    output.uv = UVS[input.idx];
    return output;
  }`);
