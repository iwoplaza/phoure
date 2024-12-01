import tgpu, { builtin } from 'typegpu/experimental';
import * as d from 'typegpu/data';

export const fullScreenQuadVertexFn = tgpu
  .vertexFn(
    { idx: builtin.vertexIndex },
    { pos: builtin.position, uv: d.vec2f },
  )
  .does(/* wgsl */ `(@builtin(vertex_index) idx: u32) -> VertexOutput {
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
    output.pos = vec4f(SCREEN_RECT[idx], 0.0, 1.0);
    output.uv = UVS[idx];
    return output;
  }`)
  .$uses({
    get VertexOutput() {
      return fullScreenQuadVertexFn.Output;
    },
  });
