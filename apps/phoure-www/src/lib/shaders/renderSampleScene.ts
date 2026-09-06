import { tgpu, d, std } from 'typegpu';
import { convertRgbToY } from '#src/lib-phoure/color.ts';

export const sampleSceneLayout = tgpu.bindGroupLayout({
  uniforms: {
    uniform: d.struct({ modelMatrix: d.mat4x4f, normalModelMatrix: d.mat4x4f }),
  },
  projection: { uniform: d.struct({ projectionMatrix: d.mat4x4f }) },
  camera: {
    uniform: d.struct({ viewMatrix: d.mat4x4f, normalViewMatrix: d.mat3x3f }),
  },
});

const varyings = {
  worldNormal: d.vec3f,
  fragNormal: d.vec3f,
  fragUV: d.vec2f,
  fragDepth: d.f32,
};

export const sampleSceneVertex = tgpu.vertexFn({
  in: { position: d.vec3f, normal: d.vec3f, uv: d.vec2f },
  out: { position: d.builtin.position, ...varyings },
})((input) => {
  'use gpu';
  const worldPosition = (
    sampleSceneLayout.$.uniforms.modelMatrix * d.vec4f(input.position, 1)
  ).xyz;
  const viewPosition =
    sampleSceneLayout.$.camera.viewMatrix * d.vec4f(worldPosition, 1);
  const worldNormal = (
    sampleSceneLayout.$.uniforms.normalModelMatrix * d.vec4f(input.normal, 1)
  ).xyz;
  return {
    position: sampleSceneLayout.$.projection.projectionMatrix * viewPosition,
    worldNormal,
    fragNormal: sampleSceneLayout.$.camera.normalViewMatrix * worldNormal,
    fragUV: input.uv,
    fragDepth: -viewPosition.z * 0.05,
  };
});

export const checkerboardAlbedo = (uv: d.v2f) => {
  'use gpu';
  const cell = std.floor(30 * uv);
  const c =
    0.2 + 0.5 * (cell.x + cell.y - 2 * std.floor((cell.x + cell.y) / 2));
  return d.vec3f(c);
};

export const sampleSceneFragment = tgpu.fragmentFn({
  in: varyings,
  out: { color: d.vec4f },
})((input) => {
  'use gpu';
  const albedo = checkerboardAlbedo(input.fragUV);
  const attenuation = std.max(
    0,
    std.dot(std.normalize(input.worldNormal), d.vec3f(0, 1, 0)),
  );
  return {
    color: d.vec4f(
      albedo * (d.vec3f(1, 0.2, 0.1) * attenuation + d.vec3f(0.3, 0.3, 0.4)),
      1,
    ),
  };
});

export const sampleSceneAux = tgpu.fragmentFn({
  in: varyings,
  out: { aux: d.vec4f },
})((input) => {
  'use gpu';
  return {
    aux: d.vec4f(
      std.normalize(input.fragNormal).xy,
      convertRgbToY(checkerboardAlbedo(input.fragUV)),
      0,
    ),
  };
});
