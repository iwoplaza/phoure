import { struct, vec3f, f32 } from 'typegpu/data';

export const ShapeContext = struct({
  rayPos: vec3f,
  rayDir: vec3f,
  rayDistance: f32,
});
