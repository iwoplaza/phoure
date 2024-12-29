import { f32, struct, vec3f } from 'typegpu/data';

export const ShapeContext = struct({
  rayPos: vec3f,
  rayDir: vec3f,
  rayDistance: f32,
});
