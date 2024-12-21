import { f32, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';

export const sphere = tgpu.fn([vec3f, vec3f, f32]).does(`(pos: vec3f, origin: vec3f, radius: f32) -> f32 {
  return distance(pos, origin) - radius;
}`);

export const circle = tgpu.fn([vec3f, vec3f, f32]).does(`(pos: vec2f, origin: vec2f, radius: f32) -> f32 {
  return distance(pos, origin) - radius;
}`);
