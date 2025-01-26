import tgpu from 'typegpu';
import { f32, mat3x3f, vec3f } from 'typegpu/data';
import { dot } from 'typegpu/std';

const rgbToYFactor = tgpu['~unstable'].const(
  vec3f,
  vec3f(0.2538745098, 0.5061058824, 0.09829019608),
);

export const convertRgbToY = tgpu['~unstable']
  .fn([vec3f], f32)
  .does((rgb) => 0.06274509804 + dot(rgb, rgbToYFactor.value))
  .$name('convert_rgb_to_y');

export const rgbToYcbcrMatrix = tgpu['~unstable'].const(
  mat3x3f,
  mat3x3f(
    vec3f(0.299, 0.587, 0.114),
    vec3f(-0.168736, -0.331264, 0.5),
    vec3f(0.5, -0.418688, -0.081312),
  ),
);

export const ycbcrToRgbMatrix = tgpu['~unstable'].const(
  mat3x3f,
  mat3x3f(
    vec3f(1.0, 0, 1.402),
    vec3f(1.0, -0.344136, -0.714136),
    vec3f(1.0, 1.772, 0),
  ),
);
