import { f32, mat3x3f, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';

export const convertRgbToY = tgpu
  .fn([vec3f], f32)
  .does(
    (rgb) =>
      16 / 255 + (64.738 * rgb.x + 129.057 * rgb.y + 25.064 * rgb.z) / 255,
  )
  .$name('convert_rgb_to_y');

export const rgbToYcbcrMatrix = tgpu.const(
  mat3x3f,
  mat3x3f(
    vec3f(0.299, 0.587, 0.114),
    vec3f(-0.168736, -0.331264, 0.5),
    vec3f(0.5, -0.418688, -0.081312),
  ),
);

export const ycbcrToRgbMatrix = tgpu.const(
  mat3x3f,
  mat3x3f(
    vec3f(1.0, 0, 1.402),
    vec3f(1.0, -0.344136, -0.714136),
    vec3f(1.0, 1.772, 0),
  ),
);
