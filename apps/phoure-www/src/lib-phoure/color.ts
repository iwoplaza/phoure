import { d, tgpu } from 'typegpu';
import { dot } from 'typegpu/std';

// Preserve the luminance coefficients used by the trained upscaler.
const rgbToYFactor = tgpu.const(
  d.vec3f,
  d.vec3f(0.2538745098, 0.5061058824, 0.09829019608),
);

export const convertRgbToY = tgpu
  .fn(
    [d.vec3f],
    d.f32,
  )((rgb) => {
    'use gpu';
    return 0.06274509804 + dot(rgb, rgbToYFactor.$);
  })
  .$name('convert_rgb_to_y');

export const ycbcrToRgbMatrix = tgpu.const(
  d.mat3x3f,
  d.mat3x3f(
    d.vec3f(1.0, 0, 1.402),
    d.vec3f(1.0, -0.344136, -0.714136),
    d.vec3f(1.0, 1.772, 0),
  ),
);
