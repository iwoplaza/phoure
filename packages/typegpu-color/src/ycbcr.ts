import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { dot } from 'typegpu/std';

const rgbToYFactor = tgpu['~unstable'].const(
  d.vec3f,
  d.vec3f(0.2538745098, 0.5061058824, 0.09829019608),
);

export const convertRgbToY = tgpu['~unstable']
  .fn(
    [d.vec3f],
    d.f32,
  )((rgb) => 0.06274509804 + dot(rgb, rgbToYFactor.value))
  .$name('convert_rgb_to_y');

export const rgbToYcbcrMatrix = tgpu['~unstable'].const(
  d.mat3x3f,
  d.mat3x3f(
    d.vec3f(0.299, 0.587, 0.114),
    d.vec3f(-0.168736, -0.331264, 0.5),
    d.vec3f(0.5, -0.418688, -0.081312),
  ),
);

export const ycbcrToRgbMatrix = tgpu['~unstable'].const(
  d.mat3x3f,
  d.mat3x3f(
    d.vec3f(1.0, 0, 1.402),
    d.vec3f(1.0, -0.344136, -0.714136),
    d.vec3f(1.0, 1.772, 0),
  ),
);
