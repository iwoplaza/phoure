import { f32, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';

export const convertRgbToY = tgpu
  .fn([vec3f], f32)
  .does(`(rgb: vec3f) -> f32 {
    return 16./255. + (64.738 * rgb.r + 129.057 * rgb.g + 25.064 * rgb.b) / 255.;
  }`)
  .$name('convert_rgb_to_y');
