import { vec2f } from 'typegpu/data';
import tgpu from 'typegpu';

export const accessViewportSize = tgpu['~unstable']
  .accessor(vec2f)
  .$name('accessViewportSize');
