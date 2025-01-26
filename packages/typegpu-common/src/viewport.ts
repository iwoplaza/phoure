import tgpu from 'typegpu';
import { vec2f } from 'typegpu/data';

export const accessViewportSize = tgpu['~unstable']
  .accessor(vec2f)
  .$name('accessViewportSize');
