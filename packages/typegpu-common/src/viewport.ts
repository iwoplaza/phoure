import { vec2f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';

export const accessViewportSize = tgpu
  .accessor(vec2f)
  .$name('accessViewportSize');
