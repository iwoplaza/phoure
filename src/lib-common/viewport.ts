import tgpu from 'typegpu/experimental';
import { vec2f } from 'typegpu/data';

export const accessViewportSize = tgpu
  .accessor(vec2f)
  .$name('accessViewportSize');
