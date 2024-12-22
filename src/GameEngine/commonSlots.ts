import tgpu from 'typegpu/experimental';
import { vec2f } from 'typegpu/data';

export const getViewportSize = tgpu.accessor(vec2f).$name('getViewportSize');
