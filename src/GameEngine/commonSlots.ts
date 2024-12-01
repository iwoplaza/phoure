import { type TgpuFn, wgsl } from 'typegpu/experimental';
import type * as d from 'typegpu/data';

export const getViewportSizeSlot = wgsl
  .slot<TgpuFn<[], d.Vec2f>>()
  .$name('getViewportSizeSlot');
