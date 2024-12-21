import tgpu, { type TgpuFn } from 'typegpu/experimental';
import type * as d from 'typegpu/data';

export const getViewportSizeSlot = tgpu
  .slot<TgpuFn<[], d.Vec2f>>()
  .$name('getViewportSizeSlot');
