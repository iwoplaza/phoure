import tgpu, { type TgpuFn } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { ShapeContext } from './types.ts';

type SampleSdf = TgpuFn<[d.Vec3f], d.F32>;

const defaultGetSurfaceThreshold = tgpu['~unstable'].fn([ShapeContext], d.f32)(
  (_ctx) => 0.001,
);

export const MarchParams = {
  maxSteps: tgpu['~unstable'].slot(500),

  /**
   * The distance from the camera at which the sky should be drawn instead of the world.
   */
  farPlane: tgpu['~unstable'].slot(100),

  getSurfaceThreshold: tgpu['~unstable'].slot(defaultGetSurfaceThreshold),
  sampleSdf: tgpu['~unstable'].slot<SampleSdf>(),
};

export const MarchResult = d.struct({
  steps: d.u32,
  position: d.vec3f,
});

export const march = tgpu['~unstable'].fn([
  d.ptrFn(ShapeContext),
  d.u32,
  d.ptrFn(MarchResult),
])((ctx, limit, out) => {
  let pos = d.vec3f(ctx.rayPos);
  let prev_dist = d.f32(-1);
  let min_dist = d.f32(MarchParams.farPlane.value);

  let step = d.u32(0);
  let progress = d.f32(0);

  for (; step <= limit; step++) {
    pos = std.add(ctx.rayPos, std.mul(ctx.rayDir, progress));
    min_dist = MarchParams.sampleSdf.value(pos);

    // Inside volume?
    if (min_dist <= 0.) {
      // No need to check more objects.
      break;
    }

    if (
      min_dist < MarchParams.getSurfaceThreshold.value(ctx) &&
      min_dist < prev_dist
    ) {
      // No need to check more objects.
      break;
    }

    // march forward safely
    progress += min_dist;
    ctx.rayDistance += min_dist;

    if (progress > MarchParams.farPlane.value) {
      // Stop checking.
      break;
    }

    prev_dist = min_dist;
  }

  out.position = pos;

  // Not near surface or distance rising?
  if (
    min_dist > MarchParams.getSurfaceThreshold.value(ctx) * 2. ||
    min_dist > prev_dist
  ) {
    // Sky
    out.steps = MarchParams.maxSteps.value + 1;
    return;
  }

  out.steps = step;
});
