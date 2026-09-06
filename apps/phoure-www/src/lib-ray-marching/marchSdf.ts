import { d, std, tgpu, type TgpuFn } from 'typegpu';
import { ShapeContext } from './types.ts';

type SampleSdf = TgpuFn<(pos: d.Vec3f) => d.F32>;

export const MarchParams = {
  maxSteps: tgpu.slot(500),

  /**
   * The distance from the camera at which the sky should be drawn instead of the world.
   */
  farPlane: tgpu.slot(100),

  surfaceThreshold: tgpu.slot<number>(0.001),
  sampleSdf: tgpu.slot<SampleSdf>(),
};

export const MarchResult = d.struct({
  steps: d.u32,
  position: d.vec3f,
});

export const march = tgpu.fn([
  d.ptrFn(ShapeContext),
  d.u32,
  d.ptrFn(MarchResult),
])((ctx, limit, out) => {
  'use gpu';
  let pos = d.vec3f(ctx.$.rayPos);
  let prev_dist = d.f32(-1);
  let min_dist = d.f32(MarchParams.farPlane.$);

  let step = d.u32(0);
  let progress = d.f32(0);

  for (; step <= limit; step++) {
    pos = std.add(ctx.$.rayPos, std.mul(ctx.$.rayDir, progress));
    min_dist = MarchParams.sampleSdf.$(pos);

    // Inside volume?
    if (min_dist <= 0) {
      // No need to check more objects.
      break;
    }

    if (min_dist < MarchParams.surfaceThreshold.$ && min_dist < prev_dist) {
      // No need to check more objects.
      break;
    }

    // march forward safely
    progress += min_dist;
    ctx.$.rayDistance += min_dist;

    if (progress > MarchParams.farPlane.$) {
      // Stop checking.
      break;
    }

    prev_dist = min_dist;
  }

  out.$.position = d.vec3f(pos);

  // Not near surface or distance rising?
  if (min_dist > MarchParams.surfaceThreshold.$ * 2 || min_dist > prev_dist) {
    // Sky
    out.$.steps = MarchParams.maxSteps.$ + 1;
    return;
  }

  out.$.steps = step;
});
