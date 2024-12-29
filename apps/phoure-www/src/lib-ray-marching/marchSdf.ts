import { f32, struct, u32, vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { ShapeContext } from './types';

const sampleSdfShell = tgpu.fn([vec3f], f32);
type SampleSdf = ReturnType<(typeof sampleSdfShell)['does']>;

const defaultGetSurfaceThreshold = tgpu
  .fn([ShapeContext], f32)
  .does((_ctx) => 0.001);

export const MarchParams = {
  maxSteps: tgpu.slot(500),

  /**
   * The distance from the camera at which the sky should be drawn instead of the world.
   */
  farPlane: tgpu.slot(100),

  getSurfaceThreshold: tgpu.slot(defaultGetSurfaceThreshold),
  sampleSdf: tgpu.slot<SampleSdf>(),
};

export const MarchResult = struct({
  steps: u32,
  position: vec3f,
});

export const march = tgpu
  .fn([/* TODO: ptr */ ShapeContext, u32, /* TODO: ptr */ MarchResult])
  .does(`(ctx: ptr<function, ShapeContext>, limit: u32, out: ptr<function, MarchResult>) {
    var pos = (*ctx).rayPos;
    var prev_dist = -1.;
    var min_dist: f32 = FAR_PLANE;

    var step = 0u;
    var progress = 0.;

    for (; step <= limit; step++) {
      pos = (*ctx).rayPos + (*ctx).rayDir * progress;
      min_dist = sampleSdf(pos);

      // Inside volume?
      if (min_dist <= 0.) {
        // No need to check more objects.
        break;
      }

      if (min_dist < getSurfaceThreshold(*ctx) && min_dist < prev_dist) {
        // No need to check more objects.
        break;
      }

      // march forward safely
      progress += min_dist;
      (*ctx).rayDistance += min_dist;

      if (progress > FAR_PLANE) {
        // Stop checking.
        break;
      }

      prev_dist = min_dist;
    }

    (*out).position = pos;

    // Not near surface or distance rising?
    if (min_dist > getSurfaceThreshold(*ctx) * 2. || min_dist > prev_dist) {
      // Sky
      (*out).steps = MAX_STEPS + 1u;
      return;
    }

    (*out).steps = step;
  }`)
  .$uses({
    ShapeContext,
    MarchResult,
    FAR_PLANE: MarchParams.farPlane,
    MAX_STEPS: MarchParams.maxSteps,
    sampleSdf: MarchParams.sampleSdf,
    getSurfaceThreshold: MarchParams.getSurfaceThreshold,
  });
