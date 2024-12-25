import { vec3f } from 'typegpu/data';
import tgpu from 'typegpu/experimental';
import { mul, normalize } from 'typegpu/std';

import { MarchParams } from './marchSdf';
import { ShapeContext } from './types';

/**
 * Estimates the normal vector at a point in space, based on the SDF given by `MarchParams.sampleSdf`.
 */
export const estimateNormal = tgpu
  .fn([vec3f, ShapeContext], vec3f)
  .does((point, ctx) => {
    /** Arbitrary - should be smaller than any surface detail in your distance function, but not so small as to get lost in float precision */
    const epsilon = MarchParams.getSurfaceThreshold.value(ctx) * 0.5;
    const offX = vec3f(point.x + epsilon, point.y, point.z);
    const offY = vec3f(point.x, point.y + epsilon, point.z);
    const offZ = vec3f(point.x, point.y, point.z + epsilon);

    const centerDistance = MarchParams.sampleSdf.value(point);
    const xDistance = MarchParams.sampleSdf.value(offX);
    const yDistance = MarchParams.sampleSdf.value(offY);
    const zDistance = MarchParams.sampleSdf.value(offZ);

    return normalize(
      mul(
        1 / epsilon,
        vec3f(
          xDistance - centerDistance,
          yDistance - centerDistance,
          zDistance - centerDistance,
        ),
      ),
    );
  });
