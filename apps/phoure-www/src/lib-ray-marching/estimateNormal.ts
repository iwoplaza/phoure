import tgpu from 'typegpu';
import { vec2f, vec3f } from 'typegpu/data';
import { normalize } from 'typegpu/std';

import { MarchParams } from './marchSdf.ts';

// Doing it in a derived until WGSL generation can properly decide when to `let` and when to `var`
const epsilon = tgpu['~unstable'].derived(() =>
  // Arbitrary - should be smaller than any surface detail in your distance function, but not so small as to get lost in float precision
  vec2f(MarchParams.surfaceThreshold.$ * 0.5, 0),
);

/**
 * Estimates the normal vector at a point in space, based on the SDF given by `MarchParams.sampleSdf`.
 */
export const estimateNormal = tgpu.fn(
  [vec3f],
  vec3f,
)((point) => {
  const centerDistance = MarchParams.sampleSdf.$(point);
  const distance = vec3f(
    MarchParams.sampleSdf.$(point.add(epsilon.$.xyy)),
    MarchParams.sampleSdf.$(point.add(epsilon.$.yxy)),
    MarchParams.sampleSdf.$(point.add(epsilon.$.yyx)),
  );

  return normalize(distance.sub(centerDistance));
});
