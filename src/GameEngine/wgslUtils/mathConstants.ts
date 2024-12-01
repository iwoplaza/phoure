import { wgsl } from 'typegpu/experimental';

export const PI = wgsl.constant(Math.PI);
export const TWO_PI = wgsl.constant(Math.PI * 2);

export const ONES_3F = wgsl.constant('vec3f(1., 1., 1.)');
export const ZEROS_3F = wgsl.constant('vec3f(0., 0., 0.)');
