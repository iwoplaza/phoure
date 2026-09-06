import { d, tgpu } from 'typegpu';
import { abs, floor, min, mix, mul, pow } from 'typegpu/std';
import { sdSphere } from '@typegpu/sdf';
import { MarchParams, ShapeContext } from '#src/lib-ray-marching/index.ts';

export const Material = d.struct({
  albedo: d.vec3f,
  roughness: d.f32,
  emissive: d.bool,
});

// const timeAccess = tgpu.accessor(d.f32);

const sdfShell = tgpu.fn([d.vec3f], d.f32);

const objLeftBlob = sdfShell((pos) => {
  'use gpu';
  return sdSphere(pos.sub(d.vec3f(-0.3, -0.2, 0)), 0.2);
});

// ANIMATED LIGHT
// const objCenterBlob = sdfShell((pos) => {
//   'use gpu';
//   return sdSphere(pos.sub(d.vec3f(-0.3, 0.7 + sin(timeAccess.$ * 0.001) * 0.4, -2.)), 0.2);
// });

const objCenterBlob = sdfShell((pos) => {
  'use gpu';
  return sdSphere(pos.sub(d.vec3f(-0.3, 0.4, 0.4)), 0.2);
});

const objRightBlob = sdfShell((pos) => {
  'use gpu';
  return sdSphere(pos.sub(d.vec3f(0.4, 0.2, 0)), 0.4);
});

const objFloor = sdfShell((pos) => {
  'use gpu';
  return pos.y + 0.3;
});

const matFloor = tgpu.fn([d.vec3f, d.ptrFn(Material)])((pos, mtr) => {
  'use gpu';
  const uv = floor(mul(5, pos.xz));
  const c = 0.2 + 0.5 * (uv.x + uv.y - 2.0 * floor((uv.x + uv.y) / 2.0));

  mtr.$.albedo = mix(d.vec3f(1, 1, 1), d.vec3f(0, 0, 0), c);
  mtr.$.roughness = 0.9;
});

export const FAR = 100;

export const worldSdf = sdfShell((pos) => {
  'use gpu';
  let minDist = d.f32(FAR);

  minDist = min(minDist, objLeftBlob(pos));
  minDist = min(minDist, objCenterBlob(pos));
  minDist = min(minDist, objRightBlob(pos));
  minDist = min(minDist, objFloor(pos));

  return minDist;
});

// MATERIALS

export const skyColor = tgpu.fn(
  [d.vec3f],
  d.vec3f,
)((dir) => {
  'use gpu';
  const t = pow(min(abs(dir.y) * 4, 1), 0.4);
  return mix(d.vec3f(0.7, 0.7, 0.75), d.vec3f(0.35, 0.4, 0.6), t);
});

export const worldMat = tgpu.fn([d.vec3f, ShapeContext, d.ptrFn(Material)])((
  pos,
  ctx,
  out,
) => {
  'use gpu';
  const d_left_blob = objLeftBlob(pos);
  const d_center_blob = objCenterBlob(pos);
  const d_right_blob = objRightBlob(pos);
  const d_floor_blob = objFloor(pos);

  // defaults
  out.$.emissive = false;
  out.$.roughness = 1;

  if (d_left_blob <= MarchParams.surfaceThreshold.$) {
    // left blob
    out.$.albedo = d.vec3f(1, 0.5, 0.2);
    out.$.roughness = 0.95;
  } else if (d_center_blob <= MarchParams.surfaceThreshold.$) {
    // test light
    out.$.albedo = mul(20, d.vec3f(1, 1, 0.5));
    out.$.emissive = true;
  } else if (d_right_blob <= MarchParams.surfaceThreshold.$) {
    out.$.albedo = mul(0.9, d.vec3f(0.5, 0.5, 0.6));
    out.$.roughness = 0.1;
  } else if (d_floor_blob <= MarchParams.surfaceThreshold.$) {
    matFloor(pos, out);
  } else {
    out.$.albedo = skyColor(ctx.rayDir);
  }
});
