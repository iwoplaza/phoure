import * as d from 'typegpu/data';
import tgpu from 'typegpu';
import { min, mul } from 'typegpu/std';
import { sphere } from '@typegpu/sdf';
import { MarchParams, ShapeContext } from 'src/lib-ray-marching';

export const Material = d.struct({
  albedo: d.vec3f,
  roughness: d.f32,
  emissive: d.bool,
});

// const getTime = tgpu.accessor(d.f32);

const sdfShell = tgpu['~unstable'].fn([d.vec3f], d.f32);

const objLeftBlob = sdfShell.does((pos) =>
  sphere(pos, d.vec3f(-0.3, -0.2, 0), 0.2),
);

// ANIMATED LIGHT
// const objCenterBlob = wgsl.fn`(pos: vec3f) -> f32 {
//   return ${sdf.sphere}(pos, vec3(-0.3, 0.7 + sin(${timeUniform} * 0.001) * 0.4, -2.), 0.2);
// }`.$name('obj_center_blob');

const objCenterBlob = sdfShell.does((pos) =>
  sphere(pos, d.vec3f(-0.3, 0.4, 0.4), 0.2),
);

const objRightBlob = sdfShell.does((pos) =>
  sphere(pos, d.vec3f(0.4, 0.2, 0), 0.4),
);

const objFloor = sdfShell.does((pos) => pos.y + 0.3);

const matFloor = tgpu['~unstable']
  .fn([d.vec3f, d.ptrFn(Material)])
  .does(/* wgsl */ `(pos: vec3f, mtr: ptr<function, Material>) {
    let uv = floor(5.0 * pos.xz);
    let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));
    
    (*mtr).albedo = mix(vec3(1., 1., 1.), vec3(0., 0., 0.), c);
    (*mtr).roughness = 0.9;
  }`)
  .$uses({ Material })
  .$name('mat_floor');

export const FAR = 100;

export const worldSdf = sdfShell.does((pos) => {
  let min_dist = d.f32(FAR);

  min_dist = min(min_dist, objLeftBlob(pos));
  min_dist = min(min_dist, objCenterBlob(pos));
  min_dist = min(min_dist, objRightBlob(pos));
  min_dist = min(min_dist, objFloor(pos));

  return min_dist;
});

// MATERIALS

export const skyColor = tgpu['~unstable']
  .fn([d.vec3f], d.vec3f)
  .does(/* wgsl */ `(dir: vec3f) -> vec3f {
    let t = pow(min(abs(dir.y) * 4, 1.), 0.4);
    
    let uv = floor(30.0 * dir.xy);
    let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));

    return mix(
      vec3f(0.7, 0.7, 0.75), // horizon
      vec3f(0.35, 0.4, 0.6),
      t,
    );
  }`);

export const worldMat = tgpu['~unstable']
  .fn([d.vec3f, ShapeContext, d.ptrFn(Material)])
  .does((pos, ctx, out) => {
    const sd = MarchParams.getSurfaceThreshold.value(ctx);
    const d_left_blob = objLeftBlob(pos);
    const d_center_blob = objCenterBlob(pos);
    const d_right_blob = objRightBlob(pos);
    const d_floor_blob = objFloor(pos);

    // defaults
    out.emissive = false;
    out.roughness = 1;

    if (d_left_blob <= sd) {
      // left blob
      out.albedo = d.vec3f(0.2, 0.2, 1);
      out.roughness = 0.95;
    } else if (d_center_blob <= sd) {
      // test light
      out.albedo = mul(20, d.vec3f(1, 1, 0.5));
      out.emissive = true;
    } else if (d_right_blob <= sd) {
      out.albedo = mul(0.9, d.vec3f(0.5, 0.5, 0.6));
      out.roughness = 0.1;
    } else if (d_floor_blob <= sd) {
      matFloor(pos, out);
    } else {
      // out.albedo = vec3f(0.5, 0.5, 0.2);
      out.albedo = skyColor(ctx.rayDir);
    }
  })
  .$name('world_mat');
