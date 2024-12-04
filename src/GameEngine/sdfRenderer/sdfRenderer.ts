import { atom } from 'jotai';
import tgpu, {
  type TgpuFn,
  wgsl,
  type ExperimentalTgpuRoot,
  asUniform,
} from 'typegpu/experimental';
import * as d from 'typegpu/data';
import type { GBuffer } from '../../gBuffer';
import {
  Camera,
  CameraStruct,
  constructRayDir,
  constructRayPos,
  getCameraProps,
} from './camera';
import { randOnHemisphere, setupRandomSeed } from '../wgslUtils/random';
import worldSdf, {
  Material,
  ShapeContext,
  skyColor,
  surfaceDist,
  worldMat,
} from './worldSdf';
import { ONES_3F } from '../wgslUtils/mathConstants';
import { MAX_STEPS, MarchResult, march } from './marchSdf';
import { convertRgbToY } from './colorUtils';
import { store } from '@/store';
import { getViewportSizeSlot } from '../commonSlots';

const BlockSize = 8;

// parameters
const OutputFormat = wgsl.slot().$name('output_format');

const SUPER_SAMPLES = 4;
const ONE_OVER_SUPER_SAMPLES = 1 / SUPER_SAMPLES;
const SUB_SAMPLES = 16;
const MAX_REFL = 3;

const getRandomSeedPrimerSlot = wgsl.slot<TgpuFn<[], d.F32>>();
const getAccumulatedLayersSlot = wgsl.slot<TgpuFn<[], d.F32>>();

const Reflection = d.struct({
  color: d.vec3f,
  roughness: d.f32,
});

export const accumulatedLayersAtom = atom(0);

const marchWithSurfaceDist = march(surfaceDist).$name(
  'march_with_surface_dist',
);

/**
 * Reflecting: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
 * @param ray_dir
 * @param normal
 * @param mat_roughness
 */
const reflect = wgsl.fn`(ray_dir: vec3f, normal: vec3f, mat_roughness: f32, out_roughness: ptr<function, f32>) -> vec3f {
  let slope = dot(ray_dir, normal);
  let dn2 = 2. * slope;
  let refl_dir = ray_dir - dn2 * normal;

  let fresnel = 1. - pow(1. + slope, 16.);
  let roughness = mat_roughness * fresnel;
  *out_roughness = roughness;

  var new_ray_dir = ${randOnHemisphere}(normal);
  new_ray_dir = mix(refl_dir, new_ray_dir, roughness);
  return normalize(new_ray_dir);
}`.$name('reflect');

const worldNormals = tgpu
  .fn([d.vec3f, ShapeContext], d.vec3f)
  .does(/* wgsl */ `(point: vec3f, ctx: ShapeContext) -> vec3f {
    let epsilon = surfaceDist(ctx) * 0.5; // arbitrary - should be smaller than any surface detail in your distance function, but not so small as to get lost in float precision
    let offX = vec3f(point.x + epsilon, point.y, point.z);
    let offY = vec3f(point.x, point.y + epsilon, point.z);
    let offZ = vec3f(point.x, point.y, point.z + epsilon);
    
    let centerDistance = worldSdf(point);
    let xDistance = worldSdf(offX);
    let yDistance = worldSdf(offY);
    let zDistance = worldSdf(offZ);

    return normalize(vec3f(
      (xDistance - centerDistance),
      (yDistance - centerDistance),
      (zDistance - centerDistance),
    ) / epsilon);
  }`)
  .$uses({ ShapeContext, surfaceDist, worldSdf });

const renderSubPixel = tgpu
  .fn([d.vec2f], d.vec3f)
  .does(/* wgsl */ `(coord: vec2f) -> vec3f {
    // doing the first march before each sub-sample, since the first march result is the same for all of them

    var init_shape_ctx: ShapeContext;
    init_shape_ctx.ray_pos = constructRayPos();
    init_shape_ctx.ray_dir = constructRayDir(coord);
    init_shape_ctx.ray_distance = 0.;
    var init_march_result: MarchResult;

    marchWithSurfaceDist(&init_shape_ctx, MAX_STEPS, &init_march_result);

    if (init_march_result.steps >= MAX_STEPS) {
      return min(skyColor(init_shape_ctx.ray_dir), ONES_3F);
    }

    let init_normal = worldNormals(init_march_result.position, init_shape_ctx);

    var init_material: Material;
    worldMat(init_march_result.position, init_shape_ctx, &init_material);

    if (init_material.emissive) {
      return min(init_material.albedo, ONES_3F);
    }

    var reflections: array<Reflection, MAX_REFL>;
    
    var acc = vec3f(0., 0., 0.);
    for (var sub = 0u; sub < SUB_SAMPLES; sub++) {
      var material: Material = init_material;
      var normal = init_normal;

      var emissive_color = vec3f(0., 0., 0.);
      var refl_count = 0u;

      var shape_ctx: ShapeContext;
      shape_ctx.ray_pos = init_march_result.position;
      shape_ctx.ray_dir = init_shape_ctx.ray_dir;
      shape_ctx.ray_distance = init_shape_ctx.ray_distance;

      for (var refl = 0u; refl < MAX_REFL; refl++) {
        var roughness: f32 = 0.;
        shape_ctx.ray_dir = reflect(
          shape_ctx.ray_dir,
          normal,
          material.roughness,
          &roughness,
        );
        reflections[refl_count].color = material.albedo;
        reflections[refl_count].roughness = roughness;
        refl_count++;

        var march_result: MarchResult;
        marchWithSurfaceDist(&shape_ctx, MAX_STEPS, &march_result);
        shape_ctx.ray_pos = march_result.position;

        if (march_result.steps >= MAX_STEPS) {
          emissive_color = skyColor(shape_ctx.ray_dir);
          break;
        }

        normal = worldNormals(shape_ctx.ray_pos, shape_ctx);

        worldMat(shape_ctx.ray_pos, shape_ctx, &material);

        if (material.emissive) {
          emissive_color = material.albedo;
          break;
        }
      }

      var sub_acc = emissive_color;
      for (var i = i32(refl_count) - 1; i >= 0; i--) {
        let mat_color = reflections[i].color;
        let reflectivity = 1. - reflections[i].roughness;

        sub_acc *= mix(mat_color, ONES_3F, max(0., min(reflectivity, 1.))); // absorb the ray color based on reflectivity
      }

      acc += sub_acc;
    }

    // averaging
    acc /= f32(SUB_SAMPLES);

    // clipping
    acc = min(acc, ONES_3F);

    return acc;
  }`)
  .$uses({
    SUB_SAMPLES,
    MAX_STEPS,
    MAX_REFL,
    ONES_3F,
    MarchResult,
    Material,
    Reflection,
    ShapeContext,
    constructRayPos,
    constructRayDir,
    worldNormals,
    marchWithSurfaceDist,
    skyColor,
    worldMat,
    reflect,
  });

const mainLayout = tgpu.bindGroupLayout({
  previousRender: { texture: 'unfilterable-float' },
  mainOutput: { storageTexture: 'rgba8unorm', access: 'writeonly' },
});

const mainComputeFn = tgpu
  .computeFn([], { workgroupSize: [BlockSize, BlockSize] })
  .does(/* wgsl */ `(@builtin(global_invocation_id) gid: vec3u) {
    setupRandomSeed(vec2f(gid.xy) * ${Math.random()} + getRandomSeedPrimerSlot() * ${Math.random()});

    let prev_layers = getAccumulatedLayersSlot();
    let prev_render = textureLoad(previousRender, gid.xy, 0);
  
    var acc = vec3f(0., 0., 0.);
    for (var sx = 0u; sx < SUPER_SAMPLES; sx++) {
      for (var sy = 0u; sy < SUPER_SAMPLES; sy++) {
        let offset = vec2f(
          (f32(sx) + 0.5) * ONE_OVER_SUPER_SAMPLES,
          (f32(sy) + 0.5) * ONE_OVER_SUPER_SAMPLES,
        );
  
        acc += renderSubPixel(vec2f(gid.xy) + offset);
      }
    }
  
    acc *= ONE_OVER_SUPER_SAMPLES * ONE_OVER_SUPER_SAMPLES;
  
    // applying gamma correction
    let gamma = 2.2;
    acc = pow(acc, vec3(1.0 / gamma));
  
    var new_render = vec4(acc, 1.0);
    if (prev_layers > 0) {
      new_render = (prev_render * prev_layers + vec4(acc, 1.0)) / (prev_layers + 1);
    }
  
    textureStore(mainOutput, gid.xy, new_render);
  }`)
  .$uses({
    SUPER_SAMPLES,
    ONE_OVER_SUPER_SAMPLES,
    previousRender: mainLayout.bound.previousRender,
    mainOutput: mainLayout.bound.mainOutput,
    setupRandomSeed,
    renderSubPixel,
    getRandomSeedPrimerSlot,
    getAccumulatedLayersSlot,
  });

const auxLayout = tgpu
  .bindGroupLayout({
    auxOutput: { storageTexture: 'rgba16float' },
  })
  .$name('SDF Renderer: Aux Bind Group Layout');

const auxComputeFn = tgpu
  .computeFn([], { workgroupSize: [BlockSize, BlockSize] })
  .does(/* wgsl */ `(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let offset = vec2f(
      0.5,
      0.5,
    );
    
    var march_result: MarchResult;
    var shape_ctx: ShapeContext;
    shape_ctx.ray_pos = constructRayPos();
    shape_ctx.ray_dir = constructRayDir(
      vec2f(GlobalInvocationID.xy) + offset
    );
    shape_ctx.ray_distance = 0.;
  
    marchWithSurfaceDist(&shape_ctx, MAX_STEPS, &march_result);
  
    var world_normal: vec3f;
  
    if (march_result.steps >= MAX_STEPS) {
      world_normal = -shape_ctx.ray_dir;
    }
    else {
      world_normal = worldNormals(march_result.position, shape_ctx);
    }
  
    var material: Material;
    worldMat(march_result.position, shape_ctx, &material);
  
    let white = vec3f(1., 1., 1.);
    let mat_color = min(material.albedo, white);
  
    var albedo_luminance = convertRgbToY(mat_color);
    var emission_luminance = 0.;
    if (material.emissive) {
      // albedo_luminance = 0.3;
      // emission_luminance = albedo_luminance;
    }
  
    let camera = getCameraProps();
    let view_normal = camera.view_matrix * vec4f(world_normal, 0);
  
    let aux = vec4(
      view_normal.xy,
      albedo_luminance,
      emission_luminance
    );
  
    // TODO: maybe apply gamma correction to the albedo luminance parameter??
  
    textureStore(auxOutput, GlobalInvocationID.xy, aux);
  }`)
  .$uses({
    MAX_STEPS,
    MarchResult,
    ShapeContext,
    Material,
    constructRayPos,
    constructRayDir,
    marchWithSurfaceDist,
    worldNormals,
    worldMat,
    convertRgbToY,
    getCameraProps,
    auxOutput: auxLayout.bound.auxOutput,
  });

export interface SDFRendererOptions {
  root: ExperimentalTgpuRoot;
  gBuffer: GBuffer;
  quarterResolution?: boolean;
}

export function createSDFRenderer(options: SDFRendererOptions) {
  const { root, gBuffer, quarterResolution } = options;
  const mainPassSize = quarterResolution ? gBuffer.quarterSize : gBuffer.size;
  const auxPassSize = gBuffer.size;

  const LABEL = 'SDF Renderer';
  const camera = new Camera(root);

  const randomSeedPrimerBuffer = root.createBuffer(d.f32).$usage('uniform');
  // How many layers (previous renders) are stacked on top of each other to reduce noise.
  const layersBuffer = root.createBuffer(d.f32).$usage('uniform');

  // Resource locators

  const myGetRandomSeedPrimer = tgpu
    .fn([], d.f32)
    .does(/* wgsl */ `() -> f32 {
      return randomSeedPrimer;
    }`)
    .$uses({ randomSeedPrimer: asUniform(randomSeedPrimerBuffer) });

  const myGetAccumulatedLayers = tgpu
    .fn([], d.f32)
    .does(/* wgsl */ `() -> f32 {
      return layers;
    }`)
    .$uses({ layers: asUniform(layersBuffer) });

  const getMainViewportSize = tgpu
    .fn([], d.vec2f)
    .does(/* wgsl */ `() -> vec2f {
      return vec2f(width, height);
    }`)
    .$uses({ width: mainPassSize[0], height: mainPassSize[1] });

  const getAuxViewportSize = tgpu
    .fn([], d.vec2f)
    .does(/* wgsl */ `() -> vec2f {
      return vec2f(width, height);
    }`)
    .$uses({ width: auxPassSize[0], height: auxPassSize[1] });

  const myGetCameraProps = tgpu
    .fn([], CameraStruct)
    .does(/* wgsl */ `() -> CameraStruct {
      return camera;
    }`)
    .$uses({ CameraStruct, camera: asUniform(camera.cameraBuffer) });

  // ---

  const auxBindGroup = auxLayout.populate({
    auxOutput: gBuffer.auxView,
  });

  const mainPipeline = root
    // filling slots
    .with(OutputFormat, 'rgba8unorm')
    .with(getRandomSeedPrimerSlot, myGetRandomSeedPrimer)
    .with(getAccumulatedLayersSlot, myGetAccumulatedLayers)
    .with(getCameraProps, myGetCameraProps)
    .with(getViewportSizeSlot, getMainViewportSize)
    // ---
    .withCompute(mainComputeFn)
    .createPipeline()
    .$name(`${LABEL} - main pipeline`);

  const auxPipeline = root
    // filling slots
    .with(OutputFormat, 'rgba16float')
    .with(getCameraProps, myGetCameraProps)
    .with(getViewportSizeSlot, getAuxViewportSize)
    // ---
    .withCompute(auxComputeFn)
    .createPipeline()
    .$name(`${LABEL} - aux pipeline`)
    //
    .with(auxLayout, auxBindGroup);

  return {
    perform() {
      randomSeedPrimerBuffer.write(Math.random());
      layersBuffer.write(store.get(accumulatedLayersAtom));
      camera.update();

      const mainBindGroup = mainLayout.populate({
        previousRender: quarterResolution
          ? gBuffer.inQuarterView
          : gBuffer.inRawRenderView,
        mainOutput: quarterResolution
          ? gBuffer.outQuarterView
          : gBuffer.outRawRenderView,
      });

      mainPipeline
        .with(mainLayout, mainBindGroup)
        .dispatchWorkgroups(
          Math.ceil(mainPassSize[0] / BlockSize),
          Math.ceil(mainPassSize[1] / BlockSize),
        );

      auxPipeline.dispatchWorkgroups(
        Math.ceil(gBuffer.size[0] / BlockSize),
        Math.ceil(gBuffer.size[1] / BlockSize),
      );
    },
  };
}
