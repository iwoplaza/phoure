import { atom } from 'jotai';
import tgpu, {
  type ExperimentalTgpuRoot,
  asUniform,
} from 'typegpu/experimental';
import { rand, DefaultGenerator } from '@typegpu/noise';
import { MarchResult, ShapeContext } from '@/lib-ray-marching';
import * as d from 'typegpu/data';
import type { GBuffer } from '../../gBuffer';
import {
  Camera,
  constructRayDir,
  constructRayPos,
  getCameraProps,
} from '@/lib-camera';
import {
  Material,
  skyColor,
  surfaceDist,
  worldMat,
  worldSdf,
} from './worldSdf';
import { ONES_3F } from '../wgslUtils/mathConstants';
import { convertRgbToY } from './colorUtils';
import { store } from '@/store';
import { march, MarchParams } from '@/lib-ray-marching';
import { getViewportSize } from '../commonSlots';
import { normalize, mul } from 'typegpu/std';

const BlockSize = 8;

// parameters
const OutputFormat = tgpu.slot().$name('output_format');

const SUPER_SAMPLES = 4;
const ONE_OVER_SUPER_SAMPLES = 1 / SUPER_SAMPLES;
const SUB_SAMPLES = 16;
const MAX_REFL = 3;

const getRandomSeedPrimer = tgpu.accessor(d.f32);
const getAccumulatedLayers = tgpu.accessor(d.f32);

const Reflection = d.struct({
  color: d.vec3f,
  roughness: d.f32,
});

export const accumulatedLayersAtom = atom(0);

/**
 * Reflecting: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
 * @param ray_dir
 * @param normal
 * @param mat_roughness
 */
const reflect = tgpu
  .fn([d.vec3f, d.vec3f, d.f32, /* TODO: ptr */ d.f32])
  .does(`(ray_dir: vec3f, normal: vec3f, mat_roughness: f32, out_roughness: ptr<function, f32>) -> vec3f {
    let slope = dot(ray_dir, normal);
    let dn2 = 2. * slope;
    let refl_dir = ray_dir - dn2 * normal;

    let fresnel = 1. - pow(1. + slope, 16.);
    let roughness = mat_roughness * fresnel;
    *out_roughness = roughness;

    var new_ray_dir = randOnHemisphere(normal);
    new_ray_dir = mix(refl_dir, new_ray_dir, roughness);
    return normalize(new_ray_dir);
  }`)
  .$uses({ randOnHemisphere: rand.onHemisphere })
  .$name('reflect');

const worldNormals = tgpu
  .fn([d.vec3f, ShapeContext], d.vec3f)
  .does((point, ctx) => {
    const epsilon = surfaceDist(ctx) * 0.5; // arbitrary - should be smaller than any surface detail in your distance function, but not so small as to get lost in float precision
    const offX = d.vec3f(point.x + epsilon, point.y, point.z);
    const offY = d.vec3f(point.x, point.y + epsilon, point.z);
    const offZ = d.vec3f(point.x, point.y, point.z + epsilon);

    const centerDistance = MarchParams.sampleSdf.value(point);
    const xDistance = MarchParams.sampleSdf.value(offX);
    const yDistance = MarchParams.sampleSdf.value(offY);
    const zDistance = MarchParams.sampleSdf.value(offZ);

    return normalize(
      mul(
        1 / epsilon,
        d.vec3f(
          xDistance - centerDistance,
          yDistance - centerDistance,
          zDistance - centerDistance,
        ),
      ),
    );
  });

const renderSubPixel = tgpu
  .fn([d.vec2f], d.vec3f)
  .does(/* wgsl */ `(coord: vec2f) -> vec3f {
    // doing the first march before each sub-sample, since the first march result is the same for all of them

    var init_shape_ctx: ShapeContext;
    init_shape_ctx.rayPos = constructRayPos();
    init_shape_ctx.rayDir = constructRayDir(coord);
    init_shape_ctx.rayDistance = 0.;
    var init_march_result: MarchResult;

    march(&init_shape_ctx, MAX_STEPS, &init_march_result);

    if (init_march_result.steps >= MAX_STEPS) {
      return min(skyColor(init_shape_ctx.rayDir), ONES_3F);
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
      shape_ctx.rayPos = init_march_result.position;
      shape_ctx.rayDir = init_shape_ctx.rayDir;
      shape_ctx.rayDistance = init_shape_ctx.rayDistance;

      for (var refl = 0u; refl < MAX_REFL; refl++) {
        var roughness: f32 = 0.;
        shape_ctx.rayDir = reflect(
          shape_ctx.rayDir,
          normal,
          material.roughness,
          &roughness,
        );
        reflections[refl_count].color = material.albedo;
        reflections[refl_count].roughness = roughness;
        refl_count++;

        var march_result: MarchResult;
        march(&shape_ctx, MAX_STEPS, &march_result);
        shape_ctx.rayPos = march_result.position;

        if (march_result.steps >= MAX_STEPS) {
          emissive_color = skyColor(shape_ctx.rayDir);
          break;
        }

        normal = worldNormals(shape_ctx.rayPos, shape_ctx);

        worldMat(shape_ctx.rayPos, shape_ctx, &material);

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
    MAX_STEPS: MarchParams.maxSteps,
    MAX_REFL,
    ONES_3F,
    MarchResult,
    Material,
    Reflection,
    ShapeContext,
    constructRayPos,
    constructRayDir,
    worldNormals,
    march,
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
    setupRandomSeed(vec2f(gid.xy) * ${Math.random()} + getRandomSeedPrimer * ${Math.random()});

    let prev_layers = getAccumulatedLayers;
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
    setupRandomSeed: DefaultGenerator.seed,
    renderSubPixel,
    getRandomSeedPrimer,
    getAccumulatedLayers,
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
    shape_ctx.rayPos = constructRayPos();
    shape_ctx.rayDir = constructRayDir(
      vec2f(GlobalInvocationID.xy) + offset
    );
    shape_ctx.rayDistance = 0.;
  
    march(&shape_ctx, MAX_STEPS, &march_result);
  
    var world_normal: vec3f;
  
    if (march_result.steps >= MAX_STEPS) {
      world_normal = -shape_ctx.rayDir;
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
  
    let camera = getCameraProps;
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
    MAX_STEPS: MarchParams.maxSteps,
    MarchResult,
    ShapeContext,
    Material,
    constructRayPos,
    constructRayDir,
    march,
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

  // ---

  const auxBindGroup = auxLayout.populate({
    auxOutput: gBuffer.auxView,
  });

  const mainPipeline = root
    // filling slots
    .with(OutputFormat, 'rgba8unorm')
    .with(getRandomSeedPrimer, asUniform(randomSeedPrimerBuffer))
    .with(getAccumulatedLayers, asUniform(layersBuffer))
    .with(getCameraProps, asUniform(camera.cameraBuffer))
    .with(getViewportSize, d.vec2f(mainPassSize[0], mainPassSize[1]))
    .with(MarchParams.sampleSdf, worldSdf)
    // ---
    .withCompute(mainComputeFn)
    .createPipeline()
    .$name(`${LABEL} - main pipeline`);

  const auxPipeline = root
    // filling slots
    .with(OutputFormat, 'rgba16float')
    .with(getCameraProps, asUniform(camera.cameraBuffer))
    .with(getViewportSize, d.vec2f(auxPassSize[0], auxPassSize[1]))
    .with(MarchParams.sampleSdf, worldSdf)
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
