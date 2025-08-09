import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { convertRgbToY } from '@typegpu/color';
import { accessViewportSize } from '@typegpu/common';
import { randf } from '@typegpu/noise';
import { atom } from 'jotai';
import {
  Camera,
  constructRayDir,
  constructRayPos,
  cameraPropsAccess,
} from 'src/lib-camera';
import {
  estimateNormal,
  march,
  MarchParams,
  MarchResult,
  ShapeContext,
} from 'src/lib-ray-marching';

import { store } from 'src/lib/store.ts';
import type { GBuffer } from '../../gBuffer.ts';
import { Material, skyColor, worldMat, worldSdf } from './worldSdf.ts';

const BlockSize = 8;

// parameters
const SUPER_SAMPLES = 4;
const ONE_OVER_SUPER_SAMPLES = 1 / SUPER_SAMPLES;
const SUB_SAMPLES = 16;
const MAX_REFL = 3;

const randomSeedPrimerAccess = tgpu['~unstable'].accessor(d.f32);
const accumulatedLayersAccess = tgpu['~unstable'].accessor(d.f32);

const Reflection = d.struct({
  color: d.vec3f,
  roughness: d.f32,
});

const ReflectionArray = d.arrayOf(Reflection, MAX_REFL);

export const accumulatedLayersAtom = atom(0);

/**
 * Reflecting: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
 * @param ray_dir
 * @param normal
 * @param mat_roughness
 */
const reflect = tgpu.fn(
  [d.vec3f, d.vec3f, d.f32, d.ptrFn(d.f32)],
  d.vec3f,
)((rayDir, normal, matRoughness, outRoughness) => {
  const slope = std.dot(rayDir, normal);
  const refl_dir = rayDir.sub(normal.mul(2 * slope));

  const fresnel = 1 - std.pow(1 + slope, 16);
  const roughness = matRoughness * fresnel;
  // TODO: Fix when boxed values are introduced
  // biome-ignore lint/style/noParameterAssign: Has to be done like this for now
  outRoughness = roughness;

  let new_ray_dir = randf.onHemisphere(normal);
  new_ray_dir = std.mix(refl_dir, new_ray_dir, roughness);
  return std.normalize(new_ray_dir);
});

const renderSubPixel = tgpu.fn(
  [d.vec2f],
  d.vec3f,
)((coord) => {
  // doing the first march before each sub-sample, since the first march result is the same for all of them

  const init_shape_ctx = ShapeContext({
    rayPos: constructRayPos(),
    rayDir: constructRayDir(coord),
    rayDistance: 0,
  });
  const init_march_result = MarchResult();

  march(init_shape_ctx, MarchParams.maxSteps.$, init_march_result);

  if (init_march_result.steps >= MarchParams.maxSteps.$) {
    return std.min(skyColor(init_shape_ctx.rayDir), d.vec3f(1));
  }

  const init_normal = estimateNormal(
    init_march_result.position,
    init_shape_ctx,
  );
  const init_material = Material();

  worldMat(init_march_result.position, init_shape_ctx, init_material);

  if (init_material.emissive) {
    return std.min(init_material.albedo, d.vec3f(1));
  }

  const reflections = ReflectionArray();

  let acc = d.vec3f();
  for (let sub = d.u32(0); sub < SUB_SAMPLES; sub++) {
    const material = Material(init_material);

    let normal = d.vec3f(init_normal);
    let emissive_color = d.vec3f();
    let refl_count = d.u32(0);

    const shape_ctx = ShapeContext({
      rayPos: init_march_result.position,
      rayDir: init_shape_ctx.rayDir,
      rayDistance: init_shape_ctx.rayDistance,
    });

    for (let refl = d.u32(0); refl < MAX_REFL; refl++) {
      const roughness = d.f32(0);
      shape_ctx.rayDir = reflect(
        shape_ctx.rayDir,
        normal,
        material.roughness,
        roughness,
      );
      reflections[refl_count].color = material.albedo;
      reflections[refl_count].roughness = roughness;
      refl_count++;

      const march_result = MarchResult();
      march(shape_ctx, MarchParams.maxSteps.$, march_result);
      shape_ctx.rayPos = march_result.position;

      if (march_result.steps >= MarchParams.maxSteps.$) {
        emissive_color = skyColor(shape_ctx.rayDir);
        break;
      }

      normal = estimateNormal(shape_ctx.rayPos, shape_ctx);

      worldMat(shape_ctx.rayPos, shape_ctx, material);

      if (material.emissive) {
        emissive_color = material.albedo;
        break;
      }
    }

    let sub_acc = d.vec3f(emissive_color);
    for (let i = d.i32(refl_count) - 1; i >= 0; i--) {
      const mat_color = d.vec3f(reflections[i].color);
      const reflectivity = 1 - reflections[i].roughness;

      sub_acc = sub_acc.mul(
        std.mix(mat_color, d.vec3f(1), std.max(0, std.min(reflectivity, 1))),
      ); // absorb the ray color based on reflectivity
    }

    acc = acc.add(sub_acc);
  }

  // averaging
  acc = acc.div(SUB_SAMPLES);

  // clipping
  acc = std.min(acc, d.vec3f(1));

  return acc;
});

const mainLayout = tgpu.bindGroupLayout({
  previousRender: { texture: 'unfilterable-float' },
  mainOutput: { storageTexture: 'rgba8unorm', access: 'writeonly' },
});

const mainComputeFn = tgpu['~unstable'].computeFn({
  workgroupSize: [BlockSize, BlockSize],
  in: { gid: d.builtin.globalInvocationId },
})((input) => {
  const preSeed = d.vec2f(input.gid.xy);
  randf.seed2(
    preSeed.mul(0.1646936793).add(randomSeedPrimerAccess.$ * 0.934534732),
  );

  const prevLayers = accumulatedLayersAccess.$;
  const prevRender = std.textureLoad(
    mainLayout.$.previousRender,
    input.gid.xy,
    0,
  );

  let acc = d.vec3f(0, 0, 0);
  for (let sx = d.u32(0); sx < SUPER_SAMPLES; sx++) {
    for (let sy = d.u32(0); sy < SUPER_SAMPLES; sy++) {
      const offset = d.vec2f(sx, sy).add(0.5).mul(ONE_OVER_SUPER_SAMPLES);

      acc = acc.add(renderSubPixel(d.vec2f(input.gid.xy).add(offset)));
    }
  }

  acc = acc.mul(ONE_OVER_SUPER_SAMPLES * ONE_OVER_SUPER_SAMPLES);

  // applying gamma correction
  const gamma = 2.2;
  acc = std.pow(acc, d.vec3f(1 / gamma));

  let newRender = d.vec4f(acc, 1);
  if (prevLayers > 0) {
    newRender = prevRender
      .mul(prevLayers)
      .add(d.vec4f(acc, 1.0))
      .div(prevLayers + 1);
  }

  std.textureStore(mainLayout.$.mainOutput, input.gid.xy, newRender);
});

const auxLayout = tgpu.bindGroupLayout({
  auxOutput: { storageTexture: 'rgba16float' },
});

const auxComputeFn = tgpu['~unstable'].computeFn({
  workgroupSize: [BlockSize, BlockSize],
  in: { gid: d.builtin.globalInvocationId },
})((input) => {
  const marchResult = MarchResult();
  const shapeCtx = ShapeContext({
    rayPos: constructRayPos(),
    rayDir: constructRayDir(d.vec2f(input.gid.xy).add(d.vec2f(0.5))),
    rayDistance: 0,
  });

  march(shapeCtx, MarchParams.maxSteps.$, marchResult);

  let worldNormal = d.vec3f();

  if (marchResult.steps >= MarchParams.maxSteps.$) {
    worldNormal = std.neg(shapeCtx.rayDir);
  } else {
    worldNormal = estimateNormal(marchResult.position, shapeCtx);
  }

  const material = Material();
  worldMat(marchResult.position, shapeCtx, material);

  const matColor = std.min(material.albedo, d.vec3f(1));

  const albedoLuminance = convertRgbToY(matColor);
  const emissionLuminance = 0;
  if (material.emissive) {
    // albedoLuminance = 0.3;
    // emissionLuminance = albedoLuminance;
  }

  const viewNormal = cameraPropsAccess.$.view_matrix.mul(
    d.vec4f(worldNormal, 0),
  );

  const aux = d.vec4f(viewNormal.xy, albedoLuminance, emissionLuminance);

  // TODO: maybe apply gamma correction to the albedo luminance parameter??

  std.textureStore(auxLayout.$.auxOutput, input.gid.xy, aux);
});

export interface SDFRendererOptions {
  root: TgpuRoot;
  gBuffer: GBuffer;
  quarterResolution?: boolean;
}

export function createSDFRenderer(options: SDFRendererOptions) {
  const { root, gBuffer, quarterResolution } = options;
  const mainPassSize = quarterResolution ? gBuffer.quarterSize : gBuffer.size;
  const auxPassSize = gBuffer.size;

  const LABEL = 'SDF Renderer';
  const camera = new Camera(root);

  const randomSeedPrimerUniform = root.createUniform(d.f32);
  // How many layers (previous renders) are stacked on top of each other to reduce noise.
  const layersUniform = root.createUniform(d.f32);

  // ---

  const auxBindGroup = root.createBindGroup(auxLayout, {
    auxOutput: gBuffer.auxView,
  });

  const mainPipeline = root['~unstable']
    // filling slots
    .with(randomSeedPrimerAccess, randomSeedPrimerUniform)
    .with(accumulatedLayersAccess, layersUniform)
    .with(cameraPropsAccess, camera.cameraUniform)
    .with(accessViewportSize, d.vec2f(mainPassSize[0], mainPassSize[1]))
    .with(MarchParams.sampleSdf, worldSdf)
    // ---
    .withCompute(mainComputeFn)
    .createPipeline()
    .$name(`${LABEL} - main pipeline`);

  const auxPipeline = root['~unstable']
    // filling slots
    .with(cameraPropsAccess, camera.cameraUniform)
    .with(accessViewportSize, d.vec2f(auxPassSize[0], auxPassSize[1]))
    .with(MarchParams.sampleSdf, worldSdf)
    // ---
    .withCompute(auxComputeFn)
    .createPipeline()
    .$name(`${LABEL} - aux pipeline`)
    //
    .with(auxLayout, auxBindGroup);

  return {
    perform() {
      randomSeedPrimerUniform.write(Math.random());
      layersUniform.write(store.get(accumulatedLayersAtom));
      camera.update();

      const mainBindGroup = root.createBindGroup(mainLayout, {
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
