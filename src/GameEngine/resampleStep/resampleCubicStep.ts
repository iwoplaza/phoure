import { fullScreenQuadVertexFn } from '@/shaders/fullScreenQuad';
import tgpu, {
  asUniform,
  builtin,
  type TgpuFn,
  wgsl,
  type ExperimentalTgpuRoot,
} from 'typegpu/experimental';
import * as d from 'typegpu/data';
import { getViewportSizeSlot } from '../commonSlots';

export const getTexelSizeXSlot = wgsl.slot<TgpuFn<[], d.Vec2f>>();
export const getTexelSizeYSlot = wgsl.slot<TgpuFn<[], d.Vec2f>>();

const externalLayout = tgpu
  .bindGroupLayout({
    wrappingSampler: { sampler: 'filtering' },
    clampingSampler: { sampler: 'filtering' },
    texture: { texture: 'float' },
    // filter offsets and weights
    hgLookup: { texture: 'float', viewDimension: '1d' },
  })
  .$name('Resample - external bind group layout');

/**
 * Implementation based on:
 * https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
 */
const resampleCubic = tgpu
  .fragmentFn({ pos: builtin.position, uv: d.vec2f }, d.vec4f)
  .does(/* wgsl */ `(@location(0) uv: vec2f) -> @location(0) vec4f {
    let texel_size_x = getTexelSizeXSlot();
    let texel_size_y = getTexelSizeYSlot();
    let viewport_size = getViewportSizeSlot();
    // calc filter texture coordinates where [0,1] is a single texel
    // (can be done in vertex program instead)
    let coord_hg = uv * viewport_size - vec2f(0.5f, 0.5f);      // fetch offsets and weights from filter texture
    var hg_x = textureSample(hgLookup, wrappingSampler, coord_hg.x).xyz;
    var hg_y = textureSample(hgLookup, wrappingSampler, coord_hg.y).xyz;      // determine linear sampling coordinates
    var coord_source10 = uv + hg_x.x * texel_size_x;
    var coord_source00 = uv - hg_x.y * texel_size_x;
    var coord_source11 = coord_source10 + hg_y.x * texel_size_y;
    var coord_source01 = coord_source00 + hg_y.x * texel_size_y;
    coord_source10 = coord_source10 - hg_y.y * texel_size_y;
    coord_source00 = coord_source00 - hg_y.y * texel_size_y;      // fetch four linearly interpolated inputs
    var tex_source00 = textureSample(texture, clampingSampler, coord_source00);
    var tex_source10 = textureSample(texture, clampingSampler, coord_source10);
    var tex_source01 = textureSample(texture, clampingSampler, coord_source01);
    var tex_source11 = textureSample(texture, clampingSampler, coord_source11);      // weight along y direction
    tex_source00 = mix(tex_source00, tex_source01, hg_y.z);
    tex_source10 = mix(tex_source10, tex_source11, hg_y.z);      // weight along x direction
    tex_source00 = mix(tex_source00, tex_source10, hg_x.z);
    
    return tex_source00;
    // Doing linear interpolation for now.
    // return textureSample(texture, clampingSampler, uv);
  }`)
  .$uses({
    getTexelSizeXSlot,
    getTexelSizeYSlot,
    getViewportSizeSlot,
    hgLookup: externalLayout.bound.hgLookup,
    wrappingSampler: externalLayout.bound.wrappingSampler,
    clampingSampler: externalLayout.bound.clampingSampler,
  });

/**
 * Lookup texture of `h` and `g` functions defined in https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering.
 * @param device
 * @param samples How frequently to sample the continuum. According to the source material, 128 is enough.
 */
const HGLookupTexture = (root: ExperimentalTgpuRoot, samples = 128) => {
  const textureData = new Uint8Array(samples * 4);

  const texture = root.device.createTexture({
    label: 'HG Lookup Texture',
    format: 'rgba8unorm',
    size: [samples],
    dimension: '1d',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  // Generating lookup data
  for (let i = 0; i < samples; ++i) {
    const x = i / samples;

    const x2 = x ** 2;
    const x3 = x ** 3;
    const w0 = (1 / 6) * (-x3 + 3 * x2 - 3 * x + 1);
    const w1 = (1 / 6) * (3 * x3 - 6 * x2 + 4);
    const w2 = (1 / 6) * (-3 * x3 + 3 * x2 + 3 * x + 1);
    const w3 = (1 / 6) * x3;

    // h0
    textureData[i * 4 + 0] = Math.floor((1 - w1 / (w0 + w1) + x) * 255);
    // h1
    textureData[i * 4 + 1] = Math.floor((1 + w3 / (w2 + w3) - x) * 255);
    // g0
    textureData[i * 4 + 2] = Math.floor((w0 + w1) * 255);
    // g1
    textureData[i * 4 + 3] = Math.floor((w2 + w3) * 255);
  }

  root.device.queue.writeTexture(
    { texture },
    textureData,
    { bytesPerRow: samples * 4 },
    { width: samples },
  );

  return texture;
};

type Options = {
  root: ExperimentalTgpuRoot;
  targetFormat: GPUTextureFormat;
  sourceTexture: () => GPUTextureView;
  targetTexture: GPUTextureView;
  sourceSize: [number, number];
};

export const ResampleStep = ({
  root,
  targetFormat,
  sourceTexture,
  targetTexture,
  sourceSize,
}: Options) => {
  const hgLookupTexture = HGLookupTexture(root);

  const wrappingSampler = root.device.createSampler({
    label: 'Resample - Wrapping Sampler',
    minFilter: 'linear',
    magFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    addressModeW: 'repeat',
  });

  const clampingSampler = root.device.createSampler({
    label: 'Resample - Clamping Sampler',
    minFilter: 'linear',
    magFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });

  const ViewportStruct = d.struct({
    size: d.vec2f,
    texelSizeX: d.vec2f,
    texelSizeY: d.vec2f,
  });

  const viewportBuffer = root
    .createBuffer(ViewportStruct, {
      size: d.vec2f(sourceSize[0], sourceSize[1]),
      texelSizeX: d.vec2f(1 / sourceSize[0], 0),
      texelSizeY: d.vec2f(0, 1 / sourceSize[1]),
    })
    .$usage('uniform');

  const viewportUniform = asUniform(viewportBuffer);

  const myGetViewportSize = tgpu
    .fn([], d.vec2f)
    .does(`() -> vec2f {
      return viewportUniform.size;
    }`)
    .$uses({ viewportUniform });

  const myGetTexelSizeX = tgpu
    .fn([], d.vec2f)
    .does(`() -> vec2f {
      return viewportUniform.texelSizeX;
    }`)
    .$uses({ viewportUniform });

  const myGetTexelSizeY = tgpu
    .fn([], d.vec2f)
    .does(`() -> vec2f {
      return viewportUniform.texelSizeY;
    }`)
    .$uses({ viewportUniform });

  const pipeline = root
    .with(getViewportSizeSlot, myGetViewportSize)
    .with(getTexelSizeXSlot, myGetTexelSizeX)
    .with(getTexelSizeYSlot, myGetTexelSizeY)
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(resampleCubic, { format: targetFormat })
    .createPipeline()
    .$name('Resample Pipeline');

  const passColorAttachment: GPURenderPassColorAttachment = {
    view: targetTexture,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear',
    storeOp: 'store',
  };

  const hgLookupView = hgLookupTexture.createView();

  return {
    perform() {
      const externalBindGroup = externalLayout.populate({
        wrappingSampler,
        clampingSampler,
        texture: sourceTexture(),
        hgLookup: hgLookupView,
      });

      viewportBuffer.write({
        size: d.vec2f(sourceSize[0], sourceSize[1]),
        texelSizeX: d.vec2f(1 / sourceSize[0], 0),
        texelSizeY: d.vec2f(0, 1 / sourceSize[1]),
      });

      pipeline
        .with(externalLayout, externalBindGroup)
        .withColorAttachment(passColorAttachment)
        .draw(6);
    },
  };
};
