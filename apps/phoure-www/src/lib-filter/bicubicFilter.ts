import tgpu, { type TgpuRoot } from 'typegpu';
import { builtin, vec2f, vec4f } from 'typegpu/data';

const layout = tgpu
  .bindGroupLayout({
    wrappingSampler: { sampler: 'filtering' },
    clampingSampler: { sampler: 'filtering' },
    texture: { texture: 'float' },
    // filter offsets and weights
    hgLookup: { texture: 'float', viewDimension: '1d' },
  })
  .$name('Resample - external bind group layout');

const fullScreenQuadVertexFn = tgpu['~unstable']
  .vertexFn({
    in: { idx: builtin.vertexIndex },
    out: {
      pos: builtin.position,
      uv: vec2f,
      texelSizeX: vec2f,
      texelSizeY: vec2f,
      coordHG: vec2f,
    },
  })(/* wgsl */ `{
    const SCREEN_RECT = array<vec2f, 6>(
      vec2f(-1.0, -1.0),
      vec2f(1.0, -1.0),
      vec2f(-1.0, 1.0),

      vec2f(1.0, -1.0),
      vec2f(-1.0, 1.0),
      vec2f(1.0, 1.0),
    );

    const UVS = array<vec2f, 6>(
      vec2f(0.0, 1.0),
      vec2f(1.0, 1.0),
      vec2f(0.0, 0.0),

      vec2f(1.0, 1.0),
      vec2f(0.0, 0.0),
      vec2f(1.0, 0.0),
    );

    let viewport_size = vec2f(textureDimensions(texture));

    var output: Out;
    output.pos = vec4f(SCREEN_RECT[in.idx], 0.0, 1.0);
    output.uv = UVS[in.idx];
    output.texelSizeX = vec2f(1. / f32(viewport_size.x), 0);
    output.texelSizeY = vec2f(0, 1. / f32(viewport_size.y));

    // calc filter texture coordinates where [0,1] is a single texel
    output.coordHG = UVS[in.idx] * viewport_size - vec2f(0.5f, 0.5f);      // fetch offsets and weights from filter texture

    return output;
  }`)
  .$uses({ texture: layout.bound.texture });

/**
 * Implementation based on:
 * https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
 */
const resampleCubic = tgpu['~unstable']
  .fragmentFn({
    in: {
      pos: builtin.position,
      uv: vec2f,
      texelSizeX: vec2f,
      texelSizeY: vec2f,
      coordHG: vec2f,
    },
    out: vec4f,
  })(/* wgsl */ `{
    var hg_x = textureSample(hgLookup, wrappingSampler, in.coordHG.x).xyz;
    var hg_y = textureSample(hgLookup, wrappingSampler, in.coordHG.y).xyz;      // determine linear sampling coordinates
    var coord_source10 = in.uv + hg_x.x * in.texelSizeX;
    var coord_source00 = in.uv - hg_x.y * in.texelSizeX;
    var coord_source11 = coord_source10 + hg_y.x * in.texelSizeY;
    var coord_source01 = coord_source00 + hg_y.x * in.texelSizeY;
    coord_source10 = coord_source10 - hg_y.y * in.texelSizeY;
    coord_source00 = coord_source00 - hg_y.y * in.texelSizeY;      // fetch four linearly interpolated inputs
    var tex_source00 = textureSample(texture, clampingSampler, coord_source00);
    var tex_source10 = textureSample(texture, clampingSampler, coord_source10);
    var tex_source01 = textureSample(texture, clampingSampler, coord_source01);
    var tex_source11 = textureSample(texture, clampingSampler, coord_source11);      // weight along y direction
    tex_source00 = mix(tex_source00, tex_source01, hg_y.z);
    tex_source10 = mix(tex_source10, tex_source11, hg_y.z);      // weight along x direction
    tex_source00 = mix(tex_source00, tex_source10, hg_x.z);

    return tex_source00;
  }`)
  .$uses({
    hgLookup: layout.bound.hgLookup,
    texture: layout.bound.texture,
    wrappingSampler: layout.bound.wrappingSampler,
    clampingSampler: layout.bound.clampingSampler,
  });

/**
 * Lookup texture of `h` and `g` functions defined in https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering.
 * @param device
 * @param samples How frequently to sample the continuum. According to the source material, 128 is enough.
 */
const HGLookupTexture = (root: TgpuRoot, samples = 128) => {
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
  root: TgpuRoot;
  targetFormat: GPUTextureFormat;
  sourceTexture: () => GPUTextureView;
  targetTexture: GPUTextureView;
};

export const BicubicFilter = ({
  root,
  targetFormat,
  sourceTexture,
  targetTexture,
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

  const pipeline = root['~unstable']
    .withVertex(fullScreenQuadVertexFn, {})
    .withFragment(resampleCubic, { format: targetFormat })
    .createPipeline()
    .$name('Resample (cubic) Pipeline');

  const passColorAttachment: GPURenderPassColorAttachment = {
    view: targetTexture,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear',
    storeOp: 'store',
  };

  const hgLookupView = hgLookupTexture.createView();

  return {
    perform() {
      const externalBindGroup = root.createBindGroup(layout, {
        wrappingSampler,
        clampingSampler,
        texture: sourceTexture(),
        hgLookup: hgLookupView,
      });

      pipeline
        .with(layout, externalBindGroup)
        .withColorAttachment(passColorAttachment)
        .draw(6);
    },
  };
};
