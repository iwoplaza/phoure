import { tgpu, d, std, type TgpuRoot } from 'typegpu';

const layout = tgpu.bindGroupLayout({
  wrappingSampler: { sampler: 'filtering' },
  clampingSampler: { sampler: 'filtering' },
  texture: { texture: d.texture2d(d.f32) },
  hgLookup: { texture: d.texture1d(d.f32) },
});

export const fullScreenQuadVertexFn = tgpu.vertexFn({
  in: { idx: d.builtin.vertexIndex },
  out: {
    pos: d.builtin.position,
    uv: d.vec2f,
    texelSizeX: d.vec2f,
    texelSizeY: d.vec2f,
    coordHG: d.vec2f,
  },
})((input) => {
  'use gpu';
  const positions = d.arrayOf(
    d.vec2f,
    6,
  )([
    d.vec2f(-1, -1),
    d.vec2f(1, -1),
    d.vec2f(-1, 1),
    d.vec2f(1, -1),
    d.vec2f(-1, 1),
    d.vec2f(1, 1),
  ]);
  const uvs = d.arrayOf(
    d.vec2f,
    6,
  )([
    d.vec2f(0, 1),
    d.vec2f(1, 1),
    d.vec2f(0, 0),
    d.vec2f(1, 1),
    d.vec2f(0, 0),
    d.vec2f(1, 0),
  ]);
  const viewportSize = d.vec2f(std.textureDimensions(layout.$.texture));
  return {
    pos: d.vec4f(positions[input.idx], 0, 1),
    uv: uvs[input.idx],
    texelSizeX: d.vec2f(1 / viewportSize.x, 0),
    texelSizeY: d.vec2f(0, 1 / viewportSize.y),
    coordHG: uvs[input.idx] * viewportSize - d.vec2f(0.5),
  };
});

/** Fast third-order filtering, GPU Gems 2, chapter 20. */
export const resampleCubic = tgpu.fragmentFn({
  in: {
    pos: d.builtin.position,
    uv: d.vec2f,
    texelSizeX: d.vec2f,
    texelSizeY: d.vec2f,
    coordHG: d.vec2f,
  },
  out: d.vec4f,
})((input) => {
  'use gpu';
  const hx = std.textureSample(
    layout.$.hgLookup,
    layout.$.wrappingSampler,
    input.coordHG.x,
  ).xyz;
  const hy = std.textureSample(
    layout.$.hgLookup,
    layout.$.wrappingSampler,
    input.coordHG.y,
  ).xyz;
  const right = input.uv + hx.x * input.texelSizeX;
  const left = input.uv - hx.y * input.texelSizeX;
  const c00 = left - hy.y * input.texelSizeY;
  const c10 = right - hy.y * input.texelSizeY;
  const c01 = left + hy.x * input.texelSizeY;
  const c11 = right + hy.x * input.texelSizeY;
  const t00 = std.textureSample(
    layout.$.texture,
    layout.$.clampingSampler,
    c00,
  );
  const t10 = std.textureSample(
    layout.$.texture,
    layout.$.clampingSampler,
    c10,
  );
  const t01 = std.textureSample(
    layout.$.texture,
    layout.$.clampingSampler,
    c01,
  );
  const t11 = std.textureSample(
    layout.$.texture,
    layout.$.clampingSampler,
    c11,
  );
  return std.mix(std.mix(t00, t01, hy.z), std.mix(t10, t11, hy.z), hx.z);
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

  const pipeline = root
    .createRenderPipeline({
      vertex: fullScreenQuadVertexFn,
      fragment: resampleCubic,
      targets: { format: targetFormat },
    })
    .$name('Resample (cubic) Pipeline');

  const passColorAttachment = {
    view: targetTexture,

    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: 'clear' as const,
    storeOp: 'store' as const,
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
