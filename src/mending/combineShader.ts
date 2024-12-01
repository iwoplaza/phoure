import { getViewportSizeSlot } from '@/GameEngine/commonSlots';
import * as d from 'typegpu/data';
import tgpu, { builtin, wgsl } from 'typegpu/experimental';

const rgbToYcbcrMatrix = wgsl.constant(`mat3x3f(
  0.299,     0.587,     0.114,
 -0.168736, -0.331264,  0.5,
  0.5,      -0.418688, -0.081312,
)`);

const ycbcrToRgbMatrix = wgsl.constant(`mat3x3f(
 1.0,  0,         1.402,
 1.0, -0.344136, -0.714136,
 1.0,  1.772,     0,
)`);

export const combinationLayout = tgpu
  .bindGroupLayout({
    blurredTexture: { texture: 'float' },
    mendedBuffer: { storage: (n) => d.arrayOf(d.f32, n) },
  })
  .$name('combinationLayout');

const { blurredTexture, mendedBuffer } = combinationLayout.bound;

export const combinationEntryFn = tgpu
  .fragmentFn({ pos: builtin.position, uv: d.vec2f }, d.vec4f)
  .does(/* wgsl */ `(@builtin(position) coord_f: vec4f) -> @location(0) vec4f {
    let coord = vec2u(floor(coord_f.xy));

    let blurred = textureLoad(
      blurredTexture,
      coord,
      0
    );
  
    let blurred_ycbcr = blurred.rgb * rgbToYcbcrMatrix;
  
    let buffer_idx = coord.y * u32(getViewportSizeSlot().x) + coord.x;
    let mended_lumi = mendedBuffer[buffer_idx];
  
    let combined_ycbcr = vec3f(
      blurred_ycbcr.r + mended_lumi, // Y
      blurred_ycbcr.g, // Cb
      blurred_ycbcr.b, // Cr
    );
  
    let combined = combined_ycbcr * ycbcrToRgbMatrix;
  
    return vec4f(combined, 1.0);
  }`)
  .$uses({
    rgbToYcbcrMatrix,
    ycbcrToRgbMatrix,
    blurredTexture,
    mendedBuffer,
    getViewportSizeSlot,
  });
