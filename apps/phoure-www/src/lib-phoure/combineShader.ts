import { rgbToYcbcrMatrix } from '@typegpu/color';
import { ycbcrToRgbMatrix } from './color';
import { accessViewportSize } from '@typegpu/common';
import { std, d, tgpu } from 'typegpu';

export const layout = tgpu.bindGroupLayout({
  blurredTexture: { texture: d.texture2d(d.f32) },
  mendedBuffer: { storage: (n: number) => d.arrayOf(d.f32, n) },
});

export const combinationEntryFn = tgpu.fragmentFn({
  in: { coord_f: d.builtin.position, uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  'use gpu';
  const coord = d.vec2u(input.coord_f.xy);
  const blurred = std.textureLoad(layout.$.blurredTexture, coord, 0);
  const blurred_ycbcr = blurred.xyz.mul(rgbToYcbcrMatrix.$);

  const buffer_idx = coord.y * d.u32(accessViewportSize.$.x) + coord.x;
  const mended_lumi = layout.$.mendedBuffer[buffer_idx];

  const combined_ycbcr = d.vec3f(
    blurred_ycbcr.x + mended_lumi, // Y
    blurred_ycbcr.y, // Cb
    blurred_ycbcr.z, // Cr
  );

  const combined = combined_ycbcr.mul(ycbcrToRgbMatrix.$);

  return d.vec4f(combined, 1.0);
});
