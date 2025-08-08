import { rgbToYcbcrMatrix, ycbcrToRgbMatrix } from '@typegpu/color';
import { accessViewportSize } from '@typegpu/common';
import tgpu from 'typegpu';
import * as std from 'typegpu/std';
import * as d from 'typegpu/data';

export const layout = tgpu.bindGroupLayout({
  blurredTexture: { texture: 'float' },
  mendedBuffer: { storage: (n: number) => d.arrayOf(d.f32, n) },
});

export const combinationEntryFn = tgpu['~unstable'].fragmentFn({
  in: { coord_f: d.builtin.position, uv: d.vec2f },
  out: d.vec4f,
})((input) => {
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
