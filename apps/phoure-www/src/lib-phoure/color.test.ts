import { rgbToYcbcr } from '@typegpu/color';
import { accessViewportSize } from '@typegpu/common';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { worldSdf } from '../lib/GameEngine/sdfRenderer/worldSdf';
import { combinationEntryFn } from './combineShader';
import { convertRgbToY, ycbcrToRgbMatrix } from './color';

it('preserves the luminance range expected by the upscaler', () => {
  expect(convertRgbToY(d.vec3f())).toBeCloseTo(16 / 255);
  expect(convertRgbToY(d.vec3f(1))).toBeCloseTo(235 / 255);
});

it('inverts published YCbCr conversion with the local inverse matrix', () => {
  for (const rgb of [d.vec3f(1, 0, 0), d.vec3f(0, 1, 0), d.vec3f(0, 0, 1)]) {
    const result = std.mul(rgbToYcbcr(rgb), ycbcrToRgbMatrix.$);
    expect(result.x).toBeCloseTo(rgb.x, 5);
    expect(result.y).toBeCloseTo(rgb.y, 5);
    expect(result.z).toBeCloseTo(rgb.z, 5);
  }
});

it('resolves shaders using the published SDF and color packages', () => {
  expect(tgpu.resolve([worldSdf])).toContain('fn ');
  expect(
    tgpu.resolve([combinationEntryFn], {
      config: (cfg) => cfg.with(accessViewportSize, d.vec2f(256)),
    }),
  ).toContain('@fragment');
});
