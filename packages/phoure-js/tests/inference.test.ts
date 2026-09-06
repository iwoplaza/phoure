import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { combinationEntryFn } from '../src/combineShader.js';
import {
  inChannelsSlot,
  kernelRadiusSlot,
  outChannelsSlot,
} from '../src/convolve.js';
import {
  entryComputeFn,
  inputFromGBufferSlot,
  reluSlot,
} from '../src/menderStep.js';
import { Model7 } from '../src/model7.js';
import { accessViewportSize } from '../src/viewport.js';

it.each([
  {
    radius: 4,
    input: 8,
    output: 8,
    relu: true,
    textures: true,
    weights: Model7.Conv1Weight,
    biases: Model7.Conv1Bias,
  },
  {
    radius: 2,
    input: 8,
    output: 8,
    relu: true,
    textures: false,
    weights: Model7.Conv2Weight,
    biases: Model7.Conv2Bias,
  },
  {
    radius: 2,
    input: 8,
    output: 1,
    relu: false,
    textures: false,
    weights: Model7.Conv3Weight,
    biases: Model7.Conv3Bias,
  },
])('resolves the $input → $output layer with radius $radius', (layer) => {
  expect(layer.weights.length).toBe(
    (layer.radius * 2 + 1) ** 2 * layer.input * layer.output,
  );
  expect(layer.biases.length).toBe(layer.output);
  const code = tgpu.resolve([entryComputeFn], {
    config: (cfg) =>
      cfg
        .with(accessViewportSize, d.vec2f(13, 9))
        .with(kernelRadiusSlot, layer.radius)
        .with(inChannelsSlot, layer.input)
        .with(outChannelsSlot, layer.output)
        .with(reluSlot, layer.relu)
        .with(inputFromGBufferSlot, layer.textures),
  });
  expect(code).toContain('@compute');
  expect(code).not.toContain('#src/');
});

it('resolves the color combination pass independently of the demo', () => {
  expect(
    tgpu.resolve([combinationEntryFn], {
      config: (cfg) => cfg.with(accessViewportSize, d.vec2f(13, 9)),
    }),
  ).toContain('@fragment');
});
