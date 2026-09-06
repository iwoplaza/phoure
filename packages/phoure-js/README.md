# phoure

TypeGPU inference for the pretrained Phoure image-restoration model. It runs
three convolution layers and adds the predicted luminance residual to an
already-upscaled color image, preserving its chroma.

## Usage

Install `phoure` and `typegpu` (0.12.4 or compatible 0.12.x). Use the same
TypeGPU root/device as the renderer that owns your textures.

```ts
import { MenderStep } from 'phoure';

const inference = MenderStep({
  root,
  size: [width, height],
  colorTexture: upscaledColorView,
  auxTexture: auxiliaryView,
  targetFormat: 'rgba8unorm',
});

// After rendering the inputs, run all three layers and combine into the output.
inference.perform(outputView);

// When resizing or disposing:
inference.destroy();
```

Inputs and output must be single-sample 2D textures of the specified resolution.
The color input contains bicubic-upscaled RGB; the package does not perform the
initial resize. Both inputs need `GPUTextureUsage.TEXTURE_BINDING`. The output
needs `GPUTextureUsage.RENDER_ATTACHMENT`, must match `targetFormat`, and must
not alias either input. Input views are fixed when the instance is created;
output views can change each frame. Recreate the instance to resize or replace
inputs.

The auxiliary texture packs
`(normal.x, normal.y, albedo luminance, emission luminance)` into RGBA. Use a
signed floating-point format such as `rgba16float` to preserve normals. Normals
and color values should match the renderer's training conventions; this is an
auxiliary-guided model, not an arbitrary image-only upscaler. The exported
`convertRgbToY` GPU function computes the luminance encoding used by the model
and can be used when producing auxiliary inputs. The demo's renderer is a
reference for preparing these inputs.

`perform(outputView)` records and submits one command buffer. To integrate with
an existing frame, call `perform(outputView, encoder)` instead; the caller then
finishes and submits that encoder. Record inference after writes to the input
textures and before passes that read the result.

`destroy()` is idempotent and releases the instance's intermediate and model
buffers. It does not destroy the supplied root, input textures, or output
texture. Calls to `perform` after destruction throw. Each instance owns its own
copy of the pretrained weights and allocates 17 floats per output pixel for
intermediate storage; choose a resolution that fits your device's storage buffer
limits.

## Development and packaging

Workspace imports resolve to `src/index.ts`; the app's TypeGPU Vite plugin
transforms these source shaders. `pnpm dev` at the repository root runs the app
without a library watcher.

Run `pnpm --dir packages/phoure-js pack` from the repository root to build and
pack the library. `prepack` generates ESM, CommonJS, and TypeScript
declarations; `publishConfig` redirects the tarball's entry points to `dist`.
Packed shaders already include TypeGPU metadata. The package has no dependency
on demo code or the local `@typegpu/common` workspace package.
