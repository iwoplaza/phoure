<div align="center">

![phoure (light mode)](/media/phoure-logo-light.svg#gh-light-mode-only)
![phoure (dark mode)](/media/phoure-logo-dark.svg#gh-dark-mode-only)

Ethical AI upscaling for games. _(formerly "mender")_

</div>

<div align="center">
<div style="max-width: 640px">

![hello](/media/phoure-app.png)

</div>
</div>

## What does it do?

It upscales an image by a factor of $4$ (1K → 4K, etc.) with a minimal hit to
performance. For the renderer implemented in the official example, it produces a
$2024\times2024$ image ~12 times faster than targeting $2024\times2024$
directly.

## Ethical?

The dataset that was used to train the upscaling neural network was generated
with Blender's Geometry Nodes. **No artist's work was used in the process**.

## Roadmap

- [x] Generate procedural data-set with Blender Geometry Nodes.
- [x] Create a slim network architecture.
- [x] Train the network.
- [x] Create ray marching renderer (for SDFs).
- [x] Implement inference on the WebGPU.
- [ ] Use multisampling for render passes.
- [ ] Create rasterizing renderer (polygonal geometry).
- [ ] Extract into a separate package, to be used as an upscaling step in other
      WebGPU projects.

## Development

### Local setup

- Install Node.js 22.12+
- Install pnpm
- Run `pnpm install` to install all dependencies
- Run `pnpm dev` to run the development server, making the web app available via
  the web browser.

### Linting and formatting

- Run `pnpm check` to lint with oxlint and check formatting with oxfmt.
- Run `pnpm fix` to apply lint fixes and format files.
- Use `pnpm lint` or `pnpm format:check` to run either check independently.
- Install the recommended Oxc VS Code extension for linting and formatting on
  save. Astro files continue to use the Astro extension for formatting.

### GPU development

The workspace uses TypeGPU 0.12.4 and unplugin-typegpu 0.12.3. Shaders are
TypeScript functions marked with `'use gpu'`, including the convolution passes,
bicubic filter, scene shaders, and optional edge-detection and cone-tracing
passes. The demo uses published `@typegpu/sdf` and `@typegpu/color` packages.
The pretrained inference model, convolution shaders, and color utilities live in
`packages/phoure-js`; the demo consumes its `MenderStep` API from `phoure`. See
[the package README](packages/phoure-js/README.md) for inputs and lifecycle.

The `typescript` dependency is an npm alias for **tsover 5.9.13**, including a
pnpm override for build plugins. This supports vector and matrix operators while
remaining compatible with the existing Astro/Starlight stack. Select **Use
Workspace Version** for TypeScript in VS Code; the existing SDK path still
applies.

- `pnpm check` includes the recommended TypeGPU shader rules through oxlint.
- `pnpm test:types` checks every workspace with tsover.
- `pnpm test:unit` runs math and shader-generation tests, including all three
  neural-network layer configurations.
- `pnpm test` builds the workspace and runs both checks.

The optional cone tracer now accepts `{ root, cBuffer }`, with buffers allocated
by `makeCBuffer(root, resolution)`. Edge detection accepts `root` in place of
the old runtime. Both were unfinished modules using APIs removed from TypeGPU.
