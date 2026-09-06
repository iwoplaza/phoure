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
