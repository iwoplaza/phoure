import { tgpu, d, std, type TgpuRoot } from 'typegpu';
import { accessViewportSize } from '#src/lib/viewport.ts';
import {
  constructRayDir,
  constructRayPos,
  cameraPropsAccess,
  Camera,
} from '../lib-camera';
import { worldSdf } from '../lib/GameEngine/sdfRenderer/worldSdf';
import { MarchParams } from './marchSdf';

const layout = tgpu.bindGroupLayout({
  previous: { storage: d.arrayOf(d.f32) },
  output: { storage: d.arrayOf(d.f32), access: 'mutable' },
});
export const firstStep = tgpu.slot(false);
export const previousSize = tgpu.accessor(d.vec2u);

/** Conservative coarse-to-fine ray distances; this optional pass is not used by the demo. */
export const coneCompute = tgpu.computeFn({
  workgroupSize: [8, 8],
  in: { gid: d.builtin.globalInvocationId },
})((input) => {
  'use gpu';
  const size = d.vec2u(accessViewportSize.$);
  if (input.gid.x >= size.x || input.gid.y >= size.y) return;
  const origin = constructRayPos();
  const direction = constructRayDir(d.vec2f(input.gid.xy) + d.vec2f(0.5));
  const coneSlope = Math.SQRT2 / accessViewportSize.$.y;
  let distance = d.f32(0);
  if (!firstStep.$) {
    const coord = std.min(
      d.vec2u(d.vec2f(input.gid.xy) / 2),
      previousSize.$ - d.vec2u(1),
    );
    distance = layout.$.previous[coord.y * previousSize.$.x + coord.x];
  }
  for (let step = d.u32(0); step <= MarchParams.maxSteps.$; step++) {
    if (distance >= MarchParams.farPlane.$) break;
    const minimum = worldSdf(origin + distance * direction);
    if (minimum <= coneSlope * distance) break;
    distance += minimum;
  }
  layout.$.output[input.gid.y * size.x + input.gid.x] = distance;
});

export function makeCBuffer(root: TgpuRoot, resolution: [number, number]) {
  const sizes = [16, 8, 4, 2].map(
    (scale) =>
      [Math.ceil(resolution[0] / scale), Math.ceil(resolution[1] / scale)] as [
        number,
        number,
      ],
  );
  const buffers = sizes.map(([x, y]) =>
    root.createBuffer(d.arrayOf(d.f32, x * y)).$usage('storage'),
  );
  return {
    sizes,
    buffers,
    destroy: () => buffers.forEach((buffer) => buffer.destroy()),
  };
}

export type CBuffer = ReturnType<typeof makeCBuffer>;
export type ConeTracerOptions = { root: TgpuRoot; cBuffer: CBuffer };

export default function ConeTracer({ root, cBuffer }: ConeTracerOptions) {
  const camera = new Camera(root);
  const pipelines = cBuffer.buffers.map((buffer, i) => {
    const previous = i === 0 ? cBuffer.buffers.length - 1 : i - 1;
    return root
      .with(cameraPropsAccess, camera.cameraUniform)
      .with(accessViewportSize, d.vec2f(...cBuffer.sizes[i]))
      .with(previousSize, d.vec2u(...cBuffer.sizes[previous]))
      .with(firstStep, i === 0)
      .createComputePipeline({ compute: coneCompute })
      .with(
        root.createBindGroup(layout, {
          previous: cBuffer.buffers[previous],
          output: buffer,
        }),
      );
  });
  return {
    perform() {
      camera.update();
      for (let i = 0; i < pipelines.length; i++) {
        pipelines[i].dispatchWorkgroups(
          Math.ceil(cBuffer.sizes[i][0] / 8),
          Math.ceil(cBuffer.sizes[i][1] / 8),
        );
      }
    },
  };
}
