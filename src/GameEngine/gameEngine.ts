import tgpu from 'typegpu/experimental';
import type { SetStateAction } from 'jotai';
import { store } from '../store';
import { GBuffer } from '../gBuffer';
import { MenderStep } from '../mending/menderStep';
import {
  autoRotateControlAtom,
  displayModeAtom,
  measurePerformanceAtom,
  targetResolutionAtom,
} from '../controlAtoms';
import { makeGBufferDebugger } from './gBufferDebugger';
import { PostProcessingStep } from './postProcessingStep';
import { BicubicFilter } from '@/lib-filter';
import {
  accumulatedLayersAtom,
  createSDFRenderer,
} from './sdfRenderer/sdfRenderer';
import { PerformanceManager } from '@/PerformanceManager';

class AlreadyDestroyedError extends Error {
  constructor() {
    super('This engine was already destroyed.');

    // Set the prototype explicitly.
    Object.setPrototypeOf(this, AlreadyDestroyedError.prototype);
  }
}

const settingsToPerf = new Map<string, PerformanceManager>();
// biome-ignore lint/suspicious/noExplicitAny: <hack>
(window as any).settingsToPerf = settingsToPerf;

const noteFrame = () => {
  if (!store.get(measurePerformanceAtom)) {
    // Only measuring performance when toggled.
    return;
  }

  const mode = store.get(displayModeAtom);
  const targetResolution = store.get(targetResolutionAtom);

  const key = `${mode} ${targetResolution}`;
  let perf = settingsToPerf.get(key);
  if (!perf) {
    perf = new PerformanceManager();
    settingsToPerf.set(key, perf);
  }

  perf.noteFrame();
};

export const GameEngine = (
  canvas: HTMLCanvasElement,
  targetResolution: number,
) => {
  let destroyed = false;
  const cleanups: (() => unknown)[] = [];

  store.set(accumulatedLayersAtom, 0 as SetStateAction<number>);

  const addCleanup = (cb: () => unknown) => {
    if (destroyed) {
      cb();
      throw new AlreadyDestroyedError();
    }
    cleanups.push(cb);
  };

  (async () => {
    const root = await tgpu.init();
    addCleanup(() => root.destroy());

    const context = canvas.getContext('webgpu') as GPUCanvasContext;

    const devicePixelRatio = window.devicePixelRatio || 1;
    canvas.style.width = `${targetResolution / devicePixelRatio}px`;
    canvas.style.height = `${targetResolution / devicePixelRatio}px`;
    canvas.width = targetResolution;
    canvas.height = targetResolution;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    const gBuffer = new GBuffer(root, [targetResolution, targetResolution]);
    console.log(`Rendering a ${gBuffer.size[0]} by ${gBuffer.size[1]} image`);

    let sdfRenderer: ReturnType<typeof createSDFRenderer>;
    let traditionalSdfRenderer: ReturnType<typeof createSDFRenderer>;
    try {
      sdfRenderer = createSDFRenderer({
        root,
        gBuffer,
        quarterResolution: true,
      });
      traditionalSdfRenderer = createSDFRenderer({ root, gBuffer });
    } catch (err) {
      console.error('Failed to initialize SDF renderers.');
      throw err;
    }

    const upscaleStep = BicubicFilter({
      root,
      sourceTexture: () => gBuffer.outQuarterView,
      targetTexture: gBuffer.upscaledView,
      targetFormat: 'rgba8unorm',
    });

    const menderStep = MenderStep({
      root,
      gBuffer,
      targetTexture: () => gBuffer.outRawRenderView,
    });

    const gBufferDebugger = makeGBufferDebugger(
      root,
      presentationFormat,
      gBuffer,
    );

    const postProcessing = PostProcessingStep({
      root,
      context,
      gBuffer,
      presentationFormat,
    });

    context.configure({
      device: root.device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });

    function frame() {
      if (destroyed) {
        return;
      }

      noteFrame();

      const displayMode = store.get(displayModeAtom);

      // -- Rendering the whole scene & aux.
      if (displayMode === 'traditional') {
        traditionalSdfRenderer.perform();
        root.flush();
      }

      if (
        displayMode === 'g-buffer' ||
        displayMode === 'g-buffer-color' ||
        displayMode === 'g-buffer-albedo' ||
        displayMode === 'g-buffer-normal' ||
        displayMode === 'upscaled'
      ) {
        sdfRenderer.perform();
        root.flush();
      }

      // -- Upscaling the quarter-resolution render.
      upscaleStep.perform();

      // -- Displaying a result to the screen.
      if (
        displayMode === 'g-buffer' ||
        displayMode === 'g-buffer-color' ||
        displayMode === 'g-buffer-albedo' ||
        displayMode === 'g-buffer-normal'
      ) {
        gBufferDebugger.perform(context);
      } else if (displayMode === 'traditional') {
        postProcessing.perform();
      } else if (displayMode === 'upscaled') {
        // -- Restoring quality to the render using convolution.
        menderStep.perform();
        postProcessing.perform();
      }

      root.flush();
      gBuffer.flip();
      if (store.get(autoRotateControlAtom)) {
        store.set(accumulatedLayersAtom, 0 as SetStateAction<number>);
      } else {
        store.set(
          accumulatedLayersAtom,
          (store.get(accumulatedLayersAtom) + 1) as SetStateAction<number>,
        );
      }
      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  })().catch((e) => {
    if (e instanceof AlreadyDestroyedError) {
      // Expected to happen
    } else {
      console.error(e);
    }
  });

  return {
    destroy() {
      destroyed = true;
      for (const cb of cleanups) {
        cb();
      }
    },
  };
};
