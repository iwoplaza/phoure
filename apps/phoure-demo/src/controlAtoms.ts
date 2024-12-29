import { atom } from 'jotai';
import { atomWithUrl } from './atomWithUrl';

export const DisplayModes = [
  { key: 'upscaled', label: 'Upscaled' },
  { key: 'traditional', label: 'Traditional' },
  { key: 'g-buffer', label: 'G-Buffer - Split view' },
  { key: 'g-buffer-color', label: 'G-Buffer - Color' },
  { key: 'g-buffer-albedo', label: 'G-Buffer - Albedo' },
  { key: 'g-buffer-normal', label: 'G-Buffer - Normal' },
] as const;

export type DisplayMode = (typeof DisplayModes)[number]['key'];

export const measurePerformanceAtom = atomWithUrl('perf', false);

// -------------------
// Camera controls
// -------------------

export const cameraOrientationControlAtom = atomWithUrl('cyaw', 0);
export const cameraYControlAtom = atomWithUrl('cy', 0);
export const cameraZoomControlAtom = atomWithUrl('cd', 2);
export const cameraFovControlAtom = atomWithUrl('fov', 90);

export const autoCameraOrientation = atom(0);

export const autoRotateSpeedAtom = atomWithUrl(
  'crot',
  0.5, // degrees per second
);

export const autoRotateControlAtom = (() => {
  const innerAtom = atomWithUrl('cauto', true);

  return atom(
    (get) => get(innerAtom),
    (get, set, autoRotate: boolean) => {
      // Used to start rotation from the last manual position.
      if (autoRotate) {
        set(autoCameraOrientation, get(cameraOrientationControlAtom));
      }
      set(innerAtom, autoRotate);
    },
  );
})();

// -------------------
// Rendering controls
// -------------------

export const targetResolutionAtom = atomWithUrl('res', 256);

export const displayModeAtom = atomWithUrl<DisplayMode>('mode', 'upscaled');

// -------------------
// Time controls
// -------------------

export const fixedTimestepEnabledAtom = atomWithUrl('tfix', true);
export const fixedTimestepAtom = atomWithUrl('tstep', 0.3 /* seconds */);
