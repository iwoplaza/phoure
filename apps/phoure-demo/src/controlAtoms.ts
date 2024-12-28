import { atom } from 'jotai';
import { atomWithStorage } from 'jotai/utils';

export const DisplayModes = [
  { key: 'upscaled', label: 'Upscaled' },
  { key: 'traditional', label: 'Traditional' },
  { key: 'g-buffer', label: 'G-Buffer - Split view' },
  { key: 'g-buffer-color', label: 'G-Buffer - Color' },
  { key: 'g-buffer-albedo', label: 'G-Buffer - Albedo' },
  { key: 'g-buffer-normal', label: 'G-Buffer - Normal' },
] as const;

export type DisplayMode = (typeof DisplayModes)[number]['key'];

export const measurePerformanceAtom = atomWithStorage(
  'MEASURE_PERFORMANCE',
  false,
);

export const cameraOrientationControlAtom = atomWithStorage(
  'CAMERA_ORIENTATION',
  0,
);

export const cameraYControlAtom = atomWithStorage('CAMERA_Y', 0);

export const cameraZoomControlAtom = atomWithStorage('CAMERA_ZOOM', 2);

export const cameraFovControlAtom = atomWithStorage('CAMERA_FOV', 90);

export const autoCameraOrientation = atom(0);

export const autoRotateSpeedAtom = atomWithStorage(
  'AUTO_ROTATE_SPEED',
  0.5, // degrees per second
);

export const autoRotateControlAtom = (() => {
  const innerAtom = atomWithStorage('AUTO_ROTATE', true);

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

export const targetResolutionAtom = atomWithStorage('TARGET_RESOLUTION', 256);

export const displayModeAtom = atomWithStorage<DisplayMode>(
  'DISPLAY_MODE',
  'upscaled',
);

export const fixedTimestepEnabledAtom = atomWithStorage(
  'FIXED_TIMESTEP_ENABLED',
  true,
);

export const fixedTimestepAtom = atomWithStorage(
  'FIXED_TIMESTEP',
  0.3 /* seconds */,
);
