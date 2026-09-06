import { accessViewportSize } from '#src/lib/viewport.ts';
import { d, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { normalize } from 'typegpu/std';
import { mat4, vec3 } from 'wgpu-matrix';

import {
  autoCameraOrientation,
  autoRotateControlAtom,
  autoRotateSpeedAtom,
  cameraFovControlAtom,
  cameraOrientationControlAtom,
  cameraYControlAtom,
  cameraZoomControlAtom,
} from '#src/lib/controlAtoms.ts';
import { store } from '#src/lib/store.ts';

export const CameraStruct = d.struct({
  view_matrix: d.mat4x4f,
  inv_view_matrix: d.mat4x4f,
  field_of_view: d.f32,
});

export const cameraPropsAccess = tgpu.accessor(CameraStruct);

export const constructRayPos = tgpu.fn(
  [],
  d.vec3f,
)(() => {
  'use gpu';
  return cameraPropsAccess.$.inv_view_matrix.mul(d.vec4f(0, 0, 0, 1)).xyz;
});

export const constructRayDir = tgpu.fn(
  [d.vec2f],
  d.vec3f,
)((coord) => {
  'use gpu';
  const viewCoords = coord
    .sub(accessViewportSize.$.div(2))
    .div(accessViewportSize.$.y)
    // y in [-0.5, 0.5]
    .mul(cameraPropsAccess.$.field_of_view);

  const viewRayDir = normalize(d.vec3f(viewCoords, -0.5));
  viewRayDir.y *= -1;

  return cameraPropsAccess.$.inv_view_matrix.mul(d.vec4f(viewRayDir, 0)).xyz;
});

export class Camera {
  #lastTime = performance.now();

  readonly cameraUniform: TgpuUniform<typeof CameraStruct>;

  constructor(root: TgpuRoot) {
    this.cameraUniform = root.createUniform(CameraStruct);
  }

  update() {
    const now = performance.now();
    const dt = (now - this.#lastTime) / 1000;
    this.#lastTime = now;

    const invViewMatrix = mat4.identity(d.mat4x4f());

    const manualOrientation =
      (store.get(cameraOrientationControlAtom) / 180) * Math.PI;
    const autoOrientation = (store.get(autoCameraOrientation) / 180) * Math.PI;

    const rad = store.get(autoRotateControlAtom)
      ? autoOrientation
      : manualOrientation;

    store.set(
      autoCameraOrientation,
      store.get(autoCameraOrientation) + store.get(autoRotateSpeedAtom) * dt,
    );

    // transforming the camera

    const zoom = store.get(cameraZoomControlAtom);

    mat4.rotateY(invViewMatrix, rad, invViewMatrix);
    mat4.translate(invViewMatrix, vec3.fromValues(0, 0, zoom), invViewMatrix);

    mat4.translate(
      invViewMatrix,
      vec3.fromValues(0, store.get(cameraYControlAtom), 0),
      invViewMatrix,
    );

    // calculating the 'regular' view matrix

    const viewMatrix = mat4.inverse(invViewMatrix, d.mat4x4f());

    const fovAngle = (store.get(cameraFovControlAtom) / 180) * Math.PI;

    this.cameraUniform.write({
      view_matrix: viewMatrix,
      inv_view_matrix: invViewMatrix,
      field_of_view: Math.tan(fovAngle / 2),
    });
  }
}
