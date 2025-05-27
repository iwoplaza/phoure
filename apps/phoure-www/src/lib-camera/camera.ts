import { accessViewportSize } from '@typegpu/common';
import tgpu, {
  type TgpuRoot,
  type TgpuBuffer,
  type UniformFlag,
} from 'typegpu';
import * as d from 'typegpu/data';
import { mat4, vec3 } from 'wgpu-matrix';

import {
  autoCameraOrientation,
  autoRotateControlAtom,
  autoRotateSpeedAtom,
  cameraFovControlAtom,
  cameraOrientationControlAtom,
  cameraYControlAtom,
  cameraZoomControlAtom,
} from 'src/lib/controlAtoms.ts';
import { store } from 'src/lib/store.ts';

export const CameraStruct = d.struct({
  view_matrix: d.mat4x4f,
  inv_view_matrix: d.mat4x4f,
  field_of_view: d.f32,
});

export const getCameraProps = tgpu['~unstable']
  .accessor(CameraStruct)
  .$name('getCameraProps');

export const constructRayPos = tgpu['~unstable']
  .fn(
    [],
    d.vec3f,
  )(/* wgsl */ `() -> vec3f {
    let camera = getCameraProps;
    return (camera.inv_view_matrix * vec4(0., 0., 0., 1.)).xyz;
  }`)
  .$uses({ getCameraProps });

export const constructRayDir = tgpu['~unstable']
  .fn(
    [d.vec2f],
    d.vec3f,
  )(/* wgsl */ `(coord: vec2f) -> vec3f {
    let camera = getCameraProps;
    let viewport_size = accessViewportSize;
    var view_coords = (coord - viewport_size / 2.) / viewport_size.y; // y in [-0.5, 0.5]
    view_coords = view_coords * camera.field_of_view;

    var view_ray_dir = vec3f(
      view_coords,
      -0.5,
    );
    view_ray_dir.y *= -1.;
    view_ray_dir = normalize(view_ray_dir);

    return (camera.inv_view_matrix * vec4(view_ray_dir, 0.)).xyz;
  }`)
  .$uses({ getCameraProps, accessViewportSize })
  .$name('construct_ray_dir');

export class Camera {
  public readonly cameraBuffer: TgpuBuffer<typeof CameraStruct> & UniformFlag;

  private _lastTime = Date.now();

  constructor(root: TgpuRoot) {
    this.cameraBuffer = root.createBuffer(CameraStruct).$usage('uniform');
  }

  update() {
    const now = Date.now();
    const dt = (now - this._lastTime) / 1000;
    this._lastTime = now;

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

    // Writing to buffer
    this.cameraBuffer.write({
      view_matrix: viewMatrix,
      inv_view_matrix: invViewMatrix,
      field_of_view: Math.tan(fovAngle / 2),
    });
  }
}
