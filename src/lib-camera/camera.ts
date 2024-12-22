import { mat4, vec3 } from 'wgpu-matrix';
import tgpu, {
  type TgpuFn,
  type ExperimentalTgpuRoot,
  type TgpuBuffer,
  type Uniform,
} from 'typegpu/experimental';
import * as d from 'typegpu/data';
import { store } from '@/store';
import {
  autoRotateControlAtom,
  cameraFovControlAtom,
  cameraOrientationControlAtom,
  cameraYControlAtom,
  cameraZoomControlAtom,
} from '@/controlAtoms';
import { getViewportSizeSlot } from '../GameEngine/commonSlots';

export const CameraStruct = d.struct({
  view_matrix: d.mat4x4f,
  inv_view_matrix: d.mat4x4f,
  field_of_view: d.f32,
});

export const getCameraProps = tgpu
  .slot<TgpuFn<[], typeof CameraStruct>>()
  .$name('getCameraProps');

export const constructRayPos = tgpu
  .fn([], d.vec3f)
  .does(/* wgsl */ `() -> vec3f {
    let camera = getCameraProps();
    return (camera.inv_view_matrix * vec4(0., 0., 0., 1.)).xyz;
  }`)
  .$uses({ getCameraProps });

export const constructRayDir = tgpu
  .fn([d.vec2f], d.vec3f)
  .does(/* wgsl */ `(coord: vec2f) -> vec3f {
    let camera = getCameraProps();
    let viewport_size = getViewportSize();
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
  .$uses({ getCameraProps, getViewportSize: getViewportSizeSlot })
  .$name('construct_ray_dir');

export class Camera {
  public readonly cameraBuffer: TgpuBuffer<typeof CameraStruct> & Uniform;

  constructor(root: ExperimentalTgpuRoot) {
    this.cameraBuffer = root.createBuffer(CameraStruct).$usage('uniform');
  }

  update() {
    // const upVector = vec3.fromValues(0, 1, 0);

    const invViewMatrix = mat4.identity(d.mat4x4f());

    // const rad = 2.5;
    const rad = store.get(autoRotateControlAtom)
      ? Math.PI * (Date.now() / 5000)
      : (store.get(cameraOrientationControlAtom) / 180) * Math.PI;

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
