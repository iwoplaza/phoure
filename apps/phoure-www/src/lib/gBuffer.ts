import type { TgpuRoot } from 'typegpu';

export class GBuffer {
  private quarterATexture: GPUTexture; // used by Mender (odd frames)
  private quarterBTexture: GPUTexture; // used by Mender (even frames)
  private upscaledTexture: GPUTexture; // used by Mender
  private rawRenderATexture: GPUTexture; // a render before any post-processing (odd frames)
  private rawRenderBTexture: GPUTexture; // a render before any post-processing (even frames)
  private auxTexture: GPUTexture;

  private _even = false;
  private _quarterAView: GPUTextureView;
  private _quarterBView: GPUTextureView;
  private _rawRenderAView: GPUTextureView;
  private _rawRenderBView: GPUTextureView;

  readonly upscaledView: GPUTextureView;
  readonly auxView: GPUTextureView;

  quarterSize: [number, number];

  constructor(
    root: TgpuRoot,
    private _size: [number, number],
  ) {
    this.quarterSize = [_size[0] >> 2, _size[1] >> 2];

    this.quarterATexture = root.device.createTexture({
      size: this.quarterSize,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
      format: 'rgba8unorm',
    });

    this.quarterBTexture = root.device.createTexture({
      size: this.quarterSize,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
      format: 'rgba8unorm',
    });

    this.upscaledTexture = root.device.createTexture({
      size: this.size,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      format: 'rgba8unorm',
    });

    this.rawRenderATexture = root.device.createTexture({
      size: this.size,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
      format: 'rgba8unorm',
    });

    this.rawRenderBTexture = root.device.createTexture({
      size: this.size,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
      format: 'rgba8unorm',
    });

    this.auxTexture = root.device.createTexture({
      size: _size,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
      format: 'rgba16float',
    });

    this._quarterAView = this.quarterATexture.createView();
    this._quarterBView = this.quarterBTexture.createView();
    this.upscaledView = this.upscaledTexture.createView();
    this._rawRenderAView = this.rawRenderATexture.createView();
    this._rawRenderBView = this.rawRenderBTexture.createView();
    this.auxView = this.auxTexture.createView();
  }

  flip() {
    this._even = !this._even;
  }

  get inQuarterView(): GPUTextureView {
    return this._even ? this._quarterBView : this._quarterAView;
  }

  get outQuarterView(): GPUTextureView {
    return this._even ? this._quarterAView : this._quarterBView;
  }

  get inRawRenderView(): GPUTextureView {
    return this._even ? this._rawRenderBView : this._rawRenderAView;
  }

  get outRawRenderView(): GPUTextureView {
    return this._even ? this._rawRenderAView : this._rawRenderBView;
  }

  get size() {
    return this._size;
  }

  updateSize(size: [number, number]) {
    // TODO: Recreate textures.
    this._size = size;
  }
}
