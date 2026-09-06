import { defineConfig } from 'vitest/config';
import typegpu from 'unplugin-typegpu/vite';

export default defineConfig({
  plugins: [typegpu()],
  resolve: {
    alias: { src: new URL('./apps/phoure-www/src', import.meta.url).pathname },
  },
});
