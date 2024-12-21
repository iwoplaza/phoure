import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import typegpu from 'rollup-plugin-typegpu';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), typegpu()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
