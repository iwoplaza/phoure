import path from 'node:path';
import react from '@vitejs/plugin-react';
import typegpu from 'rollup-plugin-typegpu';
import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    typegpu({
      include: [/\.ts$/],
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
