import react from '@astrojs/react';
import starlight from '@astrojs/starlight';
import tailwind from '@astrojs/tailwind';
// @ts-check
import { defineConfig } from 'astro/config';
import typegpu from 'rollup-plugin-typegpu';

// https://astro.build/config
export default defineConfig({
  base: 'phoure',
  vite: {
    plugins: [typegpu({ include: [/\.ts$/] })],
  },
  redirects: {
    '/': {
      destination: '/phoure/play',
      status: 307,
    },
  },
  integrations: [
    starlight({
      title: 'phoure',
      social: {
        github: 'https://github.com/iwoplaza/phoure',
      },
      customCss: ['./src/tailwind.css'],
      sidebar: [
        {
          label: 'Guides',
          items: [
            // Each item here is one entry in the navigation menu.
            { label: 'Example Guide', slug: 'guides/example' },
          ],
        },
      ],
    }),
    tailwind({
      applyBaseStyles: true,
    }),
    react(),
  ],
});
