// @ts-check
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';
import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';
import typegpu from 'unplugin-typegpu/vite';

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
      social: [
        {
          icon: 'github',
          href: 'https://github.com/iwoplaza/phoure',
          label: 'GitHub',
        },
      ],
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
