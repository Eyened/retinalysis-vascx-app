import { svelte } from '@sveltejs/vite-plugin-svelte';
import { webcrypto } from 'node:crypto';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

if (!globalThis.crypto) {
  (globalThis as unknown as { crypto: Crypto }).crypto = webcrypto as unknown as Crypto;
}

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? (process.env.GITHUB_REPOSITORY ? '/retinalysis-vascx-app/' : '/'),
  plugins: [
    svelte(),
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'Retinalysis VascX',
        short_name: 'VascX',
        description: 'Browser-only retinal vascular parameter extraction.',
        theme_color: '#0f766e',
        background_color: '#f8fafc',
        display: 'standalone',
        start_url: '.',
        icons: [
          {
            src: 'icons/vascx-icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          },
          {
            src: 'icons/vascx-icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,json,png,wasm}'],
        maximumFileSizeToCacheInBytes: 32 * 1024 * 1024
      }
    })
  ],
  resolve: {
    alias: {
      $lib: fileURLToPath(new URL('./src/lib', import.meta.url))
    }
  },
  worker: {
    format: 'es'
  },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.ts']
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    allowedHosts: ["eyened-supergpu"]
  }
});
