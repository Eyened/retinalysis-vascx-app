import { svelte } from '@sveltejs/vite-plugin-svelte';
import { webcrypto } from 'node:crypto';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

if (!globalThis.crypto) {
  (globalThis as unknown as { crypto: Crypto }).crypto = webcrypto as unknown as Crypto;
}

export default defineConfig({
  base: process.env.GITHUB_REPOSITORY ? '/rtnls_vascx_app/' : '/',
  plugins: [
    svelte(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['models/manifest.json'],
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
            src: 'icons/pwa-192.svg',
            sizes: '192x192',
            type: 'image/svg+xml',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,json,wasm}'],
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
