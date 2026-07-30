// @ts-check
import { defineConfig } from 'astro/config';

/**
 * FinDyn dashboard — FINDYN_V1_SPEC.md §16.
 *
 * Fully static: the build emits `dist/` as plain HTML/CSS/JS with no server
 * runtime, so the same artifact deploys to Cloudflare Pages or to the Worker's
 * static-asset binding (`serving/wrangler.jsonc` -> assets.directory).
 *
 * All data is fetched client-side at runtime. The API is live and the site is
 * not, so nothing may be baked in at build time — see src/lib/api.ts.
 */
export default defineConfig({
  output: 'static',
  trailingSlash: 'ignore',
  build: {
    // Emit `/data/index.html` rather than `/data.html`: Workers static assets
    // and Pages both resolve directory-style URLs, and it keeps the nav hrefs
    // extension-free.
    format: 'directory',
    inlineStylesheets: 'auto',
  },
  devToolbar: { enabled: false },
});
