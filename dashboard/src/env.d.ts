/// <reference types="astro/client" />

interface ImportMetaEnv {
  /**
   * Origin of the FinDyn serving plane, e.g. `https://findyn.<account>.workers.dev`
   * or `http://localhost:8787`. The dashboard appends `/api/v1`.
   * Read only through src/lib/api.ts.
   */
  readonly PUBLIC_FINDYN_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
