/**
 * Copy the KK-Ω research artifact into the build, but only behind the flag.
 *
 * The artifact used to live in `public/`, which Astro copies wholesale into
 * `dist/`. That meant a **production** build — with `PUBLIC_OMEGA_LAB` unset and
 * every Lab panel compiled out — still published 672 KB of research JSON at
 * `/research/omega.json`. More than three times the entire JavaScript bundle,
 * fetched by nothing, on a site whose own documentation says no research output
 * reaches production.
 *
 * So the file sits outside `public/` and this script copies it in after the
 * build, only when the flag that renders the panels is set. Flag off, and
 * `dist/research/` does not exist at all.
 *
 * Run from `npm run build`, after `astro build`, because Astro clears `dist/`
 * at the start of a build and anything copied before it would be deleted.
 */

import { copyFile, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(root, 'research', 'omega.json');
const target = join(root, 'dist', 'research', 'omega.json');

if (!process.env.PUBLIC_OMEGA_LAB) {
  console.log('[research] PUBLIC_OMEGA_LAB unset — omega.json not published');
  process.exit(0);
}

try {
  await stat(source);
} catch {
  // A clone that has never run `python -m jobs.omega_research --emit-lab-json`
  // has no artifact. The loader already treats a missing file as normal, so
  // this is a note rather than a failure.
  console.log('[research] no research/omega.json to publish — skipping');
  process.exit(0);
}

await mkdir(dirname(target), { recursive: true });
await copyFile(source, target);
const { size } = await stat(target);
console.log(`[research] published omega.json (${(size / 1024).toFixed(0)} KB)`);
