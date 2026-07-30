#!/usr/bin/env node
/**
 * Config drift guard: every cron declared in wrangler.jsonc must map to a job in
 * src/ingest/scheduled.ts, and every registered job must have a live cron.
 * A cron with no handler silently burns an invocation; a handler with no cron
 * never runs. Both are invisible at runtime, so they are checked in CI.
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

/** Strip // and /* *\/ comments and trailing commas, respecting string literals. */
function parseJsonc(text) {
  let out = '';
  let inString = false;
  let inLine = false;
  let inBlock = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];

    if (inLine) {
      if (ch === '\n') { inLine = false; out += ch; }
      continue;
    }
    if (inBlock) {
      if (ch === '*' && next === '/') { inBlock = false; i++; }
      continue;
    }
    if (inString) {
      out += ch;
      if (ch === '\\') { out += next ?? ''; i++; }
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') { inString = true; out += ch; continue; }
    if (ch === '/' && next === '/') { inLine = true; i++; continue; }
    if (ch === '/' && next === '*') { inBlock = true; i++; continue; }
    out += ch;
  }

  return JSON.parse(out.replace(/,(\s*[}\]])/g, '$1'));
}

const root = new URL('../', import.meta.url);
const config = parseJsonc(await readFile(fileURLToPath(new URL('wrangler.jsonc', root)), 'utf8'));
const scheduled = await readFile(fileURLToPath(new URL('src/ingest/scheduled.ts', root)), 'utf8');

const declared = config.triggers?.crons ?? [];
const registered = [...scheduled.matchAll(/^\s*'([^']+)':\s*'[a-z_]+',$/gm)].map((m) => m[1]);

const missingHandler = declared.filter((c) => !registered.includes(c));
const missingCron = registered.filter((c) => !declared.includes(c));

if (declared.length === 0) {
  console.error('✗ no cron triggers declared in wrangler.jsonc');
  process.exit(1);
}
if (missingHandler.length || missingCron.length) {
  for (const c of missingHandler) console.error(`✗ cron declared but not handled: ${c}`);
  for (const c of missingCron) console.error(`✗ job registered but cron not declared: ${c}`);
  process.exit(1);
}

console.log(`✓ ${declared.length} cron triggers mapped to handlers`);
