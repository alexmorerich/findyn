import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { cloudflareTest, readD1Migrations } from '@cloudflare/vitest-pool-workers';

// Migrations are read at config time and applied per test worker in test/setup.ts.
// This is what makes "migrations apply cleanly" (M0 acceptance) verifiable in CI
// without a Cloudflare account.
const migrationsDir = fileURLToPath(new URL('./migrations', import.meta.url));
const migrations = await readD1Migrations(migrationsDir);

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: './wrangler.jsonc' },
      miniflare: {
        bindings: { TEST_MIGRATIONS: migrations },
      },
    }),
  ],
  test: {
    setupFiles: ['./test/setup.ts'],
  },
});
