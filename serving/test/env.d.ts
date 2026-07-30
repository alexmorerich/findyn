import type { D1Migration } from '@cloudflare/vitest-pool-workers';

declare global {
  namespace Cloudflare {
    interface Env {
      /** Injected by vitest.config.ts; applied to the test D1 in test/setup.ts. */
      TEST_MIGRATIONS: D1Migration[];
    }
  }
}

export {};
