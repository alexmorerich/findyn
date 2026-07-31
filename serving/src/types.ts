/**
 * Runtime bindings for the FinDyn serving plane.
 *
 * Binding shapes (DB, CACHE, ARCHIVE, vars) are generated from wrangler.jsonc into
 * worker-configuration.d.ts by `npm run types` — never hand-edit them, or the
 * type will drift from the deployed configuration. Only secrets are declared
 * here, because `wrangler secret put` values are invisible to the generator.
 *
 * See FINDYN_V1_SPEC.md §6.
 */

declare global {
  namespace Cloudflare {
    interface Env {
      // Optional at the type level so a missing key degrades the affected
      // provider instead of crashing the Worker (§14.2).
      FRED_API_KEY?: string;
      BLS_API_KEY?: string;
      BEA_API_KEY?: string;
      /** Shared secret for HMAC-signed write-back from the Python compute plane. */
      ADMIN_HMAC_SECRET?: string;
    }
  }
}

export type Env = Cloudflare.Env;
