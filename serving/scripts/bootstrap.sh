#!/usr/bin/env bash
# Create the Cloudflare resources FinDyn needs and print the ids to paste into
# wrangler.jsonc. Run once per account. Requires `wrangler login` first.
#
# FINDYN_V1_SPEC.md §6.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating D1 database 'findyn'"
npx wrangler d1 create findyn || echo "    (already exists — reusing)"

echo
echo "==> Creating KV namespace 'CACHE'"
npx wrangler kv namespace create CACHE || echo "    (already exists — reusing)"

echo
echo "==> Creating R2 bucket 'findyn-archive'"
npx wrangler r2 bucket create findyn-archive || echo "    (already exists — reusing)"

cat <<'EOF'

------------------------------------------------------------------
Next steps
------------------------------------------------------------------
1. Paste the printed database_id and KV id into wrangler.jsonc.
2. Regenerate binding types:      npm run types
3. Apply the schema:              npm run db:migrate:remote
4. Set secrets (one per line):
     npx wrangler secret put FRED_API_KEY
     npx wrangler secret put BLS_API_KEY
     npx wrangler secret put BEA_API_KEY
     npx wrangler secret put ADMIN_HMAC_SECRET     # openssl rand -hex 32
5. Deploy:                        npm run deploy

The same ADMIN_HMAC_SECRET must be set as a GitHub Actions secret, alongside
FINDYN_ADMIN_URL (https://<worker-host>/admin/v1/results), so the compute plane
can write results back.
EOF
