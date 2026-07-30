import { applyD1Migrations, env } from 'cloudflare:test';

// Apply the real migration files to the isolated test database before any suite runs.
await applyD1Migrations(env.DB, env.TEST_MIGRATIONS);
