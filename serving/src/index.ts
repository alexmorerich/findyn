import { Hono } from 'hono';
import type { Env } from './types';
import { api } from './api/router';
import { admin } from './admin/router';
import { runScheduled } from './ingest/scheduled';
import { DISCLAIMER } from './domain';

const app = new Hono<{ Bindings: Env }>();

app.route('/api/v1', api);
app.route('/admin/v1', admin);

app.get('/', (c) =>
  c.json({
    service: 'FinDyn v1.0 — S&P500 Dynamic State Engine',
    spec: 'FINDYN_V1_SPEC.md',
    api: '/api/v1',
    health: '/api/v1/health',
    disclaimer: DISCLAIMER,
  }),
);

app.notFound((c) => c.json({ error: 'not_found', path: c.req.path }, 404));

app.onError((err, c) => {
  console.error('unhandled error', err);
  return c.json({ error: 'internal_error', message: 'request failed' }, 500);
});

export default {
  fetch: app.fetch,
  async scheduled(event: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(runScheduled(event, env));
  },
} satisfies ExportedHandler<Env>;
