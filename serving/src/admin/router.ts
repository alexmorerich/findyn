import { Hono } from 'hono';
import type { Env } from '../types';
import { verifyHmac } from './hmac';
import { PayloadError, applyWriteBack, validatePayload } from './writeback';
import { ArtifactError, getArtifact, listVersions, putArtifact } from './artifacts';

type Vars = { rawBody: string };

/**
 * Private write-back surface for the Python compute plane (FINDYN_V1_SPEC.md §6).
 * Cloudflare Workers cannot run the scientific Python stack, so all provider and
 * model output arrives here as HMAC-signed JSON rather than being computed in-Worker.
 */
export const admin = new Hono<{ Bindings: Env; Variables: Vars }>();

admin.use('*', async (c, next) => {
  const secret = c.env.ADMIN_HMAC_SECRET;
  if (!secret) {
    return c.json({ error: 'admin_disabled', message: 'ADMIN_HMAC_SECRET is not configured' }, 503);
  }

  const signature = c.req.header('x-findyn-signature');
  const timestamp = c.req.header('x-findyn-timestamp');
  if (!signature || !timestamp) {
    return c.json({ error: 'unauthorized', message: 'missing signature headers' }, 401);
  }

  // Signature covers the exact bytes, so the body is read once here and reused
  // downstream rather than re-parsed from the request.
  const body = await c.req.text();
  const ok = await verifyHmac({ secret, signature, timestamp, body });
  if (!ok) {
    return c.json({ error: 'unauthorized', message: 'invalid or expired signature' }, 401);
  }

  c.set('rawBody', body);
  await next();
});

admin.post('/results', async (c) => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(c.get('rawBody'));
  } catch {
    return c.json({ error: 'bad_request', message: 'body is not valid JSON' }, 400);
  }

  try {
    const payload = validatePayload(parsed);
    const written = await applyWriteBack(c.env, payload);
    return c.json({
      ok: true,
      model_version: payload.model_version,
      generated_at: payload.generated_at,
      written,
    });
  } catch (err) {
    if (err instanceof PayloadError) {
      return c.json({ error: 'invalid_payload', message: err.message }, 422);
    }
    throw err;
  }
});

// ---------------------------------------------------------------------------
// Fitted-model storage (§6). Behind the same HMAC as the write-back: the
// compute plane is the only thing that reads or writes these, and a fitted
// model is not public data.
// ---------------------------------------------------------------------------

function artifactFailure(err: unknown) {
  if (err instanceof ArtifactError) {
    return { body: { error: 'artifact_error', message: err.message }, status: err.status };
  }
  throw err;
}

/**
 * Store a fitted model under an exact version. Write-once.
 *
 * The version travels in the path rather than being read out of the body, so a
 * payload whose `model_version` disagrees with where it is being filed is
 * rejected here rather than becoming a mislabelled artifact.
 */
admin.put('/artifacts/:name/:version', async (c) => {
  const body = c.get('rawBody');
  const version = decodeURIComponent(c.req.param('version'));

  let declared: { model_version?: string; fit_date?: string };
  try {
    declared = JSON.parse(body) as { model_version?: string; fit_date?: string };
  } catch {
    return c.json({ error: 'bad_request', message: 'artifact body must be JSON' }, 400);
  }
  if (declared.model_version && declared.model_version !== version) {
    return c.json(
      {
        error: 'bad_request',
        message: `path version ${version} disagrees with the body's model_version ${declared.model_version}`,
      },
      400,
    );
  }

  // The fit date rides in the body beside model_version rather than in the path.
  // Both are properties of the artifact, and an artifact filed under a date its
  // own contents disagree with is the same class of mistake as one filed under
  // the wrong version — which this endpoint already refuses.
  const fit = declared.fit_date;
  if (!fit) {
    return c.json(
      {
        error: 'bad_request',
        message:
          'artifact body must carry fit_date (YYYY-MM-DD); a fitted model with no fit date ' +
          'cannot be told apart from the next months refit of the same specification',
      },
      400,
    );
  }

  try {
    const result = await putArtifact(c.env, c.req.param('name'), version, fit, body);
    return c.json({ ok: true, ...result }, result.created ? 201 : 200);
  } catch (err) {
    const failure = artifactFailure(err);
    return c.json(failure.body, failure.status);
  }
});

/** Fetch one artifact by version, or `latest` to resolve the pointer. */
admin.get('/artifacts/:name/:version', async (c) => {
  try {
    const { version, fit, body } = await getArtifact(
      c.env,
      c.req.param('name'),
      decodeURIComponent(c.req.param('version')),
      c.req.query('fit'),
    );
    // Both resolved halves ride in headers so the body stays exactly the bytes
    // that were stored — a caller re-signing or hashing it must see it unchanged.
    return new Response(body, {
      headers: {
        'content-type': 'application/json',
        'x-findyn-model-version': version,
        ...(fit ? { 'x-findyn-model-fit': fit } : {}),
      },
    });
  } catch (err) {
    const failure = artifactFailure(err);
    return c.json(failure.body, failure.status);
  }
});

admin.get('/artifacts/:name', async (c) => {
  try {
    return c.json({
      name: c.req.param('name'),
      versions: await listVersions(c.env, c.req.param('name')),
    });
  } catch (err) {
    const failure = artifactFailure(err);
    return c.json(failure.body, failure.status);
  }
});
