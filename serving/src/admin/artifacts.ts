import type { Env } from '../types';

/**
 * Fitted-model storage in R2 (FINDYN_V1_SPEC.md §6, "R2: … model artifacts").
 *
 * The problem this solves is a production blocker rather than a nicety. A refit
 * runs monthly in GitHub Actions and a prediction runs daily in a *different*
 * GitHub Actions container; `compute/artifacts/` is gitignored and the runner is
 * ephemeral, so whatever the refit fitted was deleted minutes later and the
 * daily run has never once seen it. The equity engine's honest response to a
 * missing fit is to publish no state at all, which is exactly what production
 * would have done forever.
 *
 * Three properties, in the order they matter:
 *
 * **Immutable.** A `(version, fit)` pair is written once. `PUT` over an existing
 * pair with different bytes is a 409, not an overwrite. A fitted model is the
 * thing a published `model_version` refers to; if it can change under that name,
 * every backtest of that version silently becomes a backtest of something else
 * and there is no way to notice.
 *
 * **Addressed by (model_version, fit date).** The key is
 * `<name>/<version>/<fit>.json`. Both halves are load-bearing and it took a
 * production failure to see why: `model_version` names the model *specification*
 * — which features, which estimator, which vocabulary — but a monthly
 * expanding-window refit re-estimates every parameter, so the artifact's content
 * legitimately changes each month while the specification does not. Keyed on the
 * version alone, immutability meant the second refit of any engine 409'd
 * forever; the first one to run twice (`rates-1.0.0`) did exactly that. The
 * checkable claim is not "version 1.1.0 is these bytes" — it never could be —
 * it is "the state published on this date came from *this* fit of 1.1.0", and
 * that is what this key says.
 *
 * **Pointed at by a mutable pointer.** `<name>/latest.json` names the newest
 * version *and fit*. That indirection is the only mutable part, and it has to
 * be: a daily run that has just started needs to discover the current fit
 * without being told. Immutability lives on the payloads, currency lives on the
 * pointer, and conflating the two is what makes "which model produced this
 * number" an unanswerable question.
 */

/** Version strings appear in an R2 key, so they are constrained rather than trusted. */
const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const NAME_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
/** The fit date. A calendar date sorts lexically, which is what `latest` needs. */
const FIT_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export class ArtifactError extends Error {
  constructor(
    message: string,
    readonly status: 400 | 404 | 409 | 503,
  ) {
    super(message);
  }
}

export interface ArtifactPointer {
  name: string;
  version: string;
  /** Absent on pointers written before the fit date was part of the key. */
  fit?: string;
  updated_at: string;
}

/** One stored fit: which specification, and when it was fitted. */
export interface ArtifactRef {
  version: string;
  fit: string | null;
}

function assertName(name: string): string {
  if (!NAME_PATTERN.test(name)) {
    throw new ArtifactError(`invalid artifact name: ${name}`, 400);
  }
  return name;
}

function assertVersion(version: string): string {
  if (!VERSION_PATTERN.test(version)) {
    throw new ArtifactError(`invalid artifact version: ${version}`, 400);
  }
  return version;
}

function assertFit(fit: string): string {
  if (!FIT_PATTERN.test(fit)) {
    throw new ArtifactError(`invalid artifact fit date: ${fit}; expected YYYY-MM-DD`, 400);
  }
  return fit;
}

/**
 * `+` is legal in a model_version (`equity-1.0.0+cal.yahoo_gspc`) and is the one
 * character that would be ambiguous in a URL path, so keys encode it. Decoding
 * happens at the route boundary, not here.
 */
function key(name: string, version: string, fit: string): string {
  return `artifacts/${name}/${encodeURIComponent(version)}/${fit}.json`;
}

/**
 * Where artifacts lived before the fit date joined the key.
 *
 * Read-only, and never written to again. It exists so the daily runs already in
 * flight against a deployed worker keep resolving their model through a deploy
 * rather than falling back to "no fit" — which, for the equity engine, means
 * publishing no state at all until the next monthly refit.
 */
function legacyKey(name: string, version: string): string {
  return `artifacts/${name}/${encodeURIComponent(version)}.json`;
}

function pointerKey(name: string): string {
  return `artifacts/${name}/latest.json`;
}

function bucket(env: Env): R2Bucket {
  const store = env.ARCHIVE;
  if (!store) {
    throw new ArtifactError('R2 bucket ARCHIVE is not bound', 503);
  }
  return store;
}

/**
 * Store one fitted model under `(version, fit)`. Refuses to change either.
 *
 * A re-run of the same refit producing byte-identical content is treated as
 * success rather than a conflict — retries after a network failure are routine
 * and should not need a human. Different content under the same version *and*
 * the same fit date is the case this exists to reject: two different models
 * claiming to be the same fit of the same specification is the one situation
 * where a published `model_version` stops meaning anything.
 */
export async function putArtifact(
  env: Env,
  name: string,
  version: string,
  fit: string,
  body: string,
): Promise<{ created: boolean; version: string; fit: string }> {
  assertName(name);
  assertVersion(version);
  assertFit(fit);

  const store = bucket(env);
  const existing = await store.get(key(name, version, fit));
  if (existing) {
    const current = await existing.text();
    if (current === body) return { created: false, version, fit };
    throw new ArtifactError(
      `artifact ${name}/${version} fitted ${fit} already exists with different content; ` +
        'fitted models are immutable — a re-fit on a later date is a new fit, not an edit',
      409,
    );
  }

  await store.put(key(name, version, fit), body, {
    httpMetadata: { contentType: 'application/json' },
    customMetadata: { name, version, fit, written_at: new Date().toISOString() },
  });

  // The pointer moves only forward. Two refits landing out of order — a backfill
  // of an older date after a current one — must not make the older fit current,
  // and without this check the last writer would win.
  const pointer = await getPointer(env, name);
  const newer =
    !pointer ||
    pointer.version < version ||
    (pointer.version === version && (pointer.fit ?? '') < fit);
  if (newer) {
    const next: ArtifactPointer = {
      name,
      version,
      fit,
      updated_at: new Date().toISOString(),
    };
    await store.put(pointerKey(name), JSON.stringify(next), {
      httpMetadata: { contentType: 'application/json' },
    });
  }

  return { created: true, version, fit };
}

/** What `latest` currently names, or null when nothing has been stored. */
export async function getPointer(env: Env, name: string): Promise<ArtifactPointer | null> {
  assertName(name);
  const object = await bucket(env).get(pointerKey(name));
  if (!object) return null;
  try {
    return JSON.parse(await object.text()) as ArtifactPointer;
  } catch {
    return null;
  }
}

/** Every fit stored under one version, oldest first. */
async function fitsFor(env: Env, name: string, version: string): Promise<string[]> {
  const listed = await bucket(env).list({
    prefix: `artifacts/${name}/${encodeURIComponent(version)}/`,
  });
  return listed.objects
    .map((o) => o.key.split('/').pop() ?? '')
    .filter((f) => f.endsWith('.json'))
    .map((f) => f.replace(/\.json$/, ''))
    .filter((f) => FIT_PATTERN.test(f))
    .sort();
}

/**
 * Fetch one artifact.
 *
 * `version` of `latest` resolves through the pointer. An explicit version with
 * no `fit` resolves to that version's newest fit — which is what a caller
 * pinning a *specification* wants — and an explicit fit resolves exactly, which
 * is what replaying a published state needs.
 *
 * Both are reported back alongside the body: a caller that asked for `latest`
 * has to learn what it actually got and stamp that on whatever it publishes, or
 * "the latest model" is a claim nobody can check later.
 */
export async function getArtifact(
  env: Env,
  name: string,
  version: string,
  fit?: string,
): Promise<{ version: string; fit: string | null; body: string }> {
  assertName(name);

  let resolvedVersion = version;
  let resolvedFit = fit;
  if (version === 'latest') {
    const pointer = await getPointer(env, name);
    if (!pointer) {
      throw new ArtifactError(`no artifact has been stored for ${name}`, 404);
    }
    resolvedVersion = pointer.version;
    resolvedFit = resolvedFit ?? pointer.fit;
  }
  assertVersion(resolvedVersion);
  if (resolvedFit !== undefined) assertFit(resolvedFit);

  if (resolvedFit === undefined) {
    resolvedFit = (await fitsFor(env, name, resolvedVersion)).at(-1);
  }

  const store = bucket(env);
  if (resolvedFit !== undefined) {
    const object = await store.get(key(name, resolvedVersion, resolvedFit));
    if (object) {
      return { version: resolvedVersion, fit: resolvedFit, body: await object.text() };
    }
  }

  // Nothing under the new layout. Fall back to the flat key so a worker deployed
  // ahead of the next refit still serves the model that is actually there.
  const legacy = await store.get(legacyKey(name, resolvedVersion));
  if (legacy) {
    return { version: resolvedVersion, fit: null, body: await legacy.text() };
  }

  throw new ArtifactError(
    `artifact ${name}/${resolvedVersion}${resolvedFit ? ` fitted ${resolvedFit}` : ''} not found`,
    404,
  );
}

/** Every stored (version, fit) of one artifact, oldest first. */
export async function listVersions(env: Env, name: string): Promise<ArtifactRef[]> {
  assertName(name);
  const listed = await bucket(env).list({ prefix: `artifacts/${name}/` });

  const refs: ArtifactRef[] = [];
  for (const object of listed.objects) {
    const rest = object.key.slice(`artifacts/${name}/`.length);
    if (!rest.endsWith('.json') || rest === 'latest.json') continue;
    const parts = rest.replace(/\.json$/, '').split('/');
    if (parts.length === 2) {
      refs.push({ version: decodeURIComponent(parts[0]!), fit: parts[1]! });
    } else if (parts.length === 1) {
      // A pre-fit-date artifact. Listed rather than hidden — it is a real model
      // that real published states point at.
      refs.push({ version: decodeURIComponent(parts[0]!), fit: null });
    }
  }
  return refs.sort((a, b) =>
    a.version === b.version
      ? (a.fit ?? '').localeCompare(b.fit ?? '')
      : a.version.localeCompare(b.version),
  );
}
