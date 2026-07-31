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
 * **Immutable.** A version is written once. `PUT` of an existing key is a 409,
 * not an overwrite. A fitted model is the thing a published `model_version`
 * refers to; if it can change under that name, every backtest of that version
 * silently becomes a backtest of something else and there is no way to notice.
 *
 * **Addressed by model_version.** The key is `<name>/<version>.json`, so
 * "load the model that produced this state" is a lookup rather than an
 * assumption about which file is current.
 *
 * **Pointed at by a mutable pointer.** `<name>/latest.json` names the newest
 * version. That indirection is the only mutable part, and it has to be: a daily
 * run that has just started needs to discover the current version without being
 * told. Immutability lives on the payloads, currency lives on the pointer, and
 * conflating the two is what makes "which model produced this number" an
 * unanswerable question.
 */

/** Version strings appear in an R2 key, so they are constrained rather than trusted. */
const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const NAME_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;

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
  updated_at: string;
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

/**
 * `+` is legal in a model_version (`equity-1.0.0+cal.yahoo_gspc`) and is the one
 * character that would be ambiguous in a URL path, so keys encode it. Decoding
 * happens at the route boundary, not here.
 */
function key(name: string, version: string): string {
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
 * Store one fitted model. Refuses to overwrite an existing version.
 *
 * A re-run of the same refit producing byte-identical content is treated as
 * success rather than a conflict — retries after a network failure are routine
 * and should not need a human. Different content under the same version is the
 * case this exists to reject.
 */
export async function putArtifact(
  env: Env,
  name: string,
  version: string,
  body: string,
): Promise<{ created: boolean; version: string }> {
  assertName(name);
  assertVersion(version);

  const store = bucket(env);
  const existing = await store.get(key(name, version));
  if (existing) {
    const current = await existing.text();
    if (current === body) return { created: false, version };
    throw new ArtifactError(
      `artifact ${name}/${version} already exists with different content; ` +
        'fitted models are immutable — bump the model version instead',
      409,
    );
  }

  await store.put(key(name, version), body, {
    httpMetadata: { contentType: 'application/json' },
    customMetadata: { name, version, written_at: new Date().toISOString() },
  });

  const pointer: ArtifactPointer = {
    name,
    version,
    updated_at: new Date().toISOString(),
  };
  await store.put(pointerKey(name), JSON.stringify(pointer), {
    httpMetadata: { contentType: 'application/json' },
  });

  return { created: true, version };
}

/** The version `latest` currently names, or null when nothing has been stored. */
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

/**
 * Fetch one artifact. `version` of `latest` resolves through the pointer.
 *
 * Resolution is reported back alongside the body, so a caller that asked for
 * `latest` learns which version it actually got and can stamp that on whatever
 * it publishes — otherwise "the latest model" is a claim nobody can check later.
 */
export async function getArtifact(
  env: Env,
  name: string,
  version: string,
): Promise<{ version: string; body: string }> {
  assertName(name);

  let resolved = version;
  if (version === 'latest') {
    const pointer = await getPointer(env, name);
    if (!pointer) {
      throw new ArtifactError(`no artifact has been stored for ${name}`, 404);
    }
    resolved = pointer.version;
  }
  assertVersion(resolved);

  const object = await bucket(env).get(key(name, resolved));
  if (!object) {
    throw new ArtifactError(`artifact ${name}/${resolved} not found`, 404);
  }
  return { version: resolved, body: await object.text() };
}

/** Every stored version of one artifact, newest write first. */
export async function listVersions(env: Env, name: string): Promise<string[]> {
  assertName(name);
  const listed = await bucket(env).list({ prefix: `artifacts/${name}/` });
  return listed.objects
    .map((o) => o.key.split('/').pop() ?? '')
    .filter((f) => f.endsWith('.json') && f !== 'latest.json')
    .map((f) => decodeURIComponent(f.replace(/\.json$/, '')))
    .sort();
}
