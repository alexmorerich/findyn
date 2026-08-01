import type { Env } from '../types';

/**
 * Monte Carlo path archives in R2 (FINDYN_V1_SPEC.md §11).
 *
 * Separate from `artifacts.ts` because the two answer different questions and
 * have different lifetimes. An artifact is a *model* — one per fit, immutable,
 * the thing a published `model_version` refers to. A simulation archive is a
 * *run* — one per day per asset, the outcome of pointing that model at one
 * day's information set. Filing them together would mean either the model churns
 * daily or the run is overwritten nightly, and both destroy the property the
 * other one needs.
 *
 * Key: `simulations/<asset>/<as_of>/<model_version>.json`. The model version is
 * the leaf rather than a path component so a day that was simulated by two
 * models — a refit landing mid-day — keeps both, and neither can claim to be
 * "the" simulation for that date without saying which model it came from.
 *
 * **Not served publicly.** `/api/v1/simulate` returns the run's parameters and
 * points at `/forecast` for the distribution. Ten thousand paths per horizon is
 * a denial of service with extra steps, and the quantiles are the answer to
 * every question a caller of a public API is actually asking.
 */

const NAME_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;

export class SimulationError extends Error {
  constructor(
    message: string,
    readonly status: 400 | 404 | 409 | 503,
  ) {
    super(message);
  }
}

function bucket(env: Env): R2Bucket {
  const store = env.ARCHIVE;
  if (!store) {
    throw new SimulationError('R2 bucket ARCHIVE is not bound', 503);
  }
  return store;
}

function assertParts(asset: string, asOf: string, version: string): void {
  if (!NAME_PATTERN.test(asset)) {
    throw new SimulationError(`invalid asset: ${asset}`, 400);
  }
  if (!DATE_PATTERN.test(asOf)) {
    throw new SimulationError(`invalid as_of: ${asOf}; expected YYYY-MM-DD`, 400);
  }
  if (!VERSION_PATTERN.test(version)) {
    throw new SimulationError(`invalid model_version: ${version}`, 400);
  }
}

function key(asset: string, asOf: string, version: string): string {
  return `simulations/${asset}/${asOf}/${encodeURIComponent(version)}.json`;
}

/**
 * Archive one run's per-path outcomes.
 *
 * Immutable for the same reason artifacts are: a simulation is evidence about
 * what a model said on a date, and evidence that can be edited afterwards is
 * not evidence. An identical re-write is a no-op so a retried job needs no
 * human; different bytes under the same (asset, date, model) is a conflict.
 */
export async function putSimulation(
  env: Env,
  asset: string,
  asOf: string,
  version: string,
  body: string,
): Promise<{ created: boolean; key: string }> {
  assertParts(asset, asOf, version);
  const store = bucket(env);
  const objectKey = key(asset, asOf, version);

  const existing = await store.get(objectKey);
  if (existing) {
    if ((await existing.text()) === body) return { created: false, key: objectKey };
    throw new SimulationError(
      `simulation ${asset}/${asOf}/${version} already exists with different content; ` +
        'a run is a record of what the model said that day, not a document to revise',
      409,
    );
  }

  await store.put(objectKey, body, {
    httpMetadata: { contentType: 'application/json' },
    customMetadata: { asset, as_of: asOf, model_version: version },
  });
  return { created: true, key: objectKey };
}

/** Read one archived run back, for offline analysis. */
export async function getSimulation(
  env: Env,
  asset: string,
  asOf: string,
  version: string,
): Promise<string> {
  assertParts(asset, asOf, version);
  const object = await bucket(env).get(key(asset, asOf, version));
  if (!object) {
    throw new SimulationError(`simulation ${asset}/${asOf}/${version} not found`, 404);
  }
  return object.text();
}

/** Every archived run for one asset, oldest first. */
export async function listSimulations(
  env: Env,
  asset: string,
): Promise<{ as_of: string; model_version: string }[]> {
  if (!NAME_PATTERN.test(asset)) {
    throw new SimulationError(`invalid asset: ${asset}`, 400);
  }
  const listed = await bucket(env).list({ prefix: `simulations/${asset}/` });
  return listed.objects
    .map((o) =>
      o.key
        .slice(`simulations/${asset}/`.length)
        .replace(/\.json$/, '')
        .split('/'),
    )
    .filter((parts) => parts.length === 2 && DATE_PATTERN.test(parts[0]!))
    .map((parts) => ({ as_of: parts[0]!, model_version: decodeURIComponent(parts[1]!) }))
    .sort((a, b) =>
      a.as_of === b.as_of
        ? a.model_version.localeCompare(b.model_version)
        : a.as_of.localeCompare(b.as_of),
    );
}
