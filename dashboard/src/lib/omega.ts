/**
 * The KK-Ω research artifact, and the rules for reading it.
 *
 * This is a **static file**, not an API response. `/api/v1/assets/:asset/history`
 * reads the `engine_output` table and `EngineOutput.asset` is validated against
 * the five-member `ASSETS` vocabulary, so publishing Ω through that path would
 * write research rows into a production table. The compute plane writes
 * `public/research/omega.json` instead and the Lab reads it directly: no
 * migration, no route, no `serving/` change.
 *
 * Three consequences the loader is built around:
 *
 * - **A missing file is normal.** A clone that has not run
 *   `python -m jobs.omega_research --emit-lab-json` does not have one, and the
 *   page must build and render exactly as it does today. {@link loadOmega}
 *   returns `null` for every failure — absent, malformed, wrong shape — and
 *   never throws.
 * - **A gap is a message.** The coupling is undefined wherever `|ΔΩ|` fell below
 *   its denominator floor, and Ω itself does not start until its warm-up has
 *   passed. The artifact omits those dates rather than emitting zeros, and the
 *   panels draw a gap rather than a line across them. The Lab already has this
 *   vocabulary for `jerk_z` — "the engine declining to publish, not gaps in the
 *   market" — and it means the same thing here.
 * - **The verdict ships whatever it says.** If Q1 through Q5 all read FAILS the
 *   panel still renders and still prints them. A research surface that only
 *   displays successes is advertising.
 */

import type { LabSeries, PanelSpec, Sample } from './dynamics';

/** Where the compute job writes it. Relative to the site root. */
export const OMEGA_ARTIFACT_URL = '/research/omega.json';

export interface OmegaSpec {
  estimator: string;
  columns: string[];
  dropped: string[];
  sign_reference: string;
  explained_variance_ratio: number;
  fit_start: string;
  fit_end: string;
  loadings: Record<string, number>;
}

export interface OmegaArtifact {
  disclaimer: string;
  generated: string;
  model_version: string;
  config_hash: string;
  spec: OmegaSpec;
  verdicts: Record<string, string>;
  series: Record<string, [string, number][]>;
  diagnostics: Record<string, unknown>;
}

/** The five panels, in the order they stack below the existing four. */
export const OMEGA_GAP_MS = 10 * 24 * 60 * 60 * 1000;

export const OMEGA_PANELS = [
  {
    key: 'omega',
    title: 'Omega',
    note: 'latent market-state coordinate — PC1 of the standardized observable block, higher is more stressed by construction',
  },
  {
    key: 'omega_velocity',
    title: 'Omega Velocity',
    note: 'first derivative — filtered slope of Ω, annualized',
  },
  {
    key: 'omega_acceleration',
    title: 'Omega Acceleration',
    note: 'second derivative — change in Ω̇, per year squared',
  },
  {
    key: 'coupling',
    title: 'KK Coupling',
    note: 'C ≈ ∂²P/∂t∂Ω — expanding-window β₁ of velocity on Ω̇; gaps are dates the estimator declined',
  },
  {
    key: 'curvature',
    title: 'Financial Curvature',
    note: 'K — equal-weighted composite of Z(|a|), Z(|j|), Z(|Ω̇|), Z(|Ω̈|), Z(|C|)',
  },
] as const;

function isPoint(value: unknown): value is [string, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === 'string' &&
    typeof value[1] === 'number' &&
    Number.isFinite(value[1])
  );
}

/**
 * Shape-check the artifact before anything reads it.
 *
 * Structural rather than a schema library: the file is written by one known
 * producer and the failure this guards against is a stale artifact from an
 * older model version, not a hostile one. Anything unexpected returns `null`
 * and the section stays hidden, which is the same outcome as the file being
 * absent — and that is the correct outcome, because a half-rendered research
 * panel is worse than no research panel.
 */
export function parseOmega(raw: unknown): OmegaArtifact | null {
  if (!raw || typeof raw !== 'object') return null;
  const candidate = raw as Partial<OmegaArtifact>;
  if (typeof candidate.model_version !== 'string') return null;
  if (!candidate.spec || typeof candidate.spec !== 'object') return null;
  if (!Array.isArray(candidate.spec.columns)) return null;
  if (!candidate.series || typeof candidate.series !== 'object') return null;
  if (typeof candidate.disclaimer !== 'string') return null;

  const series: Record<string, [string, number][]> = {};
  for (const [key, points] of Object.entries(candidate.series)) {
    if (!Array.isArray(points)) return null;
    series[key] = points.filter(isPoint);
  }
  return { ...(candidate as OmegaArtifact), series };
}

/**
 * Fetch and parse the artifact. `null` on any failure, never a throw.
 *
 * The whole point: a clone that has never run the research job must build and
 * render byte-identically to one that has.
 */
export async function loadOmega(
  url: string = OMEGA_ARTIFACT_URL,
  fetcher: typeof fetch = fetch,
): Promise<OmegaArtifact | null> {
  try {
    const response = await fetcher(url, { headers: { accept: 'application/json' } });
    if (!response.ok) return null;
    return parseOmega(await response.json());
  } catch {
    return null;
  }
}

function toSamples(points: [string, number][]): Sample[] {
  const out: Sample[] = [];
  for (const [iso, value] of points) {
    const t = Date.parse(`${iso}T00:00:00Z`);
    if (Number.isFinite(t) && Number.isFinite(value)) out.push({ t, v: value });
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

function format3(v: number): string {
  return v.toFixed(3);
}

/**
 * How wide a hole has to be before the line breaks rather than bridging it.
 *
 * Ten calendar days. Wide enough to survive a holiday week and the artifact's
 * own decimation — which keeps one point per bucket and so can leave legitimate
 * multi-day steps — and narrow enough that a run of dates the coupling declined
 * to publish reads as the gap it is. Never interpolate across an undefined
 * coupling value: that is the one thing this section must not do.
 */
const MAX_GAP_MS = OMEGA_GAP_MS;

/**
 * The five `PanelSpec`s, ready to append to the existing stack.
 *
 * Stacked and never overlaid, for the reason the Lab's own docstring gives: Ω is
 * a z-scale, `C` is a ratio and `K` is a composite, and one y-axis would flatten
 * two of the three onto the baseline.
 *
 * A series the artifact does not carry produces **no panel**. `curvature` covers
 * only part of the out-of-sample path on the shipped configuration, so an empty
 * panel with an axis and no line would imply the data should be there.
 */
export function omegaPanels(artifact: OmegaArtifact): PanelSpec[] {
  const panels: PanelSpec[] = [];
  for (const spec of OMEGA_PANELS) {
    const points = artifact.series[spec.key];
    if (!points || points.length === 0) continue;
    const samples = toSamples(points);
    if (samples.length === 0) continue;

    const series: LabSeries = {
      key: spec.key,
      label: spec.title,
      className: `lab-line lab-line--${spec.key.replace(/_/g, '-')}`,
      samples,
      format: format3,
      maxGap: MAX_GAP_MS,
      diagnostics: {
        available: points.length,
        returned: samples.length,
        missing: points.length - samples.length,
        truncated: false,
        decimated: null,
        first: points[0]?.[0] ?? null,
        last: points.at(-1)?.[0] ?? null,
      },
    };
    panels.push({
      key: `omega-${spec.key}`,
      title: spec.title,
      note: spec.note,
      series: [series],
      height: 150,
      zeroLine: true,
    });
  }
  return panels;
}

/** Human-readable summary of which columns Ω was built from. */
export function describeSpec(spec: OmegaSpec): string {
  const used = spec.columns.join(', ');
  const dropped = spec.dropped.length ? spec.dropped.join(', ') : 'none';
  return `${spec.columns.length} column(s): ${used}. Dropped for insufficient history: ${dropped}.`;
}
