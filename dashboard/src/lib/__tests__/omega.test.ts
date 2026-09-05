/**
 * The two behaviours a type checker cannot see.
 *
 * Both are about *absence*: an artifact that is not there, and a date the
 * estimator declined to publish. Each has an obvious wrong implementation that
 * type-checks perfectly — throwing on a 404, and drawing a straight line across
 * a hole — and each would be found only by someone looking at the page.
 */

import { describe, expect, it } from 'vitest';
import { OMEGA_GAP_MS, loadOmega, omegaPanels, parseOmega, type OmegaArtifact } from '../omega';

const SPEC = {
  estimator: 'pca',
  columns: ['realized_vol', 'vol_of_vol'],
  dropped: ['credit_velocity'],
  sign_reference: 'realized_vol',
  explained_variance_ratio: 0.437,
  fit_start: '2000-02-03',
  fit_end: '2026-07-30',
  loadings: { realized_vol: 0.5, vol_of_vol: 0.39 },
};

function artifact(series: Record<string, [string, number][]>): OmegaArtifact {
  return {
    disclaimer: 'Research artifact, not an API response.',
    generated: '2026-07-30',
    model_version: 'omega-0.1.0',
    config_hash: 'abc123',
    spec: SPEC,
    verdicts: { Q1: 'FAILS', Q2: 'PASSES (arm B)' },
    series,
    diagnostics: {},
  };
}

describe('loadOmega', () => {
  it('returns null when the artifact is absent, and never throws', async () => {
    // The state of every clone that has not run `jobs.omega_research`. The page
    // must build and render exactly as it did before the research track existed.
    const missing = await loadOmega(
      '/research/omega.json',
      async () => new Response('not found', { status: 404 }),
    );
    expect(missing).toBeNull();
  });

  it('returns null when the fetch itself fails', async () => {
    const offline = await loadOmega('/research/omega.json', async () => {
      throw new Error('network down');
    });
    expect(offline).toBeNull();
  });

  it('returns null on a malformed artifact rather than a half-rendered panel', async () => {
    const wrong = await loadOmega(
      '/research/omega.json',
      async () => new Response(JSON.stringify({ model_version: 'omega-0.1.0' }), { status: 200 }),
    );
    expect(wrong).toBeNull();
    expect(parseOmega(null)).toBeNull();
    expect(parseOmega({ model_version: 1 })).toBeNull();
  });

  it('parses a well-formed artifact and drops non-finite points', () => {
    const parsed = parseOmega({
      ...artifact({ omega: [['2020-01-02', 1.5]] }),
      series: {
        omega: [['2020-01-02', 1.5], ['2020-01-03', null], ['bad'], [1, 2]],
      },
    });
    expect(parsed).not.toBeNull();
    expect(parsed!.series.omega).toEqual([['2020-01-02', 1.5]]);
  });
});

describe('omegaPanels', () => {
  it('declares a gap tolerance so an undefined coupling breaks the line', () => {
    // The estimator emits NaN below its denominator floor and the artifact omits
    // those dates entirely, so the samples either side of a decline are far
    // apart. Without `maxGap` the chart would draw one straight segment across
    // the hole and assert a value nobody computed.
    const panels = omegaPanels(
      artifact({
        coupling: [
          ['2020-01-02', 0.1],
          ['2020-01-03', 0.2],
          // Three months the estimator declined to publish.
          ['2020-04-06', 0.3],
        ],
      }),
    );

    expect(panels).toHaveLength(1);
    const series = panels[0]!.series[0]!;
    expect(series.maxGap).toBe(OMEGA_GAP_MS);

    const gaps = series.samples.slice(1).map((s, i) => s.t - series.samples[i]!.t);
    // One ordinary step and one hole wider than the tolerance — so the renderer
    // starts a new subpath at the third point instead of joining it.
    expect(gaps.filter((g) => g > OMEGA_GAP_MS)).toHaveLength(1);
  });

  it('omits a panel entirely when the artifact carries no points for it', () => {
    // `curvature` covers only part of the out-of-sample path on the shipped
    // configuration. An empty panel with an axis and no line would imply the
    // data should have been there.
    const panels = omegaPanels(artifact({ omega: [['2020-01-02', 1.0]], curvature: [] }));
    expect(panels.map((p) => p.key)).toEqual(['omega-omega']);
  });

  it('keeps the panels in their documented stacking order', () => {
    const panels = omegaPanels(
      artifact({
        curvature: [['2020-01-02', 1.0]],
        omega: [['2020-01-02', 1.0]],
        coupling: [['2020-01-02', 1.0]],
      }),
    );
    expect(panels.map((p) => p.key)).toEqual(['omega-omega', 'omega-coupling', 'omega-curvature']);
  });
});
