# KK4 — Equity Dynamics Lab research panels (hidden by default)

Copy everything below this line to the coder agent. Requires KK3 merged.

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
and `docs/redesign/04-ui-plan.md`.

Branch: `research/kk4-lab-panels`.

## Task

Add an **experimental** Ω section to the Equity Dynamics Lab so the temporal
relationship between price dynamics, latent dynamics, coupling and regime
transitions can be looked at. Visual investigation only — the panel must not
imply causality.

## 1. The hard constraint on where the data comes from

`/api/v1/assets/:asset/history?metric=` reads the `engine_output` table, and
`EngineOutput.asset` is validated against `ASSETS`. **Publishing Ω through that
path would write research rows into a production table.** Do not do it, and do
not add a migration, an API route, or a `serving/` change of any kind.

Instead:

- `jobs/omega_research.py` gains `--emit-lab-json`, writing a single decimated
  artifact to `dashboard/public/research/omega.json` (committed).
- Schema: `{ "generated": iso, "model_version": str, "spec": {...},
  "verdict": str, "series": { "omega": [[iso, num], ...], "omega_velocity": [...],
  "omega_acceleration": [...], "coupling": [...], "curvature": [...] },
  "diagnostics": {...} }`.
- Decimate to the same ceiling the API uses for a Max-range request (see
  `serving/src/api/assets.ts` — match its cap rather than inventing one), so the
  page's memory profile is unchanged.
- The file is a **research artifact**, not an API. Say so in the JSON itself with
  a `"disclaimer"` field, and in the module docstring.

## 2. The panels

Extend `dashboard/src/scripts/equity-lab.ts`. Follow its existing design rules,
which are stated in its own module docstring — read them first:

- **Stacked, never overlaid**, sharing one x-axis. Ω is a z-scale, `C` is a
  ratio, `K` is a composite; one y-axis would flatten two of them.
- Reuse `DynamicsChart` / `PanelSpec` from `dashboard/src/lib/dynamics.ts`.
  Crosshair and time axis must be **synchronized with the existing four panels**
  — that synchronization is the entire point of the section.
- New panels, in this order below the existing Price / Velocity / Acceleration /
  Jerk stack: `Omega`, `Omega Velocity`, `Omega Acceleration`, `KK Coupling`,
  `Financial Curvature`.
- **Absent data is named.** Ω starts after its warm-up and `C` is undefined
  wherever `|ΔΩ|` fell below the floor. The Lab already has the vocabulary for
  this ("the engine declining to publish, not gaps in the market") — reuse it.
  Never interpolate across an undefined coupling value.

## 3. The banner and the flag

- A **persistent** `EXPERIMENTAL — RESEARCH ONLY` banner at the top of the
  section, in the style of the crypto page's banner. It must state, in one
  sentence a reader can act on: *these series are a research hypothesis, are not
  part of any published state or allocation, and no causal relationship is
  implied by their alignment on the time axis.*
- The section is **hidden by default**, gated on a build-time flag
  (`PUBLIC_OMEGA_LAB`, default unset/false in `.env.example` and in CI). When the
  flag is off, `equity.astro` renders byte-identically to today — assert this in
  the phase report with a diff of the built HTML.
- No change to the home page, the Engines panel, the ribbon, or any other page.

## 4. The reading, stated on screen

Below the panels, a short definitions block — the Lab already does this for jerk
and the RII. It must give, in plain language and with the formula visible:

- what Ω is (a latent coordinate inferred from N observable columns, listed);
- which columns were used and which were dropped, read from the artifact's spec;
- the sign convention (Ω increases with stress, by construction);
- what `C` and `K` are, and the fact that `K`'s weights are equal-by-rule;
- the current **verdict** from the KK5 report, rendered verbatim from the JSON.
  If the verdict is `REJECTED` or `NO INCREMENTAL INFORMATION`, the panel still
  ships and still says so. A research surface that only displays successes is
  advertising.

## 5. Tests

- Dashboard `npm run typecheck` and `npm run build` green.
- A unit test that the artifact loader handles a **missing** `omega.json` by
  hiding the section, not by throwing — a clone that has not run the research job
  must still build.
- A test that undefined (`null`) coupling values create a gap rather than a line
  segment.
- Compute-side: a test that `--emit-lab-json` output validates against the schema
  and that regenerating it twice from the same inputs is byte-identical.

## Acceptance

- [ ] Five new synchronized panels in the Lab, sharing the existing crosshair.
- [ ] Section hidden with the flag off; built HTML unchanged — paste the diff.
- [ ] Persistent experimental banner with the no-causality sentence.
- [ ] Definitions block renders the spec and the verdict from the artifact.
- [ ] `omega.json` regenerates byte-identically.
- [ ] `cd dashboard && npm run typecheck && npm run build` green.
- [ ] `cd compute && ruff check . && ruff format --check . && pytest && lint-imports` green.
- [ ] **No `serving/` change, no migration, no new API route, no cron.**
