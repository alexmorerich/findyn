# Migration Map — current repo → FinDynamics

Ground truth as of this design: M0 complete, M1a in progress (providers,
quality, resilience, dashboard scaffold, migration 0003 uncommitted). Almost
no model code exists yet — the migration is a *restructure*, not a rewrite.

## 1. Compute plane — file-by-file

| Current | Target | Action |
|---|---|---|
| `compute/findyn/providers/base.py` | `findynamics/data/providers/base.py` | Move unchanged. `Observation`/`SeriesMetadata` are already the canonical data contract |
| `compute/findyn/providers/{fred,shiller,stooq,bls,bea,mock}.py` | `findynamics/data/providers/…` | Move unchanged |
| `compute/findyn/providers/registry.py` | `findynamics/core/registry.py` (provider half) | Merge into a single registry module with two registries: providers + engines |
| `compute/findyn/providers/resilience.py` | `findynamics/data/providers/resilience.py` | Move unchanged |
| `compute/findyn/pit.py` | `findynamics/data/pit.py` | Move unchanged. Remains the **sole** gateway to `macro_series` for every engine |
| `compute/findyn/quality.py` | `findynamics/data/quality.py` | Move unchanged |
| `compute/findyn/config.py` | `findynamics/core/config.py` | Move; extend to also load `config/engines/*.yaml` |
| `compute/findyn/domain.py` | split | Shared vocabulary (QUANTILES, HORIZONS, frequencies) → `core/contracts/vocab.py`. Equity-specific (REGIMES, KINEMATIC_FEATURES, SHOCK_CLASSES) → `engines/equity/domain.py`. FORCES → `factors/definitions.py` as the initial factor set |
| `compute/config/series.yaml` | `compute/config/series.yaml` | Keep path. Restructure top level: `meta` / `factors` (was `forces`) / `engines.<name>.series` for engine-private series (e.g. the S&P500 price block moves under `engines.equity`) |
| `compute/jobs/*.py` | `compute/jobs/*.py` | Keep as thin CLIs; rewrite internals in P0/P1 to orchestrate via the engine registry |
| `compute/tests/*` | `compute/tests/…` mirrored | Move with the code; imports updated; no test deleted |

Package rename: `findyn` → `findynamics` in `pyproject.toml`
(`findyn-compute` → `findynamics`), entry points updated, egg-info
regenerated. Flat layout is kept (no `src/` churn).

## 2. Serving plane

| Current | Change |
|---|---|
| `src/domain.ts` | Add `ASSETS = ['money','rates','equity','gold','crypto']`; keep the v1 vocabulary (it mirrors `engines/equity/domain.py` — the drift test now points there) |
| `src/api/router.ts` | Mount `assets.ts` and `factors.ts` routers; `/forces` stays as an alias of `/factors` |
| `src/admin/writeback.ts` | Add `asset_state` / `engine_output` batch payload types under the existing HMAC scheme |
| `migrations/0004_multi_asset.sql` | New: `asset_state`, `engine_output`; add `asset TEXT DEFAULT 'SPX'` to the v1 output tables that lack it |
| `src/ingest/*`, `src/providers/*`, `src/lib/*` | Untouched |

## 3. What is deliberately NOT done

- No table renames (`force_scores` stays; code calls them factors).
- No repo split — monorepo with enforced boundaries, per the FinDynamics doc.
- No serving-plane rewrite; Hono/D1/KV/R2/HMAC all keep their v1 shape.
- `FINDYN_V1_SPEC.md` is not rewritten; it is re-scoped as the specification
  of `engines/equity` (a note at its top is enough).
- No new ML dependencies in P0.

## 4. Sequencing and risk

1. **P0 is a pure-move refactor** — the only risky step is import churn.
   Mitigation: `git mv` for history, run full pytest + ruff after, and land
   import-linter in the same change so drift is impossible afterwards.
2. Migration 0004 is additive-only; it cannot break existing queries.
3. `series.yaml` restructure changes `config.py`'s schema — the strict
   validator and its tests move and extend in the same commit.
4. The serving `domain.ts` drift test must be repointed to
   `engines/equity/domain.py` in P0, or CI breaks on the move.
