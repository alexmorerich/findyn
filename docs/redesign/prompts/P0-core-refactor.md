# P0 — Restructure compute plane into the `findynamics` package

Copy everything below this line to the coder agent.

---

You are working in the `findyn` repo. Before writing any code, read:
`docs/redesign/01-target-architecture.md`, `docs/redesign/02-migration-map.md`,
`docs/redesign/03-contracts.md`, and skim `FINDYN_V1_SPEC.md` §14.1
(no-lookahead rules — global law).

## Task

Phase 0 of the FinDynamics redesign: restructure `compute/` into the
`findynamics` package with core contracts and an engine registry.
**Zero behavior change. No new models. No new endpoints.**

## Steps

1. **Package rename and move** (use `git mv` to preserve history), exactly per
   the table in `02-migration-map.md` §1:
   - `compute/findyn/` → `compute/findynamics/` with subpackages
     `core/`, `core/contracts/`, `data/`, `data/providers/`, `factors/`,
     `engines/` (empty subpackages `money rates equity gold crypto` with
     `__init__.py` only), `portfolio/` (empty), `backtest/` (empty).
   - Providers, `pit.py`, `quality.py` → `data/`; `config.py` → `core/`.
   - Split `domain.py`: shared vocab → `core/contracts/vocab.py`;
     equity-specific constants → `engines/equity/domain.py`;
     FORCES tuple → `factors/definitions.py`.
   - Update `pyproject.toml` (name `findynamics`, packages, entry points) and
     every import in `jobs/` and `tests/`.

2. **Create core contracts** in `findynamics/core/` exactly as specified in
   `docs/redesign/03-contracts.md` §1–§3: `contracts/state.py`
   (`FactorState`, `WorldState`, `Signal`, `AssetState`), `engine.py`
   (`AssetEngine` ABC), `registry.py` (engine registry + the moved provider
   registry). `PITAccessor` is a thin wrapper class over `data.pit.pit_join`.
   Write unit tests for the registry (register/duplicate/lookup/enable-flags)
   and for `AssetState` invariants.

3. **Restructure `compute/config/series.yaml`**: rename top-level `forces:` to
   `factors:`; move the S&P500 `price:` block under `engines.equity.series`.
   Update `core/config.py`'s strict validator and its tests in the same
   commit. Add `compute/config/engines/` with one minimal yaml per engine
   containing `enabled: false` (all engines off — none exist yet).

4. **Import-linter**: add the three contracts from `03-contracts.md` §5 to
   `compute/pyproject.toml`, add `import-linter` to dev deps, and add
   `lint-imports` to the compute job in `.github/workflows/ci.yml`.

5. **Serving-plane touch-ups only**:
   - Repoint the `domain.ts` ↔ Python drift test to
     `findynamics/engines/equity/domain.py`.
   - Add `ASSETS` vocabulary to `serving/src/domain.ts`.
   - Nothing else in `serving/` changes.

6. **Jobs**: update imports only. Orchestration rewrite is P1.

## Constraints

- No test may be deleted; move tests alongside their code.
- No new runtime dependencies except `import-linter` (dev).
- All existing behavior identical: `jobs/backfill.py` etc. still run.

## Acceptance

- `cd compute && .venv/bin/pytest` — all pre-existing tests pass from new
  paths, plus new registry/contract tests.
- `.venv/bin/ruff check . && .venv/bin/ruff format --check .` clean.
- `lint-imports` passes.
- `cd serving && npm run typecheck && npm test` — all pass.
- `git log --follow` shows history preserved for moved files.
