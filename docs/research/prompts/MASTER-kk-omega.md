# MASTER prompt — KK-Ω research track (5D latent financial state)

Copy everything below this line to the coder agent. It drives the whole track;
for single-phase execution use the individual `KK*.md` prompts instead.

---

You are extending **FinDynamics** (`/Users/alexkou/Documents/github/findyn`) with a
Kaluza–Klein-**inspired** latent-state research module. The repo is a mature,
contract-driven, CI-enforced system. Your job is a *quarantined research
extension*, not a redesign.

## The hypothesis, stated honestly

FinDynamics represents an asset as `(P, v, a, j)` — price, and three derivatives
of the Kalman-filtered log price. This track tests whether adding a **latent
market-state coordinate Ω** and its dynamics:

    (P, v, a, j)  →  (P, v, a, j, Ω, Ω̇, Ω̈, C, K)

carries **statistically significant out-of-sample** information about future
returns, volatility, regime transitions and drawdowns.

`ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²` is **conceptual guidance for the
model's shape** — an observable block, a latent coordinate, and a coupling term
between them. It is not a claim about physics.

- Ω is a *latent market-state / liquidity-information coordinate*, inferred from
  observables. Do **not** hard-code it as "complexity".
- `C` (coupling) ≈ `∂²P/∂t∂Ω` — how price dynamics respond to latent-state motion.
- `K` (financial curvature) is a documented composite standing for regime
  instability.
- Never write "markets have a fifth dimension." Write "a fifth-dimensional
  latent-state representation inspired by Kaluza–Klein geometry was tested as a
  quantitative modelling hypothesis."
- **The model is allowed to fail.** A well-evidenced `REJECTED` is a successful
  outcome of this track. Do not tune until it passes.

## Architecture decisions — already made, do not relitigate

These are derived from the repo's actual contracts. An agent that "fixes" one of
them by loosening a validator has broken the system.

1. **Ω is NOT a sixth engine.** `core/contracts/vocab.py::ASSETS` is a closed
   5-tuple, mirrored in `serving/src/domain.ts` and guarded by
   `tests/test_domain.py`. `core/registry.py::register_engine` rejects any name
   outside it, and `AssetState`/`EngineOutput`/`DerivedFeature` all validate
   `asset in ASSETS`. **Do not touch ASSETS.** Do not call `register_engine`.
2. **Placement:** a new top layer `compute/findynamics/research/omega/`. It may
   import `findynamics.engines.equity` (that is the point — it reuses the Kalman
   / FFD / kinematics / RegimeDesign / HMM stack). Nothing else may import it.
3. **Config:** `compute/config/research/omega.yaml`, read by a research-local
   loader. **Never** create `config/engines/omega.yaml` —
   `core/config.py::load_engine_configs` raises `ConfigError` for any stem not in
   ASSETS, which would break every config load in the repo.
4. **Quarantine over flags.** The feature flag (`enabled: false`,
   `experimental: true`) is the second line of defence. The first is an
   import-linter contract that makes production dependence on this module
   structurally impossible, exactly like `Crypto is quarantined`.
5. **No writes to `engine_output` / `derived_features` / `asset_state` / D1.**
   No serving change, no migration, no cron. Research output is files:
   CSV/JSON under `compute/backtests/` and markdown under `docs/research/`.
6. **No new core dependencies.** Anything beyond numpy/pandas/scipy/scikit-learn
   (all already present) goes in a new optional extra
   `[project.optional-dependencies] research = [...]`.

## Phases — strictly in order

| Order | Prompt | Deliverable |
|---|---|---|
| 1 | `KK0-recon.md` | Reconnaissance + design note + scaffold + quarantine contract. No model code. |
| 2 | `KK1-omega-mvp.md` | Ω estimator, Ω̇/Ω̈, feature schema, unit + leakage tests |
| 3 | `KK2-coupling-curvature.md` | Coupling `C`, curvature `K`, diagnostics |
| 4 | `KK3-walk-forward.md` | Walk-forward comparison of models A/B/C/D/E, strict OOS |
| 5 | `KK4-diagnostics-ui.md` | Equity Dynamics Lab experimental panels (hidden by default) |
| 6 | `KK5-research-report.md` | The empirical report and the verdict |
| 7 | `KK6-ci-acceptance.md` | Determinism, leakage, production-unchanged proof, full CI |

Phase discipline (same as `docs/redesign/prompts/MASTER.md`):

- **One phase = one branch + one PR.** Branch names `research/kk0-recon`,
  `research/kk1-omega`, …
- A phase is done only when every item in its Acceptance section passes locally:
  ```
  cd compute && .venv/bin/ruff check . && .venv/bin/ruff format --check . \
    && .venv/bin/pytest && .venv/bin/lint-imports
  ```
  and, for KK4 only, `cd dashboard && npm run typecheck && npm run build`.
- **Stop after each phase.** Report deliverables, paste real test output (not
  claims), list deviations, then wait for approval.

## Global law (repeated because these are the rules agents break)

1. **No-lookahead is law** (`FINDYN_V1_SPEC.md` §14.1). Ω at date *t* is a
   function of data with `release_date <= t` and nothing else. No full-sample
   scaler, no full-sample PCA/HMM fit, no centred window, no negative shift.
   `tests/engines/equity/test_no_lookahead_guards.py` greps the source for
   lookahead-*shaped* code; you will add the same guard over `research/`.
2. **Determinism is mandatory.** Same data + same config + same model version ⇒
   byte-identical output. Follow `engines/equity/regime/hmm.py`: a fixed seed
   constant *and* `threadpool_limits(1)` around any sklearn/hmmlearn fit, because
   the seed alone does not buy reproducibility (issue #6, documented at length in
   that module). Use `math.fsum` for order-sensitive sums (commit bdf5e2d).
3. **Config over code.** Every threshold, window, weight and horizon lives in
   `config/research/omega.yaml`. A rule that exists only in Python cannot be
   recalibrated without a deploy.
4. **Never delete or weaken an existing test**, contract, or import-linter rule.
5. **Match the house style.** Frozen dataclasses, strict `__post_init__`
   validation, module docstrings that explain *why* a choice was made and what
   the rejected alternative did wrong, tests that assert on named historical
   windows.
6. Outputs are states, scores, quantiles and signals — never price targets,
   never trade commands.

Begin with `KK0-recon.md`.
