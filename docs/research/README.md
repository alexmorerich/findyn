# FinDynamics research track — KK-Ω (latent 5D financial state)

An **experimental, quarantined** research extension testing whether a latent
market-state coordinate Ω and its dynamics add out-of-sample information to the
existing `(P, v, a, j)` representation:

    (P, v, a, j)  →  (P, v, a, j, Ω, Ω̇, Ω̈, C, K)

The `ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²` form of Kaluza–Klein geometry is
used as **guidance for the model's shape** — an observable block, a latent
coordinate, and a coupling between them. Nothing here claims markets have a
physical fifth dimension. The hypothesis is falsifiable and is allowed to fail.

## Status

**KK0, KK1 and KK2 delivered.** The design note is pre-registered at
[`kk-omega-design.md`](./kk-omega-design.md); the `Research is quarantined`
import-linter contract is green and demonstrated to fail on a planted production
import. The current in-sample diagnostics are at
[`kk-omega-diagnostics.md`](./kk-omega-diagnostics.md).

Ω, Ω̇, Ω̈, three coupling estimators and the curvature composite `K` are computed
from the committed fixture:

| Quantity | Span on the shipped config | Note |
|---|---|---|
| Ω | 2000-02-03 → 2026-07-30 | 8 columns; `credit_velocity` dropped at 10.9% coverage; PC1 explains 43.7% |
| `C` (regression) | 2003-02-07 → 2026-07-29 | β1 on Ω̇, Newey–West t = +2.36 on the velocity specification |
| `K` | 2013-02-11 → 2026-07-29 | five terms, equal weights — **2008 is outside it**, see below |

**Nothing here is a result.** Every number is from a whole-record fit and is
therefore in-sample. KK3's walk-forward is the only thing that may report one.

### Two findings KK3 has to carry

1. **No latent quantity is a re-derivation of what the engine publishes.** The
   largest rank correlation against any published quantity is `K` vs `rii_jerk`
   at +0.72; `K` vs the RII composite is **+0.42**, well below the 0.9
   redundancy threshold. That is a statement about redundancy, not usefulness.
2. **`K` reaches back only to 2013 under the shipped defaults**, because the
   coupling term warms up twice — 504 observations of expanding regression, then
   a ten-year expanding z-baseline on top. Shortening
   `curvature.zscore_min_years` to 5.0 moves the start to 2008-02 and to 3.0
   moves it to 2006-02. That is a KK3/KK5 decision, deliberately not taken here.

KK3 is next: the walk-forward comparison of arms A / A′ / B / C / D / E.

## Structure of the delivered module (target)

| Path | Content |
|---|---|
| `compute/findynamics/research/omega/` | The module. A new top layer, **not** a sixth engine. |
| `compute/config/research/omega.yaml` | Flag (`enabled: false`) and every parameter |
| `compute/tests/research/omega/` | Unit, leakage, determinism, quarantine tests |
| `compute/jobs/omega_research.py` | Deliberate-run research job. No cron. |
| `compute/backtests/omega/` | Walk-forward CSV artifacts |
| `docs/research/kk-omega-design.md` | The pre-registered design + falsification plan (KK0) |
| `docs/research/kk-omega-report.md` | The empirical result and verdict (KK5) |

## Prompts (execution order)

| Prompt | Phase | Deliverable |
|---|---|---|
| [`prompts/MASTER-kk-omega.md`](./prompts/MASTER-kk-omega.md) | — | Driver: architecture decisions, global law, phase table |
| [`prompts/KK0-recon.md`](./prompts/KK0-recon.md) | KK0 | Recon, design note, scaffold, import quarantine. No model code. |
| [`prompts/KK1-omega-mvp.md`](./prompts/KK1-omega-mvp.md) | KK1 | Ω estimator (PCA), Ω̇/Ω̈, features, sign pinning, leakage tests |
| [`prompts/KK2-coupling-curvature.md`](./prompts/KK2-coupling-curvature.md) | KK2 | Coupling `C` (3 estimators), curvature `K`, diagnostics |
| [`prompts/KK3-walk-forward.md`](./prompts/KK3-walk-forward.md) | KK3 | Walk-forward arms A/A′/B/C/D/E, Q1–Q5, shuffled-target control |
| [`prompts/KK4-diagnostics-ui.md`](./prompts/KK4-diagnostics-ui.md) | KK4 | Equity Dynamics Lab panels, hidden behind a build flag |
| [`prompts/KK5-research-report.md`](./prompts/KK5-research-report.md) | KK5 | Report §A–H and the pre-registered verdict |
| [`prompts/KK6-ci-acceptance.md`](./prompts/KK6-ci-acceptance.md) | KK6 | Determinism, production-unchanged proof, CI, closeout |

## The four rules that make this safe

1. **Ω is not an asset.** `core/contracts/vocab.py::ASSETS` is a closed 5-tuple
   mirrored in `serving/src/domain.ts`. The module never registers an engine,
   never writes to `engine_output` / `derived_features` / `asset_state`, and
   never adds a migration.
2. **Quarantine before flags.** An import-linter contract makes production
   dependence structurally impossible; `enabled: false` is the second line.
3. **The RII is the control arm.** `engines/equity/rii.py` already publishes a
   composite built from nearly the same observables, so "Ω beats price alone" is
   not a result. "Ω beats the RII" would be.
4. **Pre-registration.** The decision rules are written in the design note before
   any number exists, and the report quotes them verbatim above the verdict.
