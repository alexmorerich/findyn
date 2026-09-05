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

**KK0 through KK4 delivered.** The design note is pre-registered at
[`kk-omega-design.md`](./kk-omega-design.md); the walk-forward results are at
[`kk-omega-walkforward.md`](./kk-omega-walkforward.md); the in-sample
construction diagnostics are at
[`kk-omega-diagnostics.md`](./kk-omega-diagnostics.md).

### The out-of-sample answer so far

258 monthly rebalances, 5,407 out-of-sample rows, 2005-02 → 2026-07. Six arms,
four horizons, one Benjamini–Hochberg family of 70 p-values.

| Question | Real run | Shuffled control |
|---|---|---|
| **Q1** forward return | FAILS | FAILS |
| **Q2** forward volatility | **PASSES (arm B)** | FAILS |
| **Q3** regime transition | FAILS | FAILS |
| **Q4** coupling | FAILS | FAILS |
| **Q5** decision rule | PASSES (arm B, 3/5) | **PASSES — the rule is not a test** |

**Ω says nothing about returns and something real about volatility.** Arm B
(`P,v,a,j` + Ω) beats the RII control arm A′ on forward realized volatility by
ΔR² of +0.137 / +0.247 / +0.337 at h = 5 / 21 / 63, HAC t of 4.54 / 4.03 / 3.14,
q of 0.00002 / 0.00018 / 0.00417 — positive in **5 of 5** sub-periods. That is a
claim about volatility only; §1.2 of the design note is explicit that returns
and volatility are different claims.

The shuffled-target control clears Q1–Q4 (max |OOS R²| 0.040, smallest q 0.197).
It also *passes* Q5, which is not a leak — it is a defect in the pre-registered
Q5 rule, which has no significance test. KK5 must read Q5's PASS as
uninformative.

### The two open items

1. **`curvature.zscore_min_years` is still your call.** At the shipped 10.0, `K`
   covers 2,067 of 5,407 out-of-sample rows and arm E therefore ran without it.
   At 5.0 `K` would reach 2008.
2. **Q5's rule needs a significance test** before KK5 can use it. A HAC t-test
   on arm B's daily excess return over A′ gives t = 0.37, p = 0.71.

### The Lab panels (KK4)

Five synchronized panels — Ω, Ω̇, Ω̈, `C`, `K` — share the existing chart's
x-axis, zoom and crosshair, behind the build-time flag `PUBLIC_OMEGA_LAB`.
**Unset by default**: with the flag off the built `/equity` markup is
byte-identical to the pre-KK4 build and no research code reaches the bundle. The
data is a committed static artifact at `dashboard/public/research/omega.json`,
never an API route and never an `engine_output` row.

KK5 is next: the report and the verdict.

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
