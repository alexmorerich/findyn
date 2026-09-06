# FinDynamics research track — KK-Ω (latent 5D financial state)

An **experimental, quarantined** research extension testing whether a latent
market-state coordinate Ω and its dynamics add out-of-sample information to the
existing `(P, v, a, j)` representation:

    (P, v, a, j)  →  (P, v, a, j, Ω, Ω̇, Ω̈, C, K)

The `ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²` form of Kaluza–Klein geometry is
used as **guidance for the model's shape** — an observable block, a latent
coordinate, and a coupling between them. Nothing here claims markets have a
physical fifth dimension. The hypothesis is falsifiable and is allowed to fail.

## Conclusion

**The track is complete. Ω does not carry information about forward returns
beyond what the equity engine already publishes.**

Over 5,407 out-of-sample rows across 258 monthly rebalances, no Ω arm beat the
RII control arm on forward return at any horizon. One result did survive: arm B —
price kinematics plus Ω — improves out-of-sample R² on forward realized
volatility over that control by +0.137 / +0.247 / +0.337 at 5 / 21 / 63 trading
days, with HAC t-statistics of 4.54 / 4.03 / 3.14, BH-adjusted q-values below
0.005, positive in five sub-periods of five, and stable under every sensitivity
including dropping `liquidity_stress` from Ω's own column block. The coupling `C`
and the curvature `K` did not earn their cost.

The shuffled-target control is clean on Q1–Q4 and found two defects in the
pre-registration itself: Q3 and Q5 were written without significance tests. The
pre-registered verdict vocabulary also turns out not to cover this outcome — Q1
failing with exactly one of Q2/Q3/Q4 passing — so [the
report](./kk-omega-report.md) reaches **no verdict** rather than choosing the
nearest label. These are associations measured out of sample; no causal
relationship is implied, and none was tested.

Nothing from this track is reachable from production. It sits above every other
layer, the `Research is quarantined` import-linter contract forbids `portfolio`,
`engines`, `factors`, `core` and `data` from importing it, and
`config/research/omega.yaml` ships `enabled: false`. KK6 verified that the
equity engine's write-back payload is unchanged across the whole track: the set
of fields that differ between the first commit and the last is exactly the set
that differs between two runs of the *same* commit.

## Documents

| File | Content |
|---|---|
| [`kk-omega-design.md`](./kk-omega-design.md) | **The pre-registration.** Hypothesis, the KK mapping table, Ω's inputs, the estimator choice, the sign problem, the anti-lookahead and determinism protocols, the falsification plan (§8), the verdict vocabulary (§9), and what would make us delete the module (§10). Written before any number existed. |
| [`kk-omega-report.md`](./kk-omega-report.md) | **The result.** §A–H: formulation as implemented, data, the six-arm comparison, performance, predictive statistics, failure analysis with the sensitivity table, the verdict against the quoted rule, and reproduction. Regenerated from committed CSVs by `python -m jobs.omega_research --report`. |
| [`kk-omega-walkforward.md`](./kk-omega-walkforward.md) | The KK3 walk-forward narrative — what was refit per window, what was frozen, and the three bugs the control surfaced. |
| [`kk-omega-diagnostics.md`](./kk-omega-diagnostics.md) | The KK2 in-sample construction diagnostics: loadings, coupling panels, curvature decomposition, redundancy against published quantities, stationarity. |
| [`../follow-ups/kk-omega.md`](../follow-ups/kk-omega.md) | What to delete, and the cheapest experiment that would most change the conclusion. Post-hoc by construction. |

## Where the code lives

| Path | Content |
|---|---|
| `compute/findynamics/research/omega/` | The module. A top layer, **not** a sixth engine. |
| `compute/config/research/omega.yaml` | `enabled: false` and every parameter |
| `compute/tests/research/omega/` | 250 tests: unit, leakage, determinism, quarantine, language |
| `compute/jobs/omega_research.py` | Deliberate-run job. No cron, no `[project.scripts]` entry. |
| `compute/backtests/omega/` | Walk-forward CSV artifacts; every number in the report traces to one |
| `dashboard/public/research/omega.json` | The Lab artifact, behind `PUBLIC_OMEGA_LAB` |

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
