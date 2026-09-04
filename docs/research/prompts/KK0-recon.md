# KK0 — Reconnaissance, design note, scaffold, quarantine

Copy everything below this line to the coder agent. First phase of the KK-Ω
research track. **No model code in this phase.**

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
first — its architecture decisions are binding.

Branch: `research/kk0-recon`.

## 1. Read before writing anything

Read these files and be able to answer, in your phase report, the question in
brackets. Do not skim — later phases assume you know the answers.

| File | Question you must answer |
|---|---|
| `docs/redesign/01-target-architecture.md` §3 | What the layer rules forbid |
| `docs/redesign/03-contracts.md` | The `AssetEngine` / `WorldState` contract |
| `FINDYN_V1_SPEC.md` §8.1, §8.3, §14.1 | The no-lookahead law, verbatim |
| `compute/findynamics/core/contracts/vocab.py` | Why Ω cannot be a sixth asset |
| `compute/findynamics/core/contracts/state.py` | Which records validate `asset in ASSETS` |
| `compute/findynamics/core/config.py` | What `load_engine_configs` does to an unknown stem |
| `compute/findynamics/core/registry.py` | What `portfolio_engines(include_experimental=False)` filters |
| `compute/findynamics/engines/equity/features/kalman.py` | Why `filter()` and never `smooth()` |
| `compute/findynamics/engines/equity/features/kinematics.py` | Burn-in, the expanding z-score baseline, why jerk ships only as a z-score |
| `compute/findynamics/engines/equity/features/pipeline.py` | The one transform, its three callers, `FeatureSet` |
| `compute/findynamics/engines/equity/regime/design.py` | Why the HMM design matrix is not the Kalman slope |
| `compute/findynamics/engines/equity/regime/hmm.py` | Why a seed alone is not determinism |
| `compute/findynamics/engines/equity/rii.py` | **Every component already in the RII** |
| `compute/findynamics/backtest/replay.py` | What replay can and cannot prove |
| `compute/findynamics/backtest/portfolio.py` | The existing walk-forward + metric helpers |
| `compute/tests/engines/equity/test_no_lookahead_guards.py` | The source-grep guard pattern |
| `compute/tests/engines/equity/test_reproducibility.py` | The byte-identical-artifact pattern |
| `compute/pyproject.toml` `[tool.importlinter]` | Every existing contract |

## 2. The one finding that shapes the whole track

`engines/equity/rii.py::compute_rii` already builds a composite from posterior
entropy, confidence deficit, |jerk|, vol-of-vol, equity/bond correlation
breakdown, credit velocity and liquidity change. **Ω's candidate feature vector
overlaps this almost completely.**

Therefore the research question is *not* "does Ω beat price alone" — that is
nearly guaranteed and nearly worthless. It is:

> Does Ω carry information beyond `(P, v, a, j)` **and beyond the RII the engine
> already publishes**?

Write this into the design note as the primary falsification target. Every model
comparison in KK3 must carry an RII control arm.

## 3. Deliverable A — the design note

Write `docs/research/kk-omega-design.md`. It is the contract KK1–KK5 implement.
Sections, in this order:

1. **Hypothesis and scope.** The `(P,v,a,j) → (P,v,a,j,Ω,Ω̇,Ω̈,C,K)` statement.
   An explicit paragraph separating (a) mathematical inspiration, (b) model
   assumptions, (c) empirical observables, (d) fitted parameters, (e) results.
2. **The KK mapping table**, with a "what this is NOT" column:

   | KK concept | FinDynamics implementation | What it is not |
   |---|---|---|
   | 4D spacetime | observable state `(P,v,a,j)` | not spacetime |
   | 5th dimension | latent coordinate Ω | not a physical dimension |
   | metric `g_ij` | state-space covariance of the observable block | not a spacetime metric |
   | `A_i` | coupling field between observable and latent dynamics | not a gauge field |
   | `F_ij` | cross-state interaction term | — |
   | `φ` | latent-state scale (liquidity/complexity) | not a scalar field |
   | geodesic | the unforced trajectory of the fitted linear system | not a prediction |
   | curvature | `K`, a documented instability composite | not Riemann curvature |
   | motion in the 5th dimension | latent regime transition | — |

3. **Ω's observable inputs.** Name each candidate series by its **config id**,
   its frequency, its publication lag, and its first available observation *in
   the committed fixture* `compute/tests/fixtures/equity_prices.csv`:

   | Series id | Freq | In fixture from | Note |
   |---|---|---|---|
   | `YAHOO:^GSPC` | daily | 1927-12-30 | the long price path |
   | `FRED:SP500` | daily | 2016-08-01 | publication path, short |
   | `FRED:NASDAQ100` | daily | 1986-01-02 | cross-asset / dispersion proxy |
   | `FRED:DGS10` | daily | 2000-01-03 | rates level |
   | `FRED:T10Y3M` | daily | 2000-01-03 | curve slope |
   | `FRED:NFCI` | weekly | 2000-01-07 | liquidity / financial conditions |
   | `FRED:BAMLH0A0HYM2` | daily | **2023-08-01** | credit spread — *thin*, must be optional |

   State the conclusion explicitly: **the primary research window is 2000-01 →
   2026-07 daily**, and the HY OAS column is available for ~3 years only, so Ω
   must be defined over a *configurable, availability-checked* feature set that
   drops a column and says so rather than silently starting in 2023. Any wider
   window (back to 1927) is a price-only Ω and must be reported as a different
   specification, not as the same one.

   Every one of these ids is already declared in `config/series.yaml` (under
   `engines.equity.series` or `factors`). **Adding a new data dependency is out
   of scope for this track** unless you can show in the report that no
   configured series can stand in for it.

4. **Estimator choice.** Choose **Option A (PCA first component)** for the MVP
   and justify it against B (HMM) and C (autoencoder) on the four MVP criteria:
   deterministic, explainable, cheap, easy to backtest. Record that the estimator
   is swappable behind an interface, and that Option B is the planned second
   specification because `engines/equity/regime/hmm.py` already exists.
5. **The sign problem.** A PCA component's sign is arbitrary. Under walk-forward
   refits an unpinned sign flips Ω between windows, which turns Ω̇ into a spike
   train at every refit boundary and would look exactly like a signal. Document
   the pinning rule you will implement (see KK1 §3) and mark it as a required
   test.
6. **Anti-lookahead protocol.** The literal sequence: expanding window → fit
   scaler on the window → fit PCA on the window → transform *only* observations
   at or after the window end → derivatives from the stitched causal path.
7. **Determinism protocol.** Seed constant, `threadpool_limits(1)`, `math.fsum`,
   no dict-iteration-order dependence, `np.random.default_rng`.
8. **Falsification plan.** The five research questions Q1–Q5 from the track
   brief, each with: the statistic, the null, the horizon set, and the
   **pre-registered** decision rule. Write the decision rule *before* any number
   exists. Include the RII control arm in Q1.
9. **Verdict vocabulary.** `SUPPORTED | PARTIALLY SUPPORTED | NO INCREMENTAL
   INFORMATION | UNSTABLE | REJECTED`, each with the condition that produces it.
10. **What would make us delete this module.** One short section. If you cannot
    write it, the hypothesis is not falsifiable and the track should stop here.

## 4. Deliverable B — the scaffold

Create, with real module docstrings and `NotImplementedError` bodies (no logic):

```
compute/findynamics/research/__init__.py
compute/findynamics/research/omega/__init__.py
compute/findynamics/research/omega/config.py     # research-local YAML loader
compute/findynamics/research/omega/contracts.py  # frozen dataclasses
compute/findynamics/research/omega/estimator.py
compute/findynamics/research/omega/dynamics.py
compute/findynamics/research/omega/coupling.py
compute/findynamics/research/omega/curvature.py
compute/findynamics/research/omega/features.py
compute/findynamics/research/omega/backtest.py
compute/findynamics/research/omega/diagnostics.py
compute/config/research/omega.yaml
compute/tests/research/__init__.py
compute/tests/research/omega/__init__.py
```

`config/research/omega.yaml` in this phase:

```yaml
# KK-Ω — latent market-state research module. EXPERIMENTAL, RESEARCH ONLY.
#
# Disabled by default and quarantined by import-linter: no production path can
# reach this module even if this flag is flipped. See
# docs/research/kk-omega-design.md.
enabled: false
experimental: true
```

`config.py` must:
- resolve `CONFIG_DIR / "research" / "omega.yaml"` on its own, never through
  `core.config.load_engine_configs`;
- raise its own `OmegaConfigError` on a malformed file, at load time;
- expose `is_enabled()` returning `False` when the file is missing.

## 5. Deliverable C — the quarantine

In `compute/pyproject.toml`:

- Add `findynamics.research` as the **topmost** layer in the `Layers` contract,
  above `findynamics.portfolio`.
- Add a new contract, with a comment explaining it the way the crypto one is
  explained:

```toml
[[tool.importlinter.contracts]]
name = "Research is quarantined"
type = "forbidden"
source_modules = [
    "findynamics.portfolio",
    "findynamics.engines",
    "findynamics.factors",
    "findynamics.core",
    "findynamics.data",
]
forbidden_modules = ["findynamics.research"]
```

- Add the empty optional extra:

```toml
# KK-Ω research track. Empty on purpose, and kept rather than deleted — the first
# dependency this module needs must have an obvious home that is not the core
# dependency list.
research = []
```

Do **not** add `findynamics.research` to the `Engines are independent` contract:
it is not an engine.

## 6. Deliverable D — the guard tests

`compute/tests/research/omega/test_quarantine.py`:

1. Importing `findynamics.research.omega` does not register anything in
   `core.registry.ENGINES` (assert `registered_engines()` is unchanged).
2. `core.config.load_series_config()` still loads with `config/research/`
   present — i.e. the research config is invisible to the core loader.
3. `is_enabled()` is `False` with the shipped config.
4. `ASSETS` still has exactly five members (do not duplicate `tests/test_domain.py`;
   assert the length and membership only).

`compute/tests/research/omega/test_no_lookahead_guards.py`: copy the structure of
`tests/engines/equity/test_no_lookahead_guards.py` — same `tokenize`-based
comment/string stripping, same planted-violation self-test — pointed at
`findynamics/research/`, with the same `BANNED` patterns **plus**:

```python
r"StandardScaler\(\)\.fit_transform": "fitting a scaler on the whole sample leaks the future",
r"PCA\([^)]*\)\.fit_transform":       "fitting PCA on the whole sample leaks the future",
r"\.expanding\([^)]*\)\.apply":       "expanding().apply is easy to make non-causal",
```

If the third produces false positives on legitimate causal code, narrow the
pattern and say so in the report — do not simply delete it.

## Acceptance

- [ ] `docs/research/kk-omega-design.md` exists with all ten sections, and §10
      ("what would make us delete this") is non-empty and specific.
- [ ] Scaffold modules exist, import cleanly, contain no logic.
- [ ] `config/research/omega.yaml` ships `enabled: false`, `experimental: true`.
- [ ] `lint-imports` green, including the two new/changed contracts.
- [ ] A deliberately planted `from findynamics.research.omega import ...` inside
      `findynamics/portfolio/` makes `lint-imports` **fail** — paste the failure
      output in the report, then revert the plant.
- [ ] `pytest` green; new guard tests included.
- [ ] `ruff check` and `ruff format --check` green.
- [ ] Phase report answers every bracketed question from §1 in one line each.
