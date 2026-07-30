# P5 — FinCrypto engine (experimental, quarantined)

Copy everything below this line to the coder agent. Requires P4 merged.

---

You are working in the `findyn` repo. Read `docs/redesign/01-target-architecture.md`
and `docs/redesign/03-contracts.md`. No-lookahead law applies.

## Task

Build FinCrypto in `findynamics/engines/crypto/` as a **research-only,
quarantined** module. Hard constraints:

- `experimental = True` on the engine class; the portfolio engine (P6) must
  exclude it by default.
- Nothing outside `engines/crypto/` may import it (the import-linter
  quarantine contract from P0 already enforces this — do not weaken it).
- Its dependencies, if any beyond numpy/pandas, go in an optional extra
  `[project.optional-dependencies] crypto = [...]`, never in core deps.
- API responses for `/assets/crypto/*` carry an `"experimental": true` field
  and the standard disclaimer.

## 1. Data

Add under `engines.crypto.series`: BTC-USD daily price via the existing Stooq
provider (`btcusd`, lag 0). Macro inputs (global liquidity `M2SL`/`WALCL`)
come from the shared factors — add a `global_liquidity` factor to
`series.yaml` if not present. On-chain metrics: define the series ids and
lags in config now, but implement only sources that need **no new API keys**;
stub the rest behind the provider registry with a documented TODO.

## 2. Model (deliberately modest)

- `scarcity.py`: supply schedule from the halving calendar (deterministic,
  hard-coded dates are fine — they are consensus constants, not data).
- `liquidity_beta.py`: rolling regression of BTC returns on the
  global-liquidity factor change; expanding windows.
- `jumps.py`: reuse the same jump-detection approach as gold **by
  reimplementing or extracting a shared helper into `findynamics/backtest/`
  or `factors/` — not by importing `engines.gold`**.
- `regime.py`: simple vol/drawdown regime `frenzy | normal | winter`
  (thresholds in config).
- `engine.py`: `CryptoEngine(AssetEngine)`, `name="crypto"`,
  `experimental=True`. `AssetState`: `expected_return = None` (explicitly —
  the engine does not claim one); `risk_score` from vol + jump intensity;
  `signals` ⊇ `speculation_index` (vol × volume trend × distance from
  liquidity-implied fair band), `liquidity_beta`, `regime`.
  `confidence` capped at 0.5 by construction; document why.

## 3. Integration

Config enable flag (default **false** even after this phase ships). Dashboard
per `docs/redesign/04-ui-plan.md` §P5: `crypto.astro` with a persistent
EXPERIMENTAL banner, speculation index, liquidity-beta chart; the home card is
tagged and de-emphasized. Build, deploy, verify live.

## 4. Tests (acceptance)

- Halving schedule fixtures; liquidity-beta on synthetic data with known beta.
- Regime fixtures: 2017Q4/2021 → `frenzy`; 2018/2022 drawdowns → `winter`.
- PIT replay test at ≥2 cutoffs.
- `lint-imports` quarantine contract still green.
- With `enabled: false`, daily job output is byte-identical to before this
  phase (prove with a test or a recorded run diff).
