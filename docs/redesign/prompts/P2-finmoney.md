# P2 — FinMoney engine

Copy everything below this line to the coder agent. Requires P1 merged.

---

You are working in the `findyn` repo. Read `docs/redesign/01-target-architecture.md`
and `docs/redesign/03-contracts.md` first. No-lookahead law applies
(`FINDYN_V1_SPEC.md` §14.1): all data via `WorldState.series`.

## Task

Build FinMoney in `findynamics/engines/money/`: the risk-free / time-value
engine. It is deliberately simple — **no ML, no regression**. It is the
numeraire foundation the other engines discount against.

## 1. Data

Add under `engines.money.series` in `series.yaml`: `DGS3MO` (already ingested
by P1 — reuse, do not duplicate rows), `DTB3` (3m T-bill secondary market),
`SOFR` (daily, lag 1), `RRPONTSYD` (ON RRP volume, liquidity gauge). Extend
backfill config only; no provider code.

## 2. Model

- `account.py`: money-market account `M(t) = M(0)·exp(Σ r(tᵢ)·Δtᵢ)` on the
  PIT short-rate path (SOFR primary, DTB3 fallback pre-2018; ACT/360;
  document the day-count in the docstring). Expose cumulative wealth index
  and realized carry over trailing windows (1m/3m/12m).
- `discount.py`: discount factors `D(t, h)` for the standard horizons in
  `core/contracts/vocab.py`, from the short rate (h ≤ 1y) and the P1 fitted
  NS curve **read from `engine_output` via `WorldState.series`, not by
  importing the rates engine** (independence contract — CI enforces it).
- `liquidity.py`: liquidity state from RRP volume trend + 3m bill−SOFR
  spread; vocabulary `abundant | normal | tightening | stressed`, thresholds
  in `config/engines/money.yaml`.
- `engine.py`: `MoneyEngine(AssetEngine)`, `name="money"`. `predict` →
  `AssetState`: `regime` = liquidity state; `expected_return` = current
  annualized short rate; `risk_score` = near 0 by construction (document
  scale); `signals` ⊇ `real_carry` (nominal carry − inflation factor score
  direction), `bill_sofr_spread`. `outputs()` → `carry_1m carry_3m carry_12m`,
  `discount_1y discount_3y discount_10y`, `wealth_index`.

## 3. Integration

- Enable in `config/engines/money.yaml`; daily job picks it up with **zero
  job-code changes** (that was the point of the P1 orchestrator — if it needs
  changes, fix the orchestrator, not the job).
- Dashboard per `docs/redesign/04-ui-plan.md` §P2: header numeraire ribbon
  (short rate + liquidity chip on all pages) and `money.astro` (wealth index
  chart, carry table, liquidity badge, discount-factor curve). Build, deploy,
  verify live — deploy is part of Done.

## 4. Tests (acceptance)

- `M(t)` against hand-computed fixtures incl. the SOFR/DTB3 splice date.
- Discount-factor monotonicity and `D(t,0)=1`.
- PIT replay test at ≥2 cutoffs via `backtest/replay.py`.
- Liquidity regime fixtures for Sep-2019 repo stress and Mar-2020 (should
  read `stressed`).
- `lint-imports` proves money does not import `engines.rates`.
- Full pytest, ruff, serving vitest green.
