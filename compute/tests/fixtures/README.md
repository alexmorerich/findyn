# Test fixtures

## `treasury_monthly.csv`

A committed snapshot of the real US Treasury constant-maturity curve, so the
sanity backtest can assert against history that actually happened rather than
against numbers we made up.

* **Source**: FRED `DGS1MO … DGS30`, fetched through the normal provider path.
* **Shape**: the last trading day of each month, 1988-01 to 2026-07. Monthly,
  not daily, because the daily series is ~180k rows and a repository is not a
  data warehouse; month-end is enough resolution to identify every inversion
  since 1988.
* **Vintages**: original prints only (`revision_date == release_date`). DGS is
  revised four times in sixty years, so keeping revisions would double the
  ambiguity for none of the signal.
* **`DGS1MO` starts in 2001** — that is real, not a gap in the snapshot, and it
  is precisely the ragged-curve case `curve.py` has to handle.

Tests that use it must set `trend_days` to a monthly-appropriate window
(12 periods ≈ 1 year); the shipped config value of 252 is in trading days.

Regenerate with the fetch in `jobs/backfill.py`; do not hand-edit. If a
regenerated file changes historical values, that is a finding about the source,
not a reason to overwrite the fixture silently.
