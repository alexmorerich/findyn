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

## `money_daily.csv`

The real money-market inputs FinMoney reads, so the liquidity regimes are
asserted against the two funding events everyone agrees on rather than against
thresholds tuned until a made-up fixture passed.

* **Source**: FRED `SOFR`, `DTB3`, `DGS3MO`, `RRPONTSYD`.
* **Shape**: daily, two windows — 2018-01-02 to 2020-12-31 and 2024-06-03 to
  2024-10-31. Daily is unavoidable here: a repo squeeze is a two-session event
  and month-end sampling cannot see it.
* **Why those windows.** The first covers `SOFR`'s 2018-04-03 start (so the
  SOFR/DTB3 splice is exercised on the real splice date), the September 2019
  repo blowup and the March 2020 dash for cash. The second is a deliberate
  **negative** control: the September 2024 easing cycle puts the 3m bill half a
  point below SOFR for the same reason a crisis does, and the regime rules have
  to tell the two apart.
* **Vintages**: `release_date = revision_date = obs_date + 1 day`. None of these
  four series is revised, and each is published the next business morning, so
  the synthesized lag is the true one rather than a conservative guess.
* **`SOFR` genuinely starts 2018-04-03** — the gap before it is the case
  `account.py` splices `DTB3` into, not a hole in the snapshot.

Regenerate the same way as above. Contiguous windows matter: the trailing carry
windows and the spread trend are computed off consecutive observations, and
stitching two disjoint months together would silently compute a 1-month carry
across a four-year gap.

## `equity_prices.csv`

The three price series FinEquity resolves its roles from, so the role
resolution, the `d` search and the rule-5 replay are all asserted against real
index history rather than against a random walk that was tuned until they
passed.

* **Source**: FRED `SP500`, FRED `NASDAQ100`, Shiller nominal price.
* **Shape**: one row per vintage, 1871-01 to 2026-07-29, 16,351 rows.
  - `FRED:SP500` — 2,512 daily observations from 2016-08-01. That is the whole
    series: FRED's licence caps it at a rolling ten-year window, which is
    exactly why the engine cannot fit a regime model on it.
  - `FRED:NASDAQ100` — 11,994 rows over 10,223 distinct trading days from
    1986-01-02. The row count exceeds the day count because this one *does*
    carry ALFRED vintages; `pit_history` keeps the newest per period.
  - `SHILLER:NOMINAL_PRICE` — 1,845 monthly observations from 1871-01.
    It ends 2024-09 because the published dataset does; that lag is real.
* **Vintages**: as the providers returned them. The FRED price series carry no
  true ALFRED archive, so their release dates are synthesised from the
  configured one-day lag by `data/vintages.py` — which means the replay test
  proves *lag discipline* for them, not vintage fidelity.
* **Why all three.** They are not interchangeable and the tests exist to prove
  it: each needs its own FFD `d` (0.35 / 0.55 / 0.90 as committed), and the
  monthly series annualizes at 12 periods a year where the daily ones use 252.
  A single-series fixture could not fail the assertions that matter.
* **`STOOQ:^SPX` is deliberately absent.** Daily S&P history before 2016 is not
  reachable — the endpoint bot-filters CI as well as developer networks — so
  `prices.py` resolves `calibration` to the NASDAQ proxy. `test_prices.py`
  asserts that absence, because the resolution is meant to be a function of
  what D1 actually holds.

Regenerate the same way as above.
