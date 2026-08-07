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

## `gold_daily.csv`

The gold fix and its four drivers, so the regime backtest asserts against 1980,
2008 and 2020 as they happened rather than against a series invented until the
windows passed.

* **Source**: `LBMA:GOLD_PM` plus FRED `DGS10`, `T10YIE`, `CPIAUCSL`,
  `DTWEXBGS`, `DTWEXM`, `NFCI`, `NASDAQ100`.
* **Shape**: one row per vintage, 1968-01 to 2026-07, 105,642 rows. Daily is
  unavoidable: the jump detector's unit of observation is a session, and
  15 April 2013 was a one-day event.
* **The price is not from FRED.** `FRED:GOLDPMGBD228NLBM` is what the P4 brief
  names and it no longer exists — FRED delisted the whole ICE Benchmark
  Administration set, and both the AM and PM series now answer
  `400 The series does not exist` (verified 2026-08-01 with a working key).
  LBMA publishes the same benchmark itself as static JSON, from 1968-04-01,
  which is more history than FRED ever carried. See
  `findynamics/data/providers/lbma.py`.
* **Why it starts in 1968 rather than at the first driver.** The drivers are
  only complete from 1974, but the fit window is what makes the model
  identifiable, and 1979-82 is the only unambiguous rate shock in the record:
  the real 10y went from -5% to +8% and gold fell 60%. Fitted on 1985 onwards
  the chain has never seen a carry headwind and cannot name one — measured in
  the walk-forward backtest rather than assumed.
* **Two spliced drivers, both flagged rather than blended.** `T10YIE` starts in
  2003, so before it the real rate is nominal minus trailing CPI (a different,
  ex-post quantity). `DTWEXBGS` starts in 2006 and `DTWEXM` ends in 2019, so
  the dollar trend is spliced on the 12-month *change* — the two are different
  baskets at different base levels, and a level splice would print a
  twenty-point step in 2006 that no dollar move produced.
* **`NFCI` carries no vintages here, and that is a repair rather than a gap.**
  Its ALFRED archive restates all 2,899 weeks on each of 789 vintages, which
  asks FRED for millions of rows and gets the first hundred thousand — the
  oldest periods of the oldest vintage. Fetched naively the series looks
  complete and stops in 2005. `providers/fred.py` now detects that truncation
  and refetches current values only, so release dates come from the configured
  7-day lag and revisions are not observable for this series.
* **Everything else keeps its vintages**, which is why the row count is several
  times the number of distinct (series, date) pairs: the dollar indices are
  reweighted annually and DGS10 is revised. `tests/engines/gold/test_replay.py`
  strips them for its cross-cutoff comparisons and explains why.

Regenerate the same way as above.

## `crypto_daily.csv`

Bitcoin's price and network metrics plus the two money-stock series, so the
regime assertions run against 2017, 2018, 2021 and 2022 as they happened rather
than against a series invented until the thresholds passed.

* **Source**: `YAHOO:BTC-USD`, blockchain.info's `market-price`,
  `estimated-transaction-volume-usd`, `n-unique-addresses`, `n-transactions` and
  `hash-rate`, plus FRED `M2SL` and `WALCL`.
* **Shape**: one row per observation, 2010-08-18 to 2026-08-05, 34,538 rows.
  Daily is unavoidable: the jump detector's unit of observation is a session and
  the drawdown windows are counted in them.
* **It starts 2010-08-18 because that is the first date bitcoin had a market
  price at all.** Not a chosen window: `market-price` is padded with 0.0 back to
  the genesis block, and the adapter drops those because "no market yet" is not a
  price of zero.
* **Two price legs, and they are different statistics.** `YAHOO:BTC-USD` is a
  daily close from 2014-09-17; `BLOCKCHAIN:MARKET_PRICE` is a volume-weighted
  daily *average* across exchanges from 2010-08-18. `engines/crypto/prices.py`
  splices the second in front of the first only after checking the three things
  that decide whether that is honest — measured on this snapshot: no step at the
  seam (−2.9%, against a 1.4% median daily move), no level bias (−0.18% mean
  signed gap over 4,340 shared dates), comparable volatility (ratio 1.016). They
  disagree hard on single high-range days (2020-03-12: a $4,971 close against a
  $7,937 average), which is the definition rather than an error, so every date the
  average supplied is flagged and the flag is published as `price_is_daily_average`.
* **Both legs are kept in the fixture rather than pre-stitched**, so the splice
  logic and its refusal branches are what the tests exercise. A pre-joined column
  would test that a file can be read.
* **The configured `STOOQ:BTCUSD` is absent.** Stooq is the role `series.yaml`
  declares and it is the right declaration; the endpoint fronts its CSV with the
  same JavaScript proof-of-work challenge that blocks `^SPX` (re-verified for
  this symbol 2026-08-05), so no automated egress this project has can reach it.
  The fallback path is therefore the path the tests exercise, which is also the
  path production takes — and
  `test_engine.py::test_the_fallback_price_source_is_reported_rather_than_hidden`
  asserts the engine says so rather than implying Stooq answered.
* **No supply data is fetched.** Issuance is generated from consensus constants
  in `engines/crypto/scarcity.py` — genesis 2009-01-03, 50 BTC, halvings every
  210,000 blocks — so there is nothing about supply in this file and nothing to
  regenerate.
* **Every day of the year is present, weekends included.** Bitcoin has no
  exchange calendar. That is why every annualization in this engine uses 365 and
  why `jumps.periods_per_year` is a configured rule rather than a constant.
* **`M2SL` and `WALCL` came from FRED's keyless CSV endpoint**
  (`fredgraph.csv?id=…`) rather than through the provider path, because no ALFRED
  key was available where this was generated. The *values* are the published
  ones either way; what is lost is the vintage archive, so release dates here are
  synthesized from the configured lag (14 days from month end for M2SL, 7 from
  period end for WALCL) exactly as `data/vintages.py` would do for a keyless
  fetch. The replay test's `SETTLEMENT_DAYS` carve-out exists because of this
  lag, and it explains itself at the point of use.
* **No paid on-chain metrics.** MVRV, SOPR and realized cap are the series this
  model would most like to have and every one is behind a vendor key; they are
  deliberately absent from the config as well as from here, and the TODO naming
  what adding one costs is in `data/providers/registry.py`.

Regenerate the same way as above. If a regeneration changes the labels the
acceptance windows assert on, that is a finding about the model or the source —
not a reason to move the fixture or the thresholds.
