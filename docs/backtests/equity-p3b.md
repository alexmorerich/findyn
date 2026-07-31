# FinEquity — walk-forward backtest (P3-B)

Generated 2026-07-31 · FINDYN_V1_SPEC.md §15

## What was measured

- **Calibration series**: `FRED:NASDAQ100`  ⚠️ **a proxy, not the S&P 500**
- **Out-of-sample span**: 1992-07-24 → 2026-07-29
- **Refits**: 18 expanding-window fits; every date is
  judged by a model fitted strictly before it.

> Every fitted parameter below comes from the NASDAQ 100, because daily
> S&P history before 2016 is not reachable (the Stooq endpoint
> bot-filters CI as well as developer networks). The NASDAQ is a more
> volatile, tech-heavy index. The transfer is defensible only because
> the design matrix is dimensionless — see the deep-history cross-check
> at the end, which runs the same pipeline on the real S&P composite.

## Episode detection

Lead is in trading days against the market peak; negative is a warning
*before* the peak. Drawdown-at-warning is measured from the running
maximum, which is the only version knowable on the day.

| Episode | First warning | Lead (d) | Drawdown at warning | Before −20%? | Bear/crisis | late_cycle+ | Recovery lag (d) |
|---|---|---:|---:|:-:|---:|---:|---:|
| 2000 dot-com | 1999-12-30 | -59 | 0.2% | yes | 59.1% | 94.8% | — |
| 2008 GFC | 2007-03-15 | -144 | 5.5% | yes | 68.0% | 68.0% | 125 |
| 2020 COVID | 2019-03-01 | -245 | 0.0% | yes | 0.0% | 0.0% | 0 |
| 2022 rate shock | 2022-01-24 | 14 | 12.5% | yes | 67.3% | 67.3% | 121 |

Against the NBER recession start, for the three episodes that have one
(2022 was a bear market without a recession):

| Episode | NBER start | Warning lead vs NBER (calendar days) |
|---|---|---:|
| 2000 dot-com | 2001-03-01 | -427 |
| 2008 GFC | 2007-12-01 | -261 |
| 2020 COVID | 2020-02-01 | -337 |

## False alarms

P(no drawdown ≥ 20% within 12 months | the engine called bear or crisis),
measured per session over the whole out-of-sample span.

- Sessions called bear or crisis: **2,724**
- False-alarm rate: **79.1%**
- Hit rate: **20.9%**

## Transition probabilities

Brier score of the calibrated probability against what the regime
actually did, out of sample. The reference is the Brier score of always
predicting the base rate — the score to beat.

| Horizon | Brier | Base rate | Reference Brier | Skill |
|---|---:|---:|---:|---:|
| 3m | 0.1907 | 24.3% | 0.1840 | -3.6% |
| 6m | 0.2525 | 43.4% | 0.2457 | -2.8% |
| 12m | 0.2208 | 68.1% | 0.2172 | -1.7% |

### Reliability, 3m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 819 | 0.025 | 0.046 |
| 0.1-0.2 | 2,404 | 0.146 | 0.218 |
| 0.2-0.3 | 1,473 | 0.249 | 0.235 |
| 0.3-0.4 | 2,314 | 0.352 | 0.325 |
| 0.4-0.5 | 764 | 0.436 | 0.323 |
| 0.5-0.6 | 444 | 0.532 | 0.241 |
| 0.6-0.7 | 203 | 0.643 | 0.227 |
| 0.7-0.8 | 80 | 0.726 | 0.062 |

### Reliability, 6m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 552 | 0.021 | 0.172 |
| 0.1-0.2 | 42 | 0.147 | 0.333 |
| 0.2-0.3 | 1,422 | 0.245 | 0.286 |
| 0.3-0.4 | 964 | 0.336 | 0.489 |
| 0.4-0.5 | 1,393 | 0.462 | 0.403 |
| 0.5-0.6 | 2,413 | 0.556 | 0.562 |
| 0.6-0.7 | 1,119 | 0.639 | 0.484 |
| 0.7-0.8 | 315 | 0.746 | 0.324 |
| 0.8-0.9 | 167 | 0.860 | 0.665 |
| 0.9-1.0 | 51 | 0.975 | 0.157 |

### Reliability, 12m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 285 | 0.076 | 0.684 |
| 0.1-0.2 | 164 | 0.119 | 0.470 |
| 0.2-0.3 | 28 | 0.228 | 0.964 |
| 0.3-0.4 | 133 | 0.344 | 0.925 |
| 0.4-0.5 | 1,282 | 0.451 | 0.504 |
| 0.5-0.6 | 186 | 0.527 | 0.629 |
| 0.6-0.7 | 2,037 | 0.638 | 0.615 |
| 0.7-0.8 | 2,030 | 0.751 | 0.739 |
| 0.8-0.9 | 651 | 0.818 | 0.555 |
| 0.9-1.0 | 1,516 | 0.978 | 0.898 |

## Deep-history cross-check

The same pipeline on `SHILLER:NOMINAL_PRICE` (monthly, 1918-07-01 → 2024-09-01) — genuinely the S&P composite
across all four episodes, where the calibration series is a proxy.

| Episode | First warning | Bear/crisis | late_cycle or worse | Mean severity |
|---|---|---:|---:|---:|
| 2000 dot-com | 1999-09-01 | 71.0% | 100.0% | 3.42 |
| 2008 GFC | 2010-02-01 | 0.0% | 100.0% | 2.00 |
| 2020 COVID | 2020-05-01 | 0.0% | 100.0% | 2.00 |
| 2022 rate shock | — | 0.0% | 66.7% | 1.67 |

Called bear or crisis: 2000 dot-com.
Deteriorated to late_cycle or worse: 2000 dot-com, 2008 GFC, 2020 COVID, 2022 rate shock.
Missed entirely: none.
