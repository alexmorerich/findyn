# FinEquity — walk-forward backtest (P3-B)

Generated 2026-07-31 · FINDYN_V1_SPEC.md §15

## What was measured

- **Calibration series**: `YAHOO:^GSPC`
- **Out-of-sample span**: 1934-08-27 → 2026-07-30
- **Refits**: 46 expanding-window fits; every date is
  judged by a model fitted strictly before it.

## Episode detection

Lead is in trading days against the market peak; negative is a warning
*before* the peak. Drawdown-at-warning is measured from the running
maximum, which is the only version knowable on the day.

| Episode | First warning | Lead (d) | Drawdown at warning | Before −20%? | Bear/crisis | late_cycle+ | Recovery lag (d) |
|---|---|---:|---:|:-:|---:|---:|---:|
| 2000 dot-com | 2000-09-19 | 123 | 4.4% | yes | 59.4% | 93.7% | 38 |
| 2008 GFC | 2008-01-08 | 62 | 11.2% | yes | 74.7% | 82.6% | 33 |
| 2020 COVID | 2019-02-19 | -252 | 0.0% | yes | 70.8% | 70.8% | 49 |
| 2022 rate shock | 2021-06-17 | -138 | 0.8% | yes | 27.6% | 86.7% | 14 |

Against the NBER recession start, for the three episodes that have one
(2022 was a bear market without a recession):

| Episode | NBER start | Warning lead vs NBER (calendar days) |
|---|---|---:|
| 2000 dot-com | 2001-03-01 | -163 |
| 2008 GFC | 2007-12-01 | 38 |
| 2020 COVID | 2020-02-01 | -347 |

## False alarms

P(no drawdown ≥ 20% within 12 months | the engine called bear or crisis),
measured per session over the whole out-of-sample span.

- Sessions called bear or crisis: **6,821**
- False-alarm rate: **79.8%**
- Hit rate: **20.2%**

## Transition probabilities

Brier score of the calibrated probability against what the regime
actually did, out of sample. The reference is the Brier score of always
predicting the base rate — the score to beat.

| Horizon | Brier | Base rate | Reference Brier | Skill |
|---|---:|---:|---:|---:|
| 3m | 0.2845 | 49.3% | 0.2500 | -13.8% |
| 6m | 0.2461 | 77.2% | 0.1761 | -39.8% |
| 12m | 0.0967 | 95.1% | 0.0462 | -109.2% |

### Reliability, 3m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 1,563 | 0.070 | 0.514 |
| 0.1-0.2 | 2,462 | 0.158 | 0.513 |
| 0.2-0.3 | 3,806 | 0.253 | 0.401 |
| 0.3-0.4 | 4,535 | 0.351 | 0.455 |
| 0.4-0.5 | 4,384 | 0.439 | 0.495 |
| 0.5-0.6 | 3,184 | 0.544 | 0.572 |
| 0.6-0.7 | 1,705 | 0.651 | 0.558 |
| 0.7-0.8 | 1,039 | 0.740 | 0.600 |
| 0.8-0.9 | 214 | 0.835 | 0.411 |
| 0.9-1.0 | 147 | 0.970 | 0.408 |

### Reliability, 6m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.0-0.1 | 256 | 0.058 | 0.773 |
| 0.1-0.2 | 875 | 0.160 | 0.826 |
| 0.2-0.3 | 989 | 0.271 | 0.875 |
| 0.3-0.4 | 2,313 | 0.355 | 0.779 |
| 0.4-0.5 | 1,455 | 0.457 | 0.752 |
| 0.5-0.6 | 3,080 | 0.559 | 0.782 |
| 0.6-0.7 | 6,077 | 0.660 | 0.704 |
| 0.7-0.8 | 4,662 | 0.748 | 0.786 |
| 0.8-0.9 | 2,492 | 0.837 | 0.826 |
| 0.9-1.0 | 777 | 0.964 | 0.828 |

### Reliability, 12m

| Predicted band | Observations | Mean predicted | Observed frequency |
|---|---:|---:|---:|
| 0.1-0.2 | 49 | 0.180 | 1.000 |
| 0.2-0.3 | 181 | 0.244 | 1.000 |
| 0.3-0.4 | 354 | 0.372 | 1.000 |
| 0.4-0.5 | 1,713 | 0.464 | 1.000 |
| 0.5-0.6 | 1,022 | 0.558 | 0.866 |
| 0.6-0.7 | 1,666 | 0.660 | 0.978 |
| 0.7-0.8 | 3,619 | 0.754 | 0.919 |
| 0.8-0.9 | 4,853 | 0.859 | 0.940 |
| 0.9-1.0 | 9,393 | 0.951 | 0.962 |

## Deep-history cross-check

The same pipeline on `SHILLER:NOMINAL_PRICE` (monthly, 1918-07-01 → 2024-09-01) — genuinely the S&P composite
across all four episodes, where the calibration series is a proxy.

| Episode | First warning | Bear/crisis | late_cycle or worse | Mean severity |
|---|---|---:|---:|---:|
| 2000 dot-com | 2000-10-01 | 41.9% | 48.4% | 2.13 |
| 2008 GFC | 2008-02-01 | 70.6% | 88.2% | 3.29 |
| 2020 COVID | 2019-03-01 | 100.0% | 100.0% | 3.00 |
| 2022 rate shock | 2021-09-01 | 44.4% | 100.0% | 2.56 |

Called bear or crisis: 2000 dot-com, 2008 GFC, 2020 COVID, 2022 rate shock.
Deteriorated to late_cycle or worse: none.
Missed entirely: none.
