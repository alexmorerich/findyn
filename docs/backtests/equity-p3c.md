# FinEquity M4 — instability diagnostics

Generated 2026-07-31 from the committed price snapshot. Calibration series `YAHOO:^GSPC`.

Regenerate with `python -m jobs.diagnostics`. This report is committed deliberately and not on a schedule: it is an artifact of a model version, and a file that changed under a cron would make every number in it un-anchored.

> Read this beside `equity-open-issues.md`: this report is the measurement,
> that one is the interpretation and what is still wrong.

## 1. Tail fit (GPD)

Peaks-over-threshold on `SHILLER:NOMINAL_PRICE`, 1845 observations at 12/year.

| | |
|---|---|
| threshold | 10% drawdown |
| exceedances, raw | 1039 |
| exceedances, declustered | 46 |
| shape ξ | +0.3305 |
| scale β | 0.0832 |
| exceedance rate | 2.4932% per observation |

**Declustering removes 993 of 1039 exceedances.** Every period a drawdown stays below the threshold is an exceedance under a naive count, so one 2008 arrives as hundreds of 'independent' tail events. Fitting a GPD to that fits the autocorrelation of drawdowns, not their tail, and the fitted rate is the fraction of history spent underwater rather than the frequency of crashes.

ξ = +0.3305 > 0 — a heavy (Fréchet-domain) tail with no finite upper bound. That is the expected sign for equity drawdowns and it is what makes the severity at long return periods extrapolate rather than saturate.

### Return levels

| return period | fitted | empirical |
|---|---|---|
| 1 in 5y | 13.6% | 12.8% |
| 1 in 10y | 21.0% | 22.0% |
| 1 in 25y | 33.8% | 42.1% |
| 1 in 50y | 46.4% | 47.3% |
| 1 in 100y | 62.2% | 50.8% |

The record is 154 years long, so the empirical column stops where the record does. Nothing is extrapolated into it — a fit is only testable against something that is not the fit.

### The declustered episodes, worst first

| rank | depth |
|---|---|
| 1 | 84.8% |
| 2 | 50.8% |
| 3 | 47.3% |
| 4 | 43.7% |
| 5 | 43.4% |
| 6 | 42.1% |
| 7 | 37.7% |
| 8 | 37.4% |
| 9 | 34.0% |
| 10 | 29.3% |


## 2. RII discrimination

Peak RII over each episode's peak-to-trough window, against the mean over calm years. The maximum rather than the trough reading: an instability index earns its keep by being high *going in*, and by the trough the transition has already happened.

| episode | peak RII |
|---|---|
| 2000 dot-com | 96.5 |
| 2008 GFC | 95.7 |
| 2020 COVID | 82.3 |
| 2022 rate shock | 93.1 |

| calm year | mean RII |
|---|---|
| 1995 | 38.4 |
| 2005 | 45.2 |
| 2017 | 34.2 |
| 2021 | 52.0 |

**Separation: +49.4 points on a 0–100 scale — strong.**

### Per component

Episode mean minus calm mean, per component. A negative row is a component that reads *calmer* during crises than during calm years.

| component | episode − calm |
|---|---|
| confidence_deficit | +69.1 |
| posterior_entropy | +61.9 |
| jerk | +51.1 |
| vol_of_vol | +44.6 |
| correlation_breakdown | +43.0 |
| liquidity_stress | +35.4 |

Every component moves with the composite. That is the result §3.2 predicts and it is not what an earlier build measured — see open issue 12 for what changed and why the earlier reading was taken on too short a window to mean anything.

## 3. Monte Carlo calibration

Annualized drift implied by the simulated median at each horizon, against the realized drift of the calibration record. A simulation whose median 12-year path implies a return the market has never delivered is wrong however good its tails look.

Realized: **+6.34%/yr** over the calibration record.

| horizon | simulated median drift | P(drawdown > 20%) | realized frequency |
|---|---|---|---|
| tactical | +14.39%/yr | 18.9% | 6.4% |
| strategic | +9.66%/yr | 73.4% | 18.2% |
| generational | +6.43%/yr | 100.0% | 20.0% |
| educational_30y | +6.11%/yr | 100.0% | 12.4% |
| educational_50y | +5.94%/yr | 100.0% | 8.8% |

Mean absolute drift gap: **2.42%/yr**.

The two right-hand columns are not directly comparable and the difference matters: the simulated probability is over independent paths, the realized frequency is over overlapping rolling windows of the same record. Read them for order of magnitude, not for calibration.

The shock overlay retraces fully (`SHOCK_RETRACE = 1.0`). The permanent loss from historical crashes is already inside the fitted regime means; a partial retrace subtracts it a second time, which measured out at 2.7%/yr against a 6.1% realized. The overlay contributes the *shape* of a tail event — a discontinuity over days — and the regimes contribute the level.

