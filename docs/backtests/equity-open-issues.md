# FinEquity — open issues

Known defects and unresolved judgements in the equity engine, recorded at the
end of Phase 3. Everything here is *shipped behaviour*: the engine publishes
against these, and each one is either visible in the committed backtest
(`equity-p3b.md`) or reachable from the live API today.

Ordered by how much damage a reader could take from not knowing.

---

## 1. The transition probability is easy to misread — and looks alarming

**Status:** shipped, mitigated by labelling, not fixed.

The engine currently publishes `bull_expansion` at 0.97 confidence *and*
`p_transition_3m = 0.89`. Both are correct for what they measure, and the
pairing still reads as a contradiction.

The cause is the definition. An "adverse entry" is the regime path moving into
`bear` or `crisis` from anywhere else. The HMM changes regime roughly four to
five times a year, and its `bear` state on the 1927+ S&P is a −6.1% annualized
soft patch lasting ~44 sessions — not a bear market in the colloquial sense. So
entries are frequent, mostly minor, and the probability of one inside three
months is genuinely high whenever the market is *not* already in one.

Calibration is not the problem; it is good. On the calibration series the mean
prediction is 0.276 against an actual rate of 0.284, and on the publication
series — which the classifier never saw — 0.443 against 0.479.

**Mitigations shipped:** the signal note states the unconditional base rate and
says an entry is a regime change rather than a collapse; `base_rate_3m/6m/12m`
are published in `components` so a reader can see 0.89-against-0.28 rather than
0.89-against-nothing.

**What would actually fix it:** define the adverse event as an entry *followed
by a material drawdown* (say 10% within the horizon), or restrict it to `crisis`
entries. Both change the published number's meaning, so both are a model change
and belong in a version bump.

---

## 2. The transition classifier has no out-of-sample skill

**Status:** shipped, measured, documented in the backtest.

Out of sample over 1992–2026 the calibrated Brier scores are *worse* than always
predicting the base rate: **−3.6% / −2.8% / −1.7%** skill at 3 / 6 / 12 months.
In-sample and in purged CV it looks fine (AUC 0.72–0.86), so the model ranks;
what does not survive is its calibration once the regime environment shifts.

The reliability table shows where: the 3-month bulk tracks well
(0.146 → 0.218, 0.249 → 0.235, 0.352 → 0.325) and the top bins do not
(0.726 → 0.063 on 80 observations).

The signals are published anyway because the spec asks for them and they are
honestly labelled, but **nothing should be built on them as if they carried
skill** — in particular the RII's transition term inherits this.

---

## 3. The horizons are only monotone because they are forced to be

**Status:** repaired, cause outstanding.

The three horizons are nested events, so `p_3m ≤ p_6m ≤ p_12m` must hold. Three
independently-fitted classifiers with independent isotonic maps do not know
that, and the first live state produced **0.892 / 0.875 / 0.916**.

`engine.py::enforce_nesting` applies a running maximum in horizon order. That
prevents publishing an incoherent triple; it does not address why the underlying
estimates disagree. A single model over a horizon feature, or a monotone
multi-output calibration, would.

---

## 4. The engine does not detect a fast crash

**Status:** structural, and a spec non-goal — but it should be said plainly.

**COVID coverage is 0%.** Out of sample, the engine never called `bear` or
`crisis` at any point during the 23-trading-day fall from 2020-02-19 to
2020-03-23. It warned early — 245 trading days before the peak, at 0% drawdown —
but that warning is better read as the 2018–2019 wobble than as foresight.

Every feature is a trailing window: one-month realized volatility, a one-year
drawdown reference, a three-year volatility baseline. A model built this way
cannot turn inside a month. §0's non-goal 3 ("no claim of crisis prediction")
covers it, and the honest framing is that the engine detects *deterioration*,
not *events*.

The slower episodes are caught well: 2008 warned 144 trading days before the
peak at 5.5% drawdown, with 68% coverage of the fall.

---

## 5. False-alarm rate is 79%

**Status:** shipped, measured.

Of the 2,724 sessions called `bear` or `crisis` out of sample, only 20.9% were
followed by a drawdown of 20% or more within twelve months.

That is arithmetically unsurprising — the engine is adverse roughly a third of
the time and 20% drawdowns are rare — but it is the §15 metric and it means the
regime label is a poor standalone trading signal. It is a *state* description,
not an alarm, and the dashboard should never present it as the latter.

---

## 6. The deep-history cross-check agrees only in direction

**Status:** amber. Was the stated gate for sub-milestone C.

Running the same pipeline on `SHILLER:NOMINAL_PRICE` (monthly, genuinely the S&P
composite across all four episodes):

| Episode | Bear/crisis | late_cycle or worse |
|---|---:|---:|
| 2000 dot-com | 71% | 100% |
| 2008 GFC | 0% | 100% |
| 2020 COVID | 0% | 100% |
| 2022 rate shock | 0% | 67% |

All four episodes deteriorate and none is missed, but only 2000 reaches
bear/crisis. Labels are relative to each series' own return distribution and
monthly returns are far less fat-tailed than daily ones, so the monthly fit's
extreme states sit at milder values. That explains the gap without excusing it:
the two resolutions do not agree on severity, only on direction.

The original strict reading of this table — "0% coverage, three episodes missed"
— was a **metric bug**, now fixed. The monthly model had called 2008
`late_cycle` for 22 of 25 months.

---

## 7. The calibration series is a vendor copy, and unversioned

**Status:** improved during the phase; a new dependency.

`YAHOO:^GSPC` now fills the `backfill` role with 24,761 daily closes back to
1927, so `calibration` is the S&P itself and `calibration_is_proxy` is false.
This retired what had been the phase's central technical risk.

The residual risk moved rather than vanishing. Yahoo is unversioned and the spec
treats it as a fallback source (§5.1 source 7); a schema change on their side is
expected rather than surprising. If it goes away, `prices.py` falls back to
`FRED:NASDAQ100` automatically and every fitted parameter silently becomes a
proxy's — the state says so through `model_version` and the
`regime_model_is_proxy_fitted` signal, but nothing *blocks* the downgrade.

`test_transfer.py` deliberately still measures the NASDAQ path for this reason.

---

## 8. Kalman MLE does not converge, by design

**Status:** benign, recorded so nobody "fixes" it.

The local-linear-trend fit stops after ~22 iterations with a line-search failure
on every price series. The likelihood is nearly flat in the trend variance — an
index price is close to a random walk — so the optimizer lands on a boundary
solution where the model collapses toward a local level.

The parameters are identical at 200, 500 and 1000 iterations, so the fit is
stable and the replay test passes. Raising `kalman_maxiter` does nothing.

---

## 9. `risk_score` is not the RII

**Status:** interim by construction.

`AssetState.risk_score` is currently the posterior-weighted regime severity on a
0–100 axis, not the §3.2 Regime Instability Index. It is one of the RII's inputs.
Named as it is because the contract field is `risk_score`; the distinction is in
the engine docstring and in this list.

---

## 10. `expected_return` is regime-conditional, not a forecast

**Status:** interim by construction.

It is the posterior-weighted mean return of the fitted states, annualized — what
the fitted model implies about *now*, not the §10 Monte Carlo distribution
median. It also inherits the calibration series' realized returns.

---

## Smaller things

- **`derived_features` and `engine_output` overlap.** The kinematic path is
  published twice, in model units and in chart units. Deliberate — see
  `domain.py::CHART_METRICS` — but it doubles the write volume for that block.
- **The 2022 episode has no NBER date**, so lead/lag against a recession
  chronology is only computable for three of the four episodes. The market peak
  is the primary reference throughout for this reason.
- **`late_cycle` is more volatile than `crisis`** on the 1927+ S&P fit (26.8%
  against 26.2%). The labelling rule sorts on mean return alone, deliberately —
  the two rules that used volatility each mislabelled a real series — so this is
  expected, and the test asserts `crisis` is at or above median volatility rather
  than the maximum.
- **The backtest takes ~80 seconds** and is therefore a committed artifact
  regenerated by `python -m jobs.backtest`, not a CI check.
