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

## 3b. RESOLVED — the regime feature set

**Status:** fixed. Kept for the record because the failure is instructive and the
trade-off it settled is still live.

`trend_to_noise` divided the **Kalman slope** by realized volatility. The
local-linear-trend MLE drives `sigma2_trend` to ~1e-12 on every price series — an
index is close to a random walk, so the likelihood is flat in that parameter —
and the fitted slope barely moved: standard deviation 0.05 against 0.28 for a
plain trailing return, correlated only 0.47 with it.

That is not a trend-to-noise ratio. It is an inverse-volatility feature wearing a
trend's name, and it reached production: the lowest-volatility state carried the
highest value, the labelling rule called it the most bullish, and a quiet rising
market at all-time highs was published as **`bear` at 97% confidence**.

The first repair — a single trailing 3-month return — fixed today's label and
broke 1987 (29% coverage) and the cross-index transfer. The settled answer is
**two horizons**, `trend_fast` (3 months) and `trend_slow` (12 months), because a
fast crash and a slow bear are different events and one window cannot see both.

What changed, measured out of sample on the 1927+ S&P:

| | before | after |
|---|---:|---:|
| 2020 COVID coverage | **0%** | **70.8%** |
| 2008 GFC coverage | 68.0% | 74.7% |
| 2000 dot-com coverage | 59.1% | 59.4% |
| 2022 coverage | 67.3% | 27.6% |
| deep-history episodes reaching bear/crisis | 1 of 4 | **4 of 4** |
| live label at all-time highs | `bear` (0.97) | `normal_expansion` (0.83) |

The deep-history cross-check — the stated gate for sub-milestone C — is now
green: 2008 goes 0% → 70.6%, 2020 0% → 100%, 2022 0% → 44.4% on the real S&P
composite.

**The trade-off that remains.** 2022 fell from 67.3% to 27.6% bear-or-crisis
coverage, though it still reaches `late_cycle` or worse for 86.7% of the episode.
2022 was a slow grind with no single violent leg, which is the case a fast
horizon does not help with and a 12-month horizon reads as merely weak.

**The method changed too, and that matters more than the answer.** The first two
attempts each fixed one criterion and broke another that was only discovered
afterwards. The fix came from scoring candidates against *every* criterion at
once — transfer, all eight historical episodes, calm-year false alarms, state
structure, persistence and the live label — before changing anything. Any future
change to this feature set should be made the same way.

---

## 3c. The transition classifier got worse, not better

**Status:** open, and now the largest problem in the L3 layer.

The new feature set produces more regime changes, so adverse *entries* are more
frequent and the base rates climbed hard:

| Horizon | base rate before | base rate after | Brier skill after |
|---|---:|---:|---:|
| 3m | 24.3% | 49.3% | −13.8% |
| 6m | 43.4% | 77.2% | −39.8% |
| 12m | 68.1% | **95.1%** | −109.2% |

A 12-month transition probability against a 95% base rate carries almost no
information — the live state publishes 0.9951 for it. The direction flags handle
this correctly (6m and 12m read neutral rather than adverse, because they are at
their base rates), but the numbers should not be given prominence on a dashboard
until the adverse event is redefined, which is issue #1.

This strengthens rather than changes the conclusion of issue #2: **nothing should
be built on the transition probabilities as though they carried skill.**

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

## 9. RESOLVED — `risk_score` is now the RII

**Status:** fixed in M4.

`AssetState.risk_score` was the posterior-weighted regime severity, which was one
of the RII's inputs standing in for the whole thing. It is now the §3.2 composite
from `rii.py`. See issue 12 for how well it actually discriminates.

---

## 10. RESOLVED — `expected_return` is now the simulated median

**Status:** fixed in M4.

It is now the median of the strategic-horizon Monte Carlo distribution rather
than the posterior-weighted mean of the fitted state returns. It still inherits
the calibration series' realized returns through the fitted regime statistics,
which is issue 7, not this one.

---

## 11. RESOLVED — fitted models were stored in an ephemeral directory

**Status:** fixed. Recorded because it was a production blocker that no test
would ever have caught.

`AssetEngine.fit` runs monthly and `predict` runs daily, in *different* GitHub
Actions containers. Artifacts were written to `compute/artifacts/`, which is
gitignored and destroyed with the runner — so the daily job never once saw what
the refit fitted. Since the equity engine's correct response to a missing fit is
to publish no state, a scheduled production run would have published nothing,
forever, without erroring.

Fitted models now live in R2 behind the serving plane's existing HMAC door
(§6), with three properties:

- **Immutable.** A version is written once. Re-writing identical bytes is a
  no-op (retries are routine); re-writing *different* bytes under the same
  version is a 409. Without that, a published `model_version` is a claim nobody
  can check.
- **Addressed by `model_version`.** The key is `artifacts/<name>/<version>.json`,
  so "load the model that produced this state" is a lookup. `FINDYN_MODEL_VERSION`
  pins a run to an exact version for replay; a daily run follows `latest`.
- **Chosen by environment, not by flag.** `build_artifact_store()` returns R2
  when the serving plane is configured and the local directory otherwise, so
  there is no argument anyone can forget that would let an ephemeral artifact
  reach production.

Verified by deleting `compute/artifacts/` entirely and running the daily job: it
loaded `equity-1.0.0+cal.yahoo_gspc` from R2 and published a state.

---

## 12. RESOLVED — the RII was being ranked against eight years

**Status:** fixed. Recorded because the broken version was measured, written up
as a design limitation of §3.2, and was nothing of the kind.

The first working RII separated March 2020 from a calm 2021 by **1.3 points on a
0–100 scale**, and the per-component breakdown showed posterior entropy,
confidence deficit and correlation breakdown all moving *against* the composite.
The conclusion drawn from that — that §3.2's use of model uncertainty is
structurally wrong, because in a crash the HMM is confident and its entropy
collapses — was wrong, and it was wrong in the most convincing possible way: it
had a mechanism, and the mechanism is real, it just was not what was happening.

What was actually happening: every RII component is scored as an **expanding
percentile**, and the index was being computed on the publication path. That path
is `FRED:SP500`, which FRED caps at ten years. So a 2020 reading was being ranked
against a window starting in 2016, `jerk` had not finished warming up, `vol_of_vol`
was ranked against a sample containing exactly one crash, and several components
were NaN for most of the window. Every one of those pushes the composite toward
the middle, and the "components disagree" table was mostly warm-up artifacts.

The index is now computed on the **calibration** path — the same index, back to
1927 — and sliced to the publication dates for publishing. On that basis
(`docs/backtests/equity-p3c.md` §2):

| | RII |
|---|---|
| 2000 dot-com, peak | 96.5 |
| 2008 GFC, peak | 95.7 |
| 2020 COVID, peak | 82.3 |
| 2022 rate shock, peak | 93.1 |
| 1995, mean | 38.4 |
| 2005, mean | 45.2 |
| 2017, mean | 34.2 |
| 2021, mean | 52.0 |

**Separation: +49.4 points.** Every component moves with the composite, including
the two that appeared to be inverted — confidence deficit is the *largest*
contributor at +69.1.

Two things worth keeping from this:

- **A percentile is a claim about the reference window**, and the window is not
  visible in the output. A 0–100 score looks equally authoritative whether it was
  ranked against a century or against four years, which is precisely why this was
  measured, believed and written down before anyone checked what it was ranked
  against.
- **A mechanism is not evidence.** The entropy-collapse story was plausible,
  correct in isolation, and had nothing to do with the observed numbers. It was
  accepted because it explained them.

---

## 13. FRED vintage requests silently truncate to the wrong century

**Status:** worked around, root cause understood.

Fetching `NFCI` through the vintage path returned exactly **100,000 rows covering
1971–1976** — a full-looking response that was a *prefix*, not the series. The
FRED observations endpoint caps a response at 100,000 rows and paginates; with
`realtime_start`/`realtime_end` set, every observation is returned once per
vintage, so a weekly series with 50 years of revisions blows through the cap
inside the first five years and the remaining 45 are simply absent.

Nothing errored. The rows were real, correctly dated, and correctly PIT — just
one twentieth of the series.

The RII's liquidity component now fetches NFCI with `use_vintages=False`. That is
a real weakening: NFCI *is* revised, so the component reads today's values at
historical dates. It is recorded here rather than buried because it is a
lookahead of exactly the kind this system exists to avoid, bounded to one
component of one composite. The correct fix is pagination in the FRED provider,
which is a Phase-1 change.

---

## 14. The high-yield spread starts in 2023

**Status:** external constraint, not a bug.

`BAMLH0A0HYM2` returns 787 rows beginning 2023-08 through the API key in use.
ICE BofA licenses the index and FRED restricts historical depth accordingly.

The consequence is concrete: `credit_velocity` is one of seven RII components and
one of four fragility sub-scores, and neither can see 2008 or 2020. The crash
decomposition's transmission factor is therefore reading a three-year window for
its credit inputs and a full history for the rest. Both are published, so the
imbalance is visible in `fragility_*`, but no test can catch a component that is
merely young.

---

## 15. The transmission floor is a judgement call

**Status:** open, and a number someone should argue with.

`TRANSMISSION_FLOOR = 0.10`. Without it, every fragility sub-score clips to zero
in benign conditions, transmission reads 0.012, and — because crash risk is a
*product* — the composite goes to roughly zero in exactly the calm periods where
a reader most wants to see a small non-zero number.

The floor asserts that a shock in calm conditions transmits *less*, not that it
fails to transmit. That is the right shape. The specific value 0.10 is not
derived from anything: it was chosen so the published composite stays legible,
which means it sets a hard lower bound on crash risk that no data can move.
A reader should treat crash-risk readings near the floor as "the model has
nothing to say", not as a measurement.

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
- **Monte Carlo shock recovery is total.** `SHOCK_RETRACE = 1.0`: the overlay
  contributes the *shape* of a tail event and none of its permanent loss, because
  the permanent loss is already inside the fitted regime means. Partial retrace
  double-counted it and took the 12-year median from 6%/yr to 2.7%/yr against a
  6.1% historical. The test pins the drift-neutrality rather than the constant.
- **The backtest takes ~80 seconds** and is therefore a committed artifact
  regenerated by `python -m jobs.backtest`, not a CI check.
