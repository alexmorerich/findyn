# KK-Ω — walk-forward results (KK3)

Out-of-sample only. Every number below comes from a model that had never seen
the observation it is being graded on, and every decision rule was fixed in
[`kk-omega-design.md`](./kk-omega-design.md) §8 before any of these numbers
existed. KK5 writes the verdict; this document is the evidence.

**Reproduction.** From `compute/`, no network and no API keys:

```
python -m jobs.omega_research --force-experimental --out backtests/omega
python -m jobs.omega_research --force-experimental --shuffled --out backtests/omega
```

Fixture `tests/fixtures/equity_prices.csv`, config `config/research/omega.yaml`
(hash `05ecb9cc9870b18e`), model version `omega-0.1.0`.

## The run

| | |
|---|---|
| Rebalances | 258, monthly, 2005-02 → 2026-07 |
| Out-of-sample rows | 5,407 |
| Warm-up discarded | 5 years (2000-02 → 2005-01), entirely |
| Arms | A, **A′**, B, C, D, E |
| Horizons | 1, 5, 21, 63 trading days |
| Multiplicity family | 70 p-values, one Benjamini–Hochberg adjustment |

Refit at **every** rebalance because their value at *t* would otherwise depend on
data after *t*: the Ω projection, and the HMM posteriors (forward-backward
conditions on the whole sequence handed to it). Frozen at the **first** training
window because with parameters fixed each becomes a forward recursion: the
Kalman variances and FFD `d` for price, the HMM fit, Ω's own filter variances,
and the ridge/logistic regularization. Freezing once is strictly *less*
information than production's monthly refit.

## Verdicts

| | Real run | Shuffled control |
|---|---|---|
| **Q1** forward return | FAILS | FAILS |
| **Q2** forward volatility | **PASSES (arm B)** | FAILS |
| **Q3** regime transition | FAILS | FAILS |
| **Q4** coupling | FAILS | FAILS |
| **Q5** decision rule | PASSES (arm B, 3/5) | **PASSES — see below** |

## Q1 — Ω says nothing about forward returns

No Ω arm beats A′ at any horizon after adjustment; the best unadjusted p-value
is 0.064 and the best q-value is 0.249. At `h = 21` and `h = 63` the Ω arms are
substantially *worse* than both baselines. This is the expected outcome and it
is reported as such.

## Q2 — Ω does carry information about forward volatility

Arm **B** (`P, v, a, j` + Ω) against **A′** (`P, v, a, j` + the RII and its
components), paired on the dates both cover:

| h | B's OOS R² | A′'s OOS R² | ΔR² | HAC t | NW lag | q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.440 | 0.304 | **+0.137** | 4.54 | 10 | 0.00002 |
| 21 | 0.406 | 0.160 | **+0.247** | 4.03 | 20 | 0.00018 |
| 63 | 0.133 | −0.201 | **+0.337** | 3.14 | 62 | 0.00417 |

Positive in **5 of 5** sub-periods, so it is not `UNSTABLE` under §9's rule.
`h = 1` is absent because a one-day realized volatility is the dispersion of a
single return, which does not exist — see the deviation note below.

This is the one place Ω beats the control the design note calls the real
comparison. It is a statement about *volatility*, not about returns, and §1.2 of
the design note is explicit that those are different claims.

## Q3 — the RII detects transitions better than Ω does

A′ reaches AUC 0.907 at `h = 1` against arm A's 0.646. No Ω arm beats A′ at any
horizon except C at `h = 63` by +0.015. Ω's `latent_stress` fires on 6.9% of
days with a 59.7% false-alarm rate against the engine's 19.0% of days at 74.2%
— earlier and rarer, but not better discriminating.

## Q4 — the coupling adds nothing

Arm D never beats A′ significantly. `C`'s largest rank IC against any target is
0.040.

## Q5 — the pre-registered rule is not a test

Arm B beats A′'s Sharpe in 3 of 5 sub-periods with a 29.9 bps break-even, so it
passes as written. **The shuffled-target control passes it too.**

That is not a leak — Q1 through Q4 are clean on the control. It is a defect in
the rule: "higher Sharpe in three of five sub-periods" is satisfied by a coin
flip with probability 0.5, and the rule carries no significance test. A HAC
t-test on arm B's daily excess return over A′ gives **t = 0.37, p = 0.71**.

**KK5 must treat Q5's PASS as uninformative.** The rule is quoted as
pre-registered and the control's result is quoted beside it.

## The shuffled-target control — the acceptance gate

Every target permuted under seed 20260904, index and missingness preserved.

| | |
|---|---|
| Max abs OOS R², Q1 | 0.040 |
| Max abs OOS R², Q2 | 0.019 |
| Smallest q vs A′ | 0.197 |
| Max AUC | 0.571 (at the rarest base rate, ~18 positives) |

Q1–Q4 all FAIL. The harness is not leaking.

## Deviations from the literal pre-registration

Three, all reported rather than absorbed:

1. **"Non-negative on all four horizons" is read as "on all horizons the
   question was tested at".** Q2 has no `h = 1` target by construction. The
   literal reading makes Q2 neither passable nor failable.
2. **Q3's significance test is DeLong's, unadjusted.** The design note asks for
   `ΔAUC > 0 at q < 0.10` but the BH family covers Q1 and Q2 only. The test was
   added *after* the shuffled control passed Q3 on noise without one.
3. **Arm E could not be evaluated as specified.** `curvature` covers 2,067 of
   5,407 out-of-sample rows — it starts 2018-02 — and fails the availability
   check, so arm E ran as `A + Ω, Ω̇, Ω̈, C`. Arm A′ likewise dropped
   `rii_credit_velocity` (687 rows). Both drops are recorded in
   `WalkForwardResult.dropped_columns` and in `windows.csv`.

Deviation 3 is the open `curvature.zscore_min_years` question from KK2, now
measured out of sample. At the shipped 10.0 the full arm does not exist; at 5.0
`K` would start 2008-02 in-sample. **This is still your decision.**
