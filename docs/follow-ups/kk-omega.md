# Follow-up — KK-Ω, after the track closed

Opened at the end of KK6, **after** `docs/research/kk-omega-report.md` was
written. Everything here is post-hoc by construction and is marked as such: none
of it may be read back into the pre-registration in
`docs/research/kk-omega-design.md`, which is dated before any number existed.

## What was learned

**The negative result is the main one.** Ω carries no information about forward
returns beyond the RII the equity engine already publishes. That was named as the
most likely outcome in §1.2 of the design note, before the data, and it is what
happened. The RII control arm is the reason the answer is trustworthy: against
price alone, several Ω arms look fine.

**One positive result survived every check.** Arm B — price kinematics plus Ω —
improves out-of-sample R² on forward realized volatility over the RII control at
three horizons, with autocorrelation-corrected t-statistics above 3, BH-adjusted
q-values below threshold, in five sub-periods of five, and it survives all five
sensitivities including dropping `liquidity_stress` from Ω's own column block.

**The pre-registration had two holes, and the shuffled-target control found
both.** Q3 and Q5 were written without significance tests, so a coin flip could
satisfy them; the control passed Q5 and would have passed Q3. A control that only
ever confirms the harness is clean is worth less than one that also audits the
rules, and this one did.

**The verdict vocabulary had a third hole.** §9 does not cover "Q1 fails and
exactly one of Q2/Q3/Q4 passes", which is what happened. The report reached no
verdict rather than rounding up.

## What should be deleted

The track's own §10 named the conditions. Applying them:

| Component | Recommendation | Why |
|---|---|---|
| `coupling.py` | **Delete** | Three estimators, no usable rank IC against either target, and arm D never beat the control. The expanding-OLS coefficient path also cannot reject a unit root (p = 0.59), so any future use needs differencing first. |
| `curvature.py` | **Delete** | `K` covered 2,067 of 5,407 out-of-sample rows because its terms warm up twice, so arm E was never really tested. Reviving it means changing `zscore_min_years` first — see below. |
| `estimator.py`, `features.py`, `dynamics.py` | **Keep, narrowed** | These produce the one result that survived. If Ω is kept at all it is as a volatility covariate, not as a general latent state. |
| `backtest.py`, `evaluate.py`, `statistics.py`, `targets.py` | **Keep** | The harness is the reusable asset. It is the only walk-forward in this repository with a shuffled-target control, and the control has already earned its keep. |
| `lab.py` + the dashboard panels | **Delete with the coupling and curvature** | Two of the five panels would be empty. |

Deleting is safe by construction: the `Research is quarantined` contract means
nothing in production imports any of it, and KK6 demonstrated that with a planted
import that fails the build.

## What would make this worth revisiting

In descending order of how much they would change the conclusion, and honestly
about cost:

1. **Decompose the volatility result — one run, two minutes.** Replace Ω with
   each of its eight columns alone. If `realized_vol` on its own reproduces most
   of arm B's ΔR², the finding is that a stress composite forecasts volatility,
   which is not news, rather than that the latent coordinate does. **This is the
   cheapest experiment that could most change the conclusion, and it should be
   run before anything else here is taken seriously.**
2. **Re-run with `curvature.zscore_min_years: 5.0`** so `K` reaches 2008 and arm
   E is tested rather than assumed to fail. One config line, one run.
3. **A second index.** Everything here is the S&P 500. A result that does not
   appear on the NASDAQ 100 — already in the fixture — is a result about one
   series.
4. **A credit spread with history.** `FRED:BAMLH0A0HYM2` covers three years of a
   26-year window and was dropped from Ω at every rebalance. The one candidate
   column most likely to carry independent information never entered the model.
5. **The HMM estimator for Ω**, which the design note named as the planned second
   specification. Only worth building if (1) says the latent coordinate is doing
   work that its individual columns are not.

## What must be repaired before any of that

- **§9 of the design note has a gap.** Any repair must be dated after
  `docs/research/kk-omega-report.md`, marked post-hoc, and must not silently
  change the rule the report quotes.
- **Q3 and Q5 need significance tests written into the pre-registration**, not
  bolted on after a control exposes them. The tests added in KK3 are documented
  in the report as post-hoc for exactly this reason.
- **The equity engine's Monte Carlo block is not reproducible run to run.** KK6
  measured it: two runs of the same commit, same fixture, same fresh artifact
  store differ on all seven `mc_*` fields and on `expected_return`, while every
  other published field is identical. That is a production determinism gap this
  track happened to surface and did not cause; it belongs in its own issue.
