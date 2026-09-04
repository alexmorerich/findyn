# KK5 — The research report and the verdict

Copy everything below this line to the coder agent. Requires KK3 merged (KK4
optional).

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
and `docs/research/kk-omega-design.md` §8–§10 — the falsification plan and the
verdict rules were pre-registered there and you are now bound by them.

Branch: `research/kk5-report`.

## Task

Write `docs/research/kk-omega-report.md`: the complete empirical account of
whether the latent-state extension carries incremental out-of-sample
information. Every number comes from the KK3 artifacts under
`compute/backtests/omega/`. Nothing is recomputed by hand, nothing is quoted from
memory, and the report is regenerable: `python -m jobs.omega_research --report`
must rewrite it from the stored CSVs.

Follow the tone of `docs/backtests/equity-p3b.md` and
`docs/backtests/equity-open-issues.md` — sober, specific, and willing to say what
did not work.

## Required sections

### A. Mathematical formulation
Define Ω, Ω̇, Ω̈, `C`, `K` **as implemented**, not as aspired to. Each definition
carries: the formula, the estimator, the config keys that parameterize it, and
the numerical safeguards. State the KK inspiration in one paragraph and the
"this is an analogy, not a physical claim" boundary in one sentence.

### B. Data
Datasets and config ids, date ranges **as actually available in the fixture**,
frequency, missing-data treatment, which columns were dropped and on which dates,
the training/test window schedule, the refit cadence, and the warm-up discarded.
If the effective research window is shorter than intended because of the HY OAS
availability, say so here in the first paragraph, not in a footnote.

### C. Model comparison
The five arms A / A′ / B / C / D / E, each with feature list and estimator.
A table of OOS R² (and ΔR² vs both A and A′) per horizon, with
autocorrelation-corrected t-stats, unadjusted p-values and BH-adjusted q-values.

**The A′ row is the headline.** If Ω beats A but not A′, the report's summary
sentence is that Ω re-derives information the RII already publishes.

### D. Performance
CAGR, Sharpe, Sortino, Max Drawdown, Calmar, volatility, hit rate, turnover, and
transaction-cost-adjusted return per arm. Plus the **break-even cost** at which
each arm's edge vanishes. Reuse the metric helpers in
`findynamics/backtest/portfolio.py` rather than writing a second Sharpe.

### E. Predictive statistics
Correlations and rank ICs, feature importance (linear coefficients with
standard errors — no SHAP unless you also report its instability across windows),
OOS R², classification AUC and Brier for the transition target, significance with
the multiplicity family stated, and the sub-period stability table.

Include the **redundancy table** from KK2 §3: Ω, Ω̇, `C`, `K` against `velocity`,
`acceleration`, `jerk_z`, RII and each RII component.

### F. Failure analysis
Not optional and not short. Required content:

- Where Ω fails: name the dates and episodes. Use `EPISODES` from
  `engines/equity/backtest.py` so it is comparable with the existing report.
- The false-alarm rate of `latent_stress` on the same 20%-drawdown criterion the
  equity backtest uses.
- Every place the result depends on a choice: the coupling denominator floor, the
  winsorization level, the curvature weighting mode, the horizon set. Show at
  least one **sensitivity table** — if the headline flips when the floor moves
  from the 25th to the 40th percentile, that is the most important table in the
  document.
- The shuffled-target control numbers.
- What the walk-forward could not test (regime coverage, one asset, one country,
  fixture-limited credit data).

### G. Verdict
One of `SUPPORTED | PARTIALLY SUPPORTED | NO INCREMENTAL INFORMATION | UNSTABLE
| REJECTED`, chosen by the **pre-registered rule from the design note**, quoted
verbatim above the verdict so a reader can check that the rule was not moved
after seeing the data.

Then a short "what this means" paragraph, and a "what we would do next" paragraph
that is honest about cost. If the verdict is `REJECTED`, say what should be
deleted and open the follow-up note in `docs/follow-ups/`.

### H. Reproduction
The exact command, the fixture, the config hash, the model version, and the
commit. A reader must be able to regenerate every number from a clone with no
API keys.

## Language rules — enforced

Forbidden anywhere in the document:
- "markets have a fifth dimension" or any variant asserting physical reality;
- "proves", "confirms the hypothesis", "the market's true state";
- any causal verb applied to a correlation ("Ω drives volatility").

Required framing: *"A fifth-dimensional latent-state representation inspired by
Kaluza–Klein geometry was tested as a quantitative modelling hypothesis."*

Add a test — `tests/research/omega/test_report_language.py` — that greps the
committed report for the forbidden phrases and fails on a hit. A rule enforced
only by review is a rule that survives exactly one busy afternoon.

## Acceptance

- [ ] `docs/research/kk-omega-report.md` exists with sections A–H.
- [ ] Every number traceable to a CSV under `compute/backtests/omega/`.
- [ ] `python -m jobs.omega_research --report` regenerates it byte-identically.
- [ ] The pre-registered decision rule is quoted above the verdict, unmodified
      from the design note — diff it and paste the result.
- [ ] At least one sensitivity table present in §F.
- [ ] Language test green.
- [ ] `ruff check`, `ruff format --check`, `pytest`, `lint-imports` green.
