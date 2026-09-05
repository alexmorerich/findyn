"""Q1–Q5, answered against the pre-registered rules in the design note §8.

Nothing here chooses a threshold. Every constant this module compares against
was fixed in ``docs/research/kk-omega-design.md`` before any number existed, and
KK5 quotes that section verbatim above its verdict so a reader can check the
rule was not moved after seeing the data.

Three rules are enforced structurally rather than remembered:

**The baseline is A′, not A.** Every question's decision rule is stated against
the RII control arm. ``engines/equity/rii.py`` already publishes a composite
built from nearly the same observables, so "Ω beats price alone" is close to
guaranteed and close to worthless. :func:`decide` reads the ``vs_a_prime``
column and nothing else; the ``vs_a`` column is computed and reported so the
gap between the two claims is visible.

**Multiplicity is taken over the whole family.** :func:`evaluate` collects every
p-value from Q1–Q5 into one Benjamini–Hochberg adjustment and records the family
size, rather than adjusting within each question — which would be five small
families and a much easier bar.

**Two of the four horizons, not one.** A result at a single horizon out of four
is what a family of twenty-four tests produces by chance; the rule asks for two,
with a non-negative sign at all four.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from findynamics.engines.equity.backtest import EPISODES, MATERIAL_DRAWDOWN
from findynamics.engines.equity.crash import ADVERSE
from findynamics.research.omega import targets as targets_mod
from findynamics.research.omega.backtest import ARMS, WalkForwardResult
from findynamics.research.omega.domain import OMEGA_REGIMES
from findynamics.research.omega.statistics import (
    auc,
    benjamini_hochberg,
    brier,
    delong_auc_test,
    delta_r2_test,
    hac_tstat,
    oos_r2,
    performance,
    rank_ic,
)

log = logging.getLogger("findynamics.research.omega.evaluate")

#: The significance threshold, fixed in the design note §8 and not a parameter.
#: "The significance threshold is q < 0.10 throughout. Fixed here, not later."
Q_THRESHOLD = 0.10

#: Horizons a result must clear before it counts. Two of four.
MIN_HORIZONS = 2

#: Sub-periods a result must appear in before it is stable rather than UNSTABLE.
MIN_SUB_PERIODS = 3

#: Rank correlation against a published quantity at or above which ``C`` is a
#: re-derivation rather than a new coordinate. The design note's number, and the
#: same one KK2's redundancy panel prints.
REDUNDANCY_GATE = 0.9

#: Break-even transaction cost, in basis points, below which an edge is not one.
#: "An edge that dies at 3 bps is a finding, not a footnote."
BREAK_EVEN_FLOOR_BPS = 10.0

#: The arms Ω contributes to. A and A′ are the baselines they are measured
#: against and are never themselves "an Ω arm".
OMEGA_ARMS: tuple[str, ...] = ("B", "C", "D", "E")

#: Forward window for the false-alarm comparison, in trading days. One year,
#: matching ``engines/equity/backtest.py::false_alarm_rate`` so the two reports
#: are measuring the same thing.
FALSE_ALARM_HORIZON = 252


def _predictions(result: WalkForwardResult, arm: str, target: str) -> pd.Series:
    column = f"{arm}::{target}"
    if column not in result.predictions.columns:
        return pd.Series(dtype=float)
    return result.predictions[column].dropna()


def _score_arm(
    result: WalkForwardResult,
    arm: str,
    kind: str,
    horizon: int,
    *,
    on: pd.Index | None = None,
) -> dict[str, float] | None:
    """Out-of-sample R² of one arm on one target, against the training mean.

    ``on`` restricts the calculation to a shared index. Two arms scored on
    their own samples are not comparable and their difference is not a ΔR²: on
    the shipped configuration arm E starts thirteen years after arm A, because
    ``curvature`` warms up twice, so an unpaired difference would be mostly a
    statement about which years each arm saw.
    """
    target = f"{kind}_{horizon}"
    predicted = _predictions(result, arm, target)
    if on is not None:
        predicted = predicted.reindex(on).dropna()
    if predicted.empty or target not in result.panel.columns:
        return None
    means = result.training_means.get(f"{arm}::{target}")
    if means is None:
        return None
    aligned = means.reindex(predicted.index).dropna()
    if aligned.empty:
        return None
    # One scalar for the whole path would be a full-sample mean. The training
    # mean travels per row with the window that produced the prediction, and the
    # R² is taken against the average of those.
    return {
        "oos_r2": oos_r2(
            result.panel[target].reindex(predicted.index), predicted, float(aligned.mean())
        ),
        "n": float(len(predicted)),
    }


def _question_frame(
    result: WalkForwardResult,
    kind: str,
    question: str,
) -> pd.DataFrame:
    """One row per (arm, horizon) with R², both ΔR² readings and a p-value."""
    rows: list[dict[str, object]] = []
    for horizon in result.params.horizons:
        target = f"{kind}_{horizon}"
        if target not in result.panel.columns:
            continue
        actual = result.panel[target]
        baselines = {name: _predictions(result, name, target) for name in ("A", "A_prime")}
        for arm in ARMS:
            scored = _score_arm(result, arm, kind, horizon)
            if scored is None:
                continue
            predicted = _predictions(result, arm, target)
            row: dict[str, object] = {
                "question": question,
                "arm": arm,
                "horizon": horizon,
                "oos_r2": scored["oos_r2"],
                "n": scored["n"],
            }
            for label, baseline in (("a", "A"), ("a_prime", "A_prime")):
                reference = baselines[baseline]
                if arm == baseline or reference.empty:
                    row[f"delta_r2_vs_{label}"] = np.nan
                    row[f"p_vs_{label}"] = np.nan
                    row[f"n_vs_{label}"] = np.nan
                    continue
                # Paired: both arms re-scored on the dates they share, so the
                # difference is about the columns and not about the calendar.
                shared = predicted.index.intersection(reference.index)
                mine = _score_arm(result, arm, kind, horizon, on=shared)
                other = _score_arm(result, baseline, kind, horizon, on=shared)
                row[f"delta_r2_vs_{label}"] = (
                    mine["oos_r2"] - other["oos_r2"] if mine and other else np.nan
                )
                row[f"n_vs_{label}"] = float(len(shared))
                test = delta_r2_test(actual, predicted, reference, horizon=horizon)
                row[f"p_vs_{label}"] = test["dm_p"]
                row[f"t_vs_{label}"] = test["dm_t"]
                row[f"lag_vs_{label}"] = test["lag"]
            rows.append(row)
    return pd.DataFrame(rows)


def question_1(result: WalkForwardResult) -> pd.DataFrame:
    """Does Ω carry incremental information about **forward returns**?"""
    return _question_frame(result, "forward_return", "Q1")


def question_2(result: WalkForwardResult) -> pd.DataFrame:
    """Does Ω̇ carry incremental information about **forward volatility**?"""
    return _question_frame(result, "forward_vol", "Q2")


def question_3(result: WalkForwardResult) -> pd.DataFrame:
    """Does Ω detect **regime transitions** earlier than the existing HMM?"""
    rows: list[dict[str, object]] = []
    for horizon in result.params.horizons:
        target = f"transition_{horizon}"
        if target not in result.panel.columns:
            continue
        labels = result.panel[target]
        baseline_auc = {
            name: auc(
                labels.reindex(_predictions(result, name, target).index),
                _predictions(result, name, target),
            )
            for name in ("A", "A_prime")
        }
        for arm in ARMS:
            probability = _predictions(result, arm, target)
            if probability.empty:
                continue
            aligned = labels.reindex(probability.index)
            score = auc(aligned, probability)
            row = {
                "question": "Q3",
                "arm": arm,
                "horizon": horizon,
                "auc": score,
                "brier": brier(aligned, probability),
                "delta_auc_vs_a": score - baseline_auc["A"],
                "delta_auc_vs_a_prime": score - baseline_auc["A_prime"],
                "base_rate": float(aligned.mean()),
                "n": float(len(probability)),
            }
            # DeLong, paired, for the reason the pre-registered rule needs one:
            # "ΔAUC > 0" with no standard error is a coin flip dressed as a
            # finding. The shuffled-target control demonstrated exactly that —
            # it reached ΔAUC > 0 on two horizons by noise and would have
            # "passed" Q3 without this column.
            for label, baseline in (("a", "A"), ("a_prime", "A_prime")):
                reference = _predictions(result, baseline, target)
                if arm == baseline or reference.empty:
                    row[f"p_auc_vs_{label}"] = np.nan
                    continue
                test = delong_auc_test(labels, probability, reference)
                row[f"p_auc_vs_{label}"] = test["p"]
                row[f"z_auc_vs_{label}"] = test["z"]
            rows.append(row)
    return pd.DataFrame(rows)


def question_4(result: WalkForwardResult) -> pd.DataFrame:
    """Does the coupling ``C`` carry information its parts do not?"""
    rows: list[dict[str, object]] = []
    if "coupling" not in result.panel.columns:
        return pd.DataFrame(rows)
    coupling = result.panel["coupling"]
    for horizon in result.params.horizons:
        for kind in ("forward_return", "forward_vol"):
            target = f"{kind}_{horizon}"
            if target not in result.panel.columns:
                continue
            rows.append(
                {
                    "question": "Q4",
                    "target": kind,
                    "horizon": horizon,
                    "rank_ic": rank_ic(coupling, result.panel[target]),
                    "n": float(
                        len(pd.concat([coupling, result.panel[target]], axis=1, sort=True).dropna())
                    ),
                }
            )
    return pd.DataFrame(rows)


def _long_flat(result: WalkForwardResult, arm: str) -> pd.Series | None:
    """Long/flat exposure from the 21-day forward-return prediction."""
    horizon = 21 if 21 in result.params.horizons else result.params.horizons[-1]
    predicted = _predictions(result, arm, f"forward_return_{horizon}")
    if predicted.empty:
        return None
    return (predicted > 0.0).astype(float)


def question_5_sub_periods(result: WalkForwardResult) -> pd.DataFrame:
    """The same long/flat rule, graded inside each sub-period.

    The pre-registered Q5 rule is stated per sub-period — "exceeds A′'s at 5 bps
    in at least three of the five" — so a pooled Sharpe cannot answer it. A
    pooled number is also the easier bar, which is exactly why the rule was not
    written that way.
    """
    daily = result.panel.get("forward_return_1")
    rows: list[dict[str, object]] = []
    if daily is None:
        return pd.DataFrame(rows)

    for name, start, end in result.params.sub_periods:
        window = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        if len(window) < 60:
            continue
        for arm in ARMS:
            exposure = _long_flat(result, arm)
            if exposure is None:
                continue
            local = exposure.reindex(window.index).dropna()
            if len(local) < 60:
                continue
            metrics = performance(local, window, cost_bps=result.params.cost_bps)
            rows.append({"sub_period": name, "arm": arm, **metrics.as_dict()})
    return pd.DataFrame(rows)


def question_5(result: WalkForwardResult) -> pd.DataFrame:
    """Does any arm survive as a **decision rule after costs**?

    Long/flat on the sign of the 21-day forward-return prediction, earning the
    next day's return. Deliberately the crudest rule that uses the prediction:
    anything richer would be a study of the rule rather than of the arm, and the
    question is whether the *columns* carry an edge.
    """
    rows: list[dict[str, object]] = []
    daily = result.panel.get("forward_return_1")
    if daily is None:
        return pd.DataFrame(rows)

    reference = _long_flat(result, "A_prime")
    for arm in ARMS:
        exposure = _long_flat(result, arm)
        if exposure is None:
            continue
        metrics = performance(exposure, daily, cost_bps=result.params.cost_bps)
        row = {"question": "Q5", "arm": arm, **metrics.as_dict()}
        # The pre-registered Q5 rule has no significance test — it asks only
        # that the Sharpe beat A′'s in three of five sub-periods, which under a
        # coin flip happens half the time. The shuffled-target control passed it
        # on noise, so this column is reported beside the verdict: a HAC t-test
        # on the daily net-return difference against A′, which is the thing the
        # rule should have asked for.
        if reference is not None and arm not in ("A_prime",):
            difference = (exposure - reference.reindex(exposure.index)).dropna() * daily
            t, p, _ = hac_tstat(difference.dropna(), horizon=1)
            row["excess_t_vs_a_prime"] = t
            row["excess_p_vs_a_prime"] = p
        rows.append(row)

    # The comparator every arm is measured against: always invested.
    always = pd.Series(1.0, index=daily.dropna().index)
    rows.append(
        {
            "question": "Q5",
            "arm": "buy_and_hold",
            **performance(always, daily, cost_bps=result.params.cost_bps).as_dict(),
        }
    )
    return pd.DataFrame(rows)


def sub_period_table(
    result: WalkForwardResult,
    kinds: tuple[str, ...] = ("forward_return", "forward_vol"),
) -> pd.DataFrame:
    """Every arm's OOS R² per horizon, per sub-period.

    A result present in exactly one sub-period is ``UNSTABLE``, not
    ``SUPPORTED``. The design note's §9 says so and :func:`decide` enforces it
    from this table rather than leaving it to the reader.
    """
    rows: list[dict[str, object]] = []
    for kind, (name, start, end) in [
        (kind, period) for kind in kinds for period in result.params.sub_periods
    ]:
        window = result.panel.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if window.empty:
            continue
        for horizon in result.params.horizons:
            target = f"{kind}_{horizon}"
            if target not in window.columns:
                continue
            baseline_scores: dict[str, float] = {}
            for arm in ARMS:
                column = f"{arm}::{target}"
                if column not in result.predictions.columns:
                    continue
                predicted = result.predictions[column].reindex(window.index).dropna()
                if len(predicted) < 30:
                    continue
                means = result.training_means[column].reindex(predicted.index).dropna()
                if means.empty:
                    continue
                score = oos_r2(
                    window[target].reindex(predicted.index), predicted, float(means.mean())
                )
                baseline_scores[arm] = score
                rows.append(
                    {
                        "target": kind,
                        "sub_period": name,
                        "arm": arm,
                        "horizon": horizon,
                        "oos_r2": score,
                        "n": float(len(predicted)),
                    }
                )
            for row in rows:
                if (
                    row["target"] == kind
                    and row["sub_period"] == name
                    and row["horizon"] == horizon
                ):
                    row["delta_r2_vs_a_prime"] = row["oos_r2"] - baseline_scores.get(
                        "A_prime", np.nan
                    )
    return pd.DataFrame(rows)


def stability(table: pd.DataFrame, arm: str, kind: str) -> dict[str, float]:
    """How many sub-periods an arm's edge over A′ survives in.

    Design note §9: a result present in **two or fewer** of the five sub-periods
    is ``UNSTABLE``, not ``SUPPORTED``. This is the number that decides it, and
    it is computed rather than eyeballed off the table.
    """
    rows = table[(table["arm"] == arm) & (table["target"] == kind)]
    if rows.empty or "delta_r2_vs_a_prime" not in rows.columns:
        return {"positive_sub_periods": 0.0, "sub_periods": 0.0}
    by_period = rows.groupby("sub_period")["delta_r2_vs_a_prime"].mean()
    return {
        "positive_sub_periods": float((by_period > 0.0).sum()),
        "sub_periods": float(len(by_period)),
    }


def _cramers_v(table: pd.DataFrame) -> float:
    """Association between two categorical labellings, 0 to 1."""
    values = table.to_numpy(dtype=float)
    if values.sum() <= 0 or min(values.shape) < 2:
        return float("nan")
    chi2 = float(stats.chi2_contingency(values, correction=False)[0])
    n = values.sum()
    return float(math.sqrt(chi2 / (n * (min(values.shape) - 1))))


def _mutual_information(table: pd.DataFrame) -> float:
    """Mutual information of the two labellings, in nats."""
    joint = table.to_numpy(dtype=float)
    total = joint.sum()
    if total <= 0:
        return float("nan")
    joint = joint / total
    rows = joint.sum(axis=1, keepdims=True)
    columns = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (rows @ columns))
    return float(np.nansum(terms))


@dataclass(frozen=True)
class RegimeComparison:
    """Ω's banding against the engine's five-state HMM, out of sample."""

    matrix: pd.DataFrame
    conditional_volatility: pd.DataFrame
    lead_lag: pd.DataFrame
    cramers_v: float
    mutual_information: float
    false_alarms: dict[str, float]


def regime_comparison(result: WalkForwardResult) -> RegimeComparison:
    """The KK3 §5 diagnostic: does Ω say anything the HMM does not?

    Every number here is on the out-of-sample path only. The episode dates come
    from ``engines/equity/backtest.py::EPISODES`` and the drawdown criterion
    from ``MATERIAL_DRAWDOWN``, so this table and the equity engine's own report
    are measuring the same events by the same rule.
    """
    panel = result.panel
    both = panel[["regime", "omega_regime"]].dropna()
    matrix = pd.crosstab(both["regime"], both["omega_regime"])

    horizon = 21 if 21 in result.params.horizons else result.params.horizons[0]
    volatility = panel.get(f"forward_vol_{horizon}")
    drawdown = panel.get(f"forward_drawdown_{horizon}")
    conditional = pd.DataFrame(
        {
            "forward_vol": both.join(volatility)
            .groupby(["regime", "omega_regime"], observed=True)[volatility.name]
            .mean()
            if volatility is not None
            else np.nan,
            "forward_drawdown": both.join(drawdown)
            .groupby(["regime", "omega_regime"], observed=True)[drawdown.name]
            .mean()
            if drawdown is not None
            else np.nan,
        }
    ).reset_index()

    stress = panel["omega_regime"] == OMEGA_REGIMES[2]
    adverse = panel["regime"].isin(ADVERSE)
    rows: list[dict[str, object]] = []
    for episode in EPISODES:
        window = slice(pd.Timestamp(episode.start), pd.Timestamp(episode.trough))
        omega_calls = panel.index[stress.loc[window].reindex(panel.index, fill_value=False)]
        hmm_calls = panel.index[adverse.loc[window].reindex(panel.index, fill_value=False)]
        first_omega = omega_calls[0] if len(omega_calls) else None
        first_hmm = hmm_calls[0] if len(hmm_calls) else None
        rows.append(
            {
                "episode": episode.name,
                "omega_first": None if first_omega is None else first_omega.date(),
                "hmm_first": None if first_hmm is None else first_hmm.date(),
                # Positive means Ω called it first. Trading days, counted on the
                # out-of-sample calendar rather than in calendar days, so a
                # holiday does not read as a lead.
                "lead_days": (
                    float(
                        panel.index.get_indexer([first_hmm])[0]
                        - panel.index.get_indexer([first_omega])[0]
                    )
                    if first_omega is not None and first_hmm is not None
                    else np.nan
                ),
            }
        )

    return RegimeComparison(
        matrix=matrix,
        conditional_volatility=conditional,
        lead_lag=pd.DataFrame(rows),
        cramers_v=_cramers_v(matrix),
        mutual_information=_mutual_information(matrix),
        false_alarms=_false_alarm_rates(panel),
    )


def _false_alarm_rates(panel: pd.DataFrame) -> dict[str, float]:
    """P(no material drawdown within a year | the caller said stress).

    The same criterion ``engines/equity/backtest.py::false_alarm_rate`` uses, so
    Ω's alarm quality and the engine's are on one scale. Computed for both
    callers side by side, because "Ω fires earlier" only means something if it
    does not also fire far more often.
    """
    forward = targets_mod.forward_drawdown(panel["log_price"], FALSE_ALARM_HORIZON)
    out: dict[str, float] = {}
    for label, called in (
        ("omega_latent_stress", panel["omega_regime"] == OMEGA_REGIMES[2]),
        ("hmm_bear_or_crisis", panel["regime"].isin(ADVERSE)),
    ):
        usable = forward.notna() & called.fillna(False)
        alarms = int(usable.sum())
        out[f"{label}_alarms"] = float(alarms)
        out[f"{label}_false_alarm_rate"] = (
            float(1.0 - (forward[usable] >= MATERIAL_DRAWDOWN).mean()) if alarms else float("nan")
        )
        out[f"{label}_share_of_days"] = float(called.fillna(False).mean())
    return out


@dataclass(frozen=True)
class Evaluation:
    """Every question's answer, with the multiplicity adjustment applied once."""

    q1: pd.DataFrame
    q2: pd.DataFrame
    q3: pd.DataFrame
    q4: pd.DataFrame
    q5: pd.DataFrame
    q5_sub_periods: pd.DataFrame
    sub_periods: pd.DataFrame
    regimes: RegimeComparison
    #: Number of p-values the Benjamini-Hochberg adjustment was taken over.
    family_size: int
    verdicts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def evaluate(
    result: WalkForwardResult,
    *,
    redundancy: float | None = None,
) -> Evaluation:
    """Answer Q1–Q5 and adjust for multiplicity across the whole family.

    ``redundancy`` is ``C``'s largest rank correlation against a quantity the
    engine already publishes, measured by KK2's redundancy panel. Q4's gate
    needs it, and a run that does not supply it reports the gate as untested
    rather than as passed.
    """
    q1, q2 = question_1(result), question_2(result)
    q3, q4 = question_3(result), question_4(result)
    q5, q5_periods = question_5(result), question_5_sub_periods(result)

    # One family, named. Adjusting within each question would be five small
    # families and a much easier bar than the design note pre-registered.
    family = (
        pd.concat(
            [frame[["p_vs_a", "p_vs_a_prime"]] for frame in (q1, q2) if not frame.empty],
            keys=["Q1", "Q2"],
        )
        if not (q1.empty and q2.empty)
        else pd.DataFrame()
    )

    if not family.empty:
        stacked = family.stack(future_stack=True)
        adjusted = benjamini_hochberg(stacked)
        unstacked = adjusted.unstack()
        for label, frame in (("Q1", q1), ("Q2", q2)):
            if frame.empty or label not in unstacked.index.get_level_values(0):
                continue
            block = unstacked.loc[label]
            frame["q_vs_a"] = block["p_vs_a"].to_numpy()
            frame["q_vs_a_prime"] = block["p_vs_a_prime"].to_numpy()

    verdicts = {
        "Q1": decide(q1, result),
        "Q2": decide(q2, result),
        "Q3": _decide_q3(q3, regime := regime_comparison(result)),
        "Q4": _decide_q4(q1, q2, redundancy),
        "Q5": _decide_q5(q5, q5_periods),
    }
    size = int(family.notna().to_numpy().sum()) if not family.empty else 0
    log.info(
        "omega evaluate: family of %d p-value(s) adjusted together; verdicts %s",
        size,
        ", ".join(f"{k}={v}" for k, v in verdicts.items()),
    )
    return Evaluation(
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        q5=q5,
        q5_sub_periods=q5_periods,
        sub_periods=sub_period_table(result),
        regimes=regime,
        family_size=size,
        verdicts=verdicts,
    )


def decide(frame: pd.DataFrame, result: WalkForwardResult) -> str:
    """The pre-registered rule for Q1 and Q2, applied without discretion.

    Design note §8: "Passes iff at least one Ω arm has ΔR² > 0 versus **A′** at
    q < 0.10 on at least two of the four horizons, and its ΔR² versus A′ is
    non-negative on all four." Anything else is ``FAILS``, including a gain over
    A alone.

    **One documented departure from the literal wording.** "All four" is read as
    "every horizon the question was actually tested at". Q2's target is realized
    volatility over the next ``h`` days, and at ``h = 1`` that is the dispersion
    of a single return, which does not exist —
    :func:`targets.forward_volatility` returns an empty column rather than a
    column of zeros, because calling one return "zero volatility" would tell a
    model that every day is perfectly calm. Requiring a non-negative reading at
    a horizon that cannot be computed would make Q2 neither passable nor
    failable, which is not a decision rule. The count is therefore taken over
    the horizons present in ``frame``, and the phase report flags the departure
    rather than leaving it in a docstring.
    """
    if frame.empty or "q_vs_a_prime" not in frame.columns:
        return "NOT TESTED"
    tested = sorted(set(frame["horizon"]))
    if not tested:
        return "NOT TESTED"
    for arm in OMEGA_ARMS:
        rows = frame[frame["arm"] == arm]
        if rows.empty:
            continue
        significant = rows[
            (rows["delta_r2_vs_a_prime"] > 0.0) & (rows["q_vs_a_prime"] < Q_THRESHOLD)
        ]
        non_negative = int((rows["delta_r2_vs_a_prime"].fillna(-1.0) >= 0.0).sum())
        if len(significant) >= MIN_HORIZONS and non_negative == len(tested):
            return f"PASSES (arm {arm})"
    return "FAILS"


def _decide_q3(frame: pd.DataFrame, regimes: RegimeComparison) -> str:
    """Earlier, **significantly** better, and not much noisier — or it fails.

    Design note §8: "(i) ΔAUC > 0 versus A′ at q < 0.10 on at least two
    horizons; (ii) the median lead across the four EPISODES is strictly
    positive; (iii) the false-alarm rate of ``latent_stress`` exceeds the equity
    engine's own by no more than 10 percentage points."

    Clause (i)'s significance test is DeLong's, added after the shuffled-target
    control passed this question on noise. The p-value is used unadjusted here
    and the BH family in :func:`evaluate` covers Q1 and Q2 only — a deviation
    from the pre-registration, and the phase report says so rather than leaving
    it implied.
    """
    if frame.empty or "p_auc_vs_a_prime" not in frame.columns:
        return "NOT TESTED"
    better = frame[
        (frame["arm"].isin(OMEGA_ARMS))
        & (frame["delta_auc_vs_a_prime"] > 0.0)
        & (frame["p_auc_vs_a_prime"] < Q_THRESHOLD)
    ]
    by_arm = better.groupby("arm").size()
    leads = regimes.lead_lag["lead_days"].dropna()
    median_lead = float(leads.median()) if not leads.empty else float("nan")
    omega_rate = regimes.false_alarms.get("omega_latent_stress_false_alarm_rate", float("nan"))
    hmm_rate = regimes.false_alarms.get("hmm_bear_or_crisis_false_alarm_rate", float("nan"))
    noisier = np.isfinite(omega_rate) and np.isfinite(hmm_rate) and (omega_rate - hmm_rate) > 0.10
    if (by_arm >= MIN_HORIZONS).any() and median_lead > 0 and not noisier:
        return "PASSES"
    return "FAILS"


def _decide_q4(q1: pd.DataFrame, q2: pd.DataFrame, redundancy: float | None) -> str:
    """Arm D beats A′ on two horizons **and** ``C`` is not a re-derivation.

    Design note §8: "Q4 passes iff ΔR²(D vs A′) > 0 at q < 0.10 on at least two
    horizons **and** ``C``'s largest |ρ_s| against an existing published
    quantity is < 0.9. If the correlation gate fails, the result is reported as
    a re-derivation regardless of the R²."

    The rank IC is *reported* by :func:`question_4` and is not the gate — an
    earlier draft of this function used it, which was a rule invented after the
    data arrived rather than the one pre-registered.

    ``redundancy`` is ``C``'s largest rank correlation against a published
    quantity, measured by KK2's redundancy panel. ``None`` means it was not
    measured on this run, and the gate is reported as untested rather than
    assumed to have passed: a gate nobody ran is not a gate that passed.
    """
    frames = [frame for frame in (q1, q2) if not frame.empty and "q_vs_a_prime" in frame]
    if not frames:
        return "NOT TESTED"
    if redundancy is not None and abs(redundancy) >= REDUNDANCY_GATE:
        return f"FAILS (C re-derives a published quantity, |rho_s| = {abs(redundancy):.2f})"

    rows = pd.concat([frame[frame["arm"] == "D"] for frame in frames])
    significant = rows[(rows["delta_r2_vs_a_prime"] > 0.0) & (rows["q_vs_a_prime"] < Q_THRESHOLD)]
    if len(significant) < MIN_HORIZONS:
        return "FAILS"
    return "PASSES" if redundancy is not None else "PASSES (redundancy gate untested)"


def _decide_q5(pooled: pd.DataFrame, by_period: pd.DataFrame) -> str:
    """Beats A′ in three of five sub-periods **and** breaks even above 10 bps.

    Design note §8: "Q5 passes iff the best Ω arm's cost-adjusted Sharpe exceeds
    A′'s at 5 bps in at least three of the five sub-periods and its break-even
    cost is at least 10 bps."

    Graded per sub-period, because that is what was pre-registered. An earlier
    draft compared pooled Sharpes, which is the easier bar and is the reason the
    rule was not written that way.
    """
    if pooled.empty or by_period.empty or "sharpe" not in by_period.columns:
        return "NOT TESTED"

    baselines = by_period[by_period["arm"] == "A_prime"].set_index("sub_period")["sharpe"]
    if baselines.empty:
        return "NOT TESTED"

    for arm in OMEGA_ARMS:
        rows = by_period[by_period["arm"] == arm].set_index("sub_period")
        if rows.empty:
            continue
        shared = rows.index.intersection(baselines.index)
        wins = int((rows.loc[shared, "sharpe"] > baselines.loc[shared]).sum())
        pooled_row = pooled[pooled["arm"] == arm]
        break_even = float(pooled_row["break_even_bps"].iloc[0]) if not pooled_row.empty else 0.0
        if wins >= MIN_SUB_PERIODS and break_even >= BREAK_EVEN_FLOOR_BPS:
            return f"PASSES (arm {arm}, {wins}/{len(shared)} sub-periods)"
    return "FAILS"


__all__ = [
    "FALSE_ALARM_HORIZON",
    "MIN_HORIZONS",
    "MIN_SUB_PERIODS",
    "OMEGA_ARMS",
    "BREAK_EVEN_FLOOR_BPS",
    "REDUNDANCY_GATE",
    "Q_THRESHOLD",
    "Evaluation",
    "RegimeComparison",
    "decide",
    "evaluate",
    "question_1",
    "question_2",
    "question_3",
    "question_4",
    "question_5",
    "question_5_sub_periods",
    "regime_comparison",
    "stability",
    "sub_period_table",
]
