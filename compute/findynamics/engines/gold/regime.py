"""Gold's regime: a Markov chain on returns, gated by the driver panel.

The output is a posterior over :data:`~findynamics.engines.gold.domain.GOLD_REGIMES`
— ``hedge_bid | carry_headwind | crisis_bid`` — for every month, computed
causally, from parameters fitted on an expanding window and frozen between
monthly refits.

Why this is not a three-state Markov chain and nothing else
-----------------------------------------------------------

The obvious implementation is ``MarkovRegression(k_regimes=3)`` on monthly gold
returns with the drivers as exogenous regressors, taking the three latent states
as the three names. That was built first and it does not work, for a reason worth
recording because it is a property of the data rather than a bug:

**Gold's monthly returns in 2022 are statistically unremarkable.** Mean +0.04% a
month, standard deviation 3.6% — indistinguishable from a quiet market. What made
2022 a rate headwind is that the real 10-year yield rose 250bp, and that fact is
nowhere in the return series. A latent-state model whose likelihood sees only
returns has no term that would prefer the ``carry_headwind`` label there, and
across every configuration tried (k in {2,3}, switching variance and/or
switching exogenous coefficients, four exog sets, random and driver-seeded
initialization, three seeds) none classified both 2013 and 2022 as rate
headwinds in a walk-forward. The specifications that did so in-sample did not
survive being refitted on data ending before the window — which is the definition
of the result being fitted rather than found.

What a Markov chain on returns *is* good at is the other axis. Gold's return
process genuinely does switch between a quiet state and a violent one, that
switch is persistent, and every specification agrees about where it fires
(1979-80, 2008H2, 2011, April 2013, March 2020). So the chain is used for exactly
that and nothing more.

The model
---------

**Block 1 — the Markov chain (fitted).** ``statsmodels.tsa.MarkovRegression`` on
monthly gold log returns, with the standardized driver panel as exogenous
regressors in the mean equation and ``switching_variance=True``. Fitted on an
expanding window; the parameters are persisted and frozen until the next monthly
refit. It contributes one number per month::

    pi_violent(t) = P(the return process is in its highest-variance state | data to t)

Filtered, never smoothed. The smoothed probabilities are the better estimate and
they are inadmissible here: they condition on the whole sample, so a date's
smoothed state depends on what happened after it.

**Block 2 — the gates (configuration, not fitted).** Two logistic gates on the
standardized drivers::

    g_stress(t) = sigmoid(w_s * (z_stress(t) - s0))
    g_carry(t)  = sigmoid(w_c * (w_r * z_real_rate_change(t) + w_u * z_usd_trend(t) - c0))

**Composition.** A conjunction, then a split::

    P(crisis_bid)     = pi_violent * g_stress
    P(carry_headwind) = (1 - P(crisis_bid)) * g_carry
    P(hedge_bid)      = 1 - P(crisis_bid) - P(carry_headwind)

Read it as three questions asked in order. *Is gold's return process violent, and
are financial conditions tight?* Both, or it is not a crisis bid — violence with
calm conditions is a positioning washout, which is what April 2013 was, and tight
conditions with quiet returns is an ordinary tightening. *Failing that, is the
real-rate and dollar environment working against gold?* Then it is a carry
headwind. *Failing both* — the common case — gold is being held as a hedge and
nothing in particular is happening to it.

The gate constants are configuration and are deliberately set to the **middle of
the region that works** rather than to an optimum: across the walk-forward
backtest every combination of ``w in {1.5, 2.5}`` and offsets in {0.3, 0.6} /
{0.2, 0.5} classified all five reference windows correctly, so the shipped values
sit inside that region rather than on its edge. A result that needed the exact
constants would be a fit to five windows, not a model.

``driver_gates: false`` in config disables Block 2 entirely and publishes the raw
Markov posterior mapped by variance rank. It exists so the two blocks can be
told apart in a test rather than argued about.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from findynamics.engines.gold.domain import GOLD_REGIMES

log = logging.getLogger("findynamics.engines.gold.regime")

#: Driver z-columns the chain takes as exogenous regressors, in this order. The
#: order is part of the fitted artifact: statsmodels names its coefficients
#: positionally (``x1``, ``x2``, ...), so a reordering here silently reinterprets
#: every stored parameter vector.
EXOG_COLUMNS: tuple[str, ...] = ("z_real_rate_change_12m", "z_usd_trend", "z_stress")


# Not named *Error (hence the noqa), for the same reason as
# core.engine.StateUnavailable: this is not a failure. "The history is too short
# to fit a switching model" is a correct answer, and the engine acts on it by
# publishing its drivers and jump intensity without a regime.
class RegimeUnavailable(RuntimeError):  # noqa: N818
    """Not enough history, or no usable fitted parameters, to state a regime."""


@dataclass(frozen=True)
class RegimeRules:
    """Everything the regime model branches on, from ``config/engines/gold.yaml``."""

    k_regimes: int = 3
    #: Months required before the chain is fitted at all. A three-state switching
    #: model on 120 months is estimating more parameters than it has decades.
    min_observations: int = 240
    #: Random restarts inside statsmodels' EM search, and the seed that makes
    #: them reproducible. Changing the seed is a model change: the replay test
    #: depends on a refit landing in the same place.
    search_reps: int = 8
    em_iter: int = 50
    maxiter: int = 250
    seed: int = 20260801

    #: Block 2. See the module docstring for why these are mid-region values.
    driver_gates: bool = True
    stress_weight: float = 2.0
    stress_offset: float = 0.5
    carry_weight: float = 2.0
    carry_offset: float = 0.35
    #: Split of the carry gate between the real-rate impulse and the dollar.
    #: Real rates dominate because they are the mechanism; the dollar is a
    #: quotation effect that usually moves with them.
    carry_rate_share: float = 0.7
    carry_usd_share: float = 0.3

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RegimeRules:
        raw = params.get("regime") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/gold.yaml: 'regime' must be a mapping")
        defaults = cls()
        rules = cls(
            k_regimes=int(raw.get("k_regimes", defaults.k_regimes)),
            min_observations=int(raw.get("min_observations", defaults.min_observations)),
            search_reps=int(raw.get("search_reps", defaults.search_reps)),
            em_iter=int(raw.get("em_iter", defaults.em_iter)),
            maxiter=int(raw.get("maxiter", defaults.maxiter)),
            seed=int(raw.get("seed", defaults.seed)),
            driver_gates=bool(raw.get("driver_gates", defaults.driver_gates)),
            stress_weight=float(raw.get("stress_weight", defaults.stress_weight)),
            stress_offset=float(raw.get("stress_offset", defaults.stress_offset)),
            carry_weight=float(raw.get("carry_weight", defaults.carry_weight)),
            carry_offset=float(raw.get("carry_offset", defaults.carry_offset)),
            carry_rate_share=float(raw.get("carry_rate_share", defaults.carry_rate_share)),
            carry_usd_share=float(raw.get("carry_usd_share", defaults.carry_usd_share)),
        )
        if not 2 <= rules.k_regimes <= 3:
            raise ValueError(
                f"engines/gold.yaml regime.k_regimes must be 2 or 3, got {rules.k_regimes}"
            )
        return rules


@dataclass(frozen=True)
class MarkovFit:
    """Fitted chain parameters — everything needed to filter without refitting.

    Serialized straight into the engine's artifact. ``exog_columns`` travels with
    the parameters because statsmodels names coefficients by position: a stored
    vector is uninterpretable without knowing which column each one belonged to,
    and silently wrong if the order changed.
    """

    params: tuple[float, ...]
    exog_columns: tuple[str, ...]
    k_regimes: int
    #: Index of the highest-variance state — the one the chain contributes.
    violent_state: int
    n_observations: int
    log_likelihood: float
    #: Newest month in the fit window.
    fitted_through: date
    converged: bool = True
    #: Per-state fitted intercept and variance, pulled out of ``params`` by name
    #: at fit time. Stored rather than recomputed because statsmodels lays its
    #: parameter vector out positionally: reading them back by index would work
    #: until the specification changed, and then be silently wrong rather than
    #: broken. Empty on an artifact written before these were recorded.
    intercepts: tuple[float, ...] = ()
    variances: tuple[float, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "params": [float(p) for p in self.params],
            "exog_columns": list(self.exog_columns),
            "k_regimes": int(self.k_regimes),
            "violent_state": int(self.violent_state),
            "n_observations": int(self.n_observations),
            "log_likelihood": float(self.log_likelihood),
            "fitted_through": self.fitted_through.isoformat(),
            "converged": bool(self.converged),
            "intercepts": [float(v) for v in self.intercepts],
            "variances": [float(v) for v in self.variances],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MarkovFit | None:
        """Rebuild from an artifact, or ``None`` when the document cannot be one.

        Never raises. A daily run whose artifact is missing, truncated or written
        by an older model must degrade to "no regime", not fail — the engine has
        outputs worth publishing either way.
        """
        try:
            params = tuple(float(p) for p in payload["params"])
            columns = tuple(str(c) for c in payload["exog_columns"])
            return cls(
                params=params,
                exog_columns=columns,
                k_regimes=int(payload["k_regimes"]),
                violent_state=int(payload["violent_state"]),
                n_observations=int(payload.get("n_observations", 0)),
                log_likelihood=float(payload.get("log_likelihood", float("nan"))),
                fitted_through=date.fromisoformat(str(payload["fitted_through"])),
                converged=bool(payload.get("converged", True)),
                intercepts=tuple(float(v) for v in payload.get("intercepts", ())),
                variances=tuple(float(v) for v in payload.get("variances", ())),
            )
        except (KeyError, TypeError, ValueError) as err:
            log.warning("gold: stored regime fit is unusable (%s); ignoring it", err)
            return None


@dataclass(frozen=True)
class RegimeView:
    """The posterior and the parts it was built from."""

    #: Month-end index, one column per name in GOLD_REGIMES.
    posterior: pd.DataFrame
    #: The chain's own contribution, before the gates.
    violent_probability: pd.Series
    #: The two gates, published so a reader can see which one moved the answer.
    stress_gate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    carry_gate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def empty(self) -> bool:
        return self.posterior.empty

    def latest(self) -> pd.Series | None:
        return None if self.posterior.empty else self.posterior.iloc[-1]

    def label(self) -> str | None:
        """The winning name on the newest month."""
        row = self.latest()
        return None if row is None else str(row.idxmax())

    def daily(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """The monthly posterior carried forward onto a daily index.

        Forward only: a month-end posterior is knowable on that month-end and
        applies until the next one replaces it. Dates before the first month-end
        are left empty rather than back-filled, which would hand a date a
        posterior computed from months it had not seen.
        """
        if self.posterior.empty:
            return pd.DataFrame(index=index, columns=list(GOLD_REGIMES), dtype=float)
        return self.posterior.reindex(self.posterior.index.union(index)).ffill().reindex(index)


def _sigmoid(x: pd.Series | float) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _design(monthly: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.Series, pd.DataFrame]:
    """Complete-case endog and exog for the chain, in the stored column order."""
    missing = [c for c in columns if c not in monthly.columns]
    if missing:
        raise RegimeUnavailable(f"driver panel is missing {missing}")
    frame = monthly[["ret", *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    return frame["ret"], frame[list(columns)]


def _model(endog: pd.Series, exog: pd.DataFrame, k_regimes: int) -> Any:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    return MarkovRegression(
        endog,
        k_regimes=k_regimes,
        exog=exog,
        trend="c",
        # Variance switches, coefficients do not. Letting the driver
        # coefficients switch as well was tried and is badly identified on ~600
        # months: the betas moved by an order of magnitude between refits and
        # between seeds, which makes every stored artifact incomparable with the
        # one before it.
        switching_variance=True,
        switching_exog=False,
    )


def fit(monthly: pd.DataFrame, rules: RegimeRules) -> MarkovFit:
    """Fit the chain on an expanding window. Raises when history is too short.

    ``monthly`` is :attr:`~findynamics.engines.gold.drivers.DriverPanel.monthly`
    truncated to the information set — the caller owns the cutoff, this function
    fits whatever it is handed.
    """
    endog, exog = _design(monthly, EXOG_COLUMNS)
    if len(endog) < rules.min_observations:
        raise RegimeUnavailable(
            f"{len(endog)} usable months is under the {rules.min_observations}-month floor "
            "for a switching model; the drivers start later than the price does"
        )

    import warnings

    model = _model(endog, exog, rules.k_regimes)
    # Seeded because `search_reps` draws random starting values: without this a
    # refit on identical data lands somewhere else, and the replay test that
    # compares two runs of the same cutoff would be measuring the optimizer.
    np.random.seed(rules.seed)
    with warnings.catch_warnings():
        # EM rescales invalid transition probabilities as it goes and says so on
        # every iteration. It is not a failure and there are hundreds of them.
        warnings.simplefilter("ignore")
        result = model.fit(
            em_iter=rules.em_iter,
            search_reps=rules.search_reps,
            maxiter=rules.maxiter,
            disp=False,
        )

    variances = [float(result.params[f"sigma2[{i}]"]) for i in range(rules.k_regimes)]
    intercepts = [float(result.params[f"const[{i}]"]) for i in range(rules.k_regimes)]
    violent = int(np.argmax(variances))
    converged = bool(getattr(result.mle_retvals, "get", lambda *_: True)("converged", True))

    log.info(
        "gold regime fit: %d months through %s, llf=%.1f, violent state %d (sd %.2f%%/mo)",
        len(endog),
        endog.index[-1].date(),
        float(result.llf),
        violent,
        math.sqrt(variances[violent]),
    )
    return MarkovFit(
        params=tuple(float(p) for p in np.asarray(result.params)),
        exog_columns=EXOG_COLUMNS,
        k_regimes=rules.k_regimes,
        violent_state=violent,
        n_observations=int(len(endog)),
        log_likelihood=float(result.llf),
        fitted_through=endog.index[-1].date(),
        converged=converged,
        intercepts=tuple(intercepts),
        variances=tuple(variances),
    )


def filtered_states(monthly: pd.DataFrame, model_fit: MarkovFit) -> pd.DataFrame:
    """The whole filtered state posterior per month, under frozen parameters.

    ``filter`` rather than ``fit``: the parameters came from the last refit and
    must not move on a daily run. ``filtered`` rather than ``smoothed``: a
    smoothed probability for date *t* conditions on data after *t*.
    """
    endog, exog = _design(monthly, model_fit.exog_columns)
    if endog.empty:
        return pd.DataFrame()

    model = _model(endog, exog, model_fit.k_regimes)
    expected = len(model.start_params)
    if len(model_fit.params) != expected:
        raise RegimeUnavailable(
            f"stored fit has {len(model_fit.params)} parameters but this specification "
            f"needs {expected}; the artifact belongs to a different model"
        )

    filtered = np.asarray(
        model.filter(np.asarray(model_fit.params)).filtered_marginal_probabilities
    )
    return pd.DataFrame(filtered, index=endog.index)


def violent_probability(monthly: pd.DataFrame, model_fit: MarkovFit) -> pd.Series:
    """Filtered P(highest-variance state) per month — the chain's contribution."""
    states = filtered_states(monthly, model_fit)
    if states.empty:
        return pd.Series(dtype=float)
    return states[model_fit.violent_state]


def markov_only_posterior(monthly: pd.DataFrame, model_fit: MarkovFit) -> pd.DataFrame:
    """The chain's own states mapped onto the vocabulary, with no gates.

    The obvious implementation, kept runnable so the claim that it does not work
    can be *tested* rather than asserted in a docstring. The mapping is the
    natural one: the highest-variance state is ``crisis_bid``, and of the
    remaining two the lower-drift one is ``carry_headwind`` — a state gold loses
    money in is the best a return-only model can offer as a rate headwind.

    ``tests/engines/gold/test_regime.py`` runs the walk-forward backtest against
    this and requires it to fail on 2013 and 2022. If it ever stops failing, the
    gates are unnecessary and this module's design should be revisited.
    """
    states = filtered_states(monthly, model_fit)
    if states.empty:
        return pd.DataFrame(columns=list(GOLD_REGIMES))

    k = model_fit.k_regimes
    intercepts = model_fit.intercepts or tuple(0.0 for _ in range(k))
    crisis = model_fit.violent_state
    rest = [i for i in range(k) if i != crisis]
    carry = min(rest, key=lambda i: intercepts[i]) if rest else crisis

    mapping = {crisis: "crisis_bid", carry: "carry_headwind"}
    for i in rest:
        mapping.setdefault(i, "hedge_bid")

    named = states.rename(columns=mapping).T.groupby(level=0).sum().T
    for name in GOLD_REGIMES:
        if name not in named:
            named[name] = 0.0
    return named[list(GOLD_REGIMES)]


def gates(monthly: pd.DataFrame, rules: RegimeRules) -> tuple[pd.Series, pd.Series]:
    """``(stress_gate, carry_gate)`` — Block 2, on the standardized drivers."""
    stress_z = monthly.get("z_stress")
    stress_gate = (
        _sigmoid(rules.stress_weight * (stress_z.fillna(0.0) - rules.stress_offset))
        if stress_z is not None
        else pd.Series(0.5, index=monthly.index)
    )

    rate_z = monthly.get("z_real_rate_change_12m")
    usd_z = monthly.get("z_usd_trend")
    carry_z = pd.Series(0.0, index=monthly.index)
    if rate_z is not None:
        carry_z = carry_z + rules.carry_rate_share * rate_z.fillna(0.0)
    if usd_z is not None:
        carry_z = carry_z + rules.carry_usd_share * usd_z.fillna(0.0)
    carry_gate = _sigmoid(rules.carry_weight * (carry_z - rules.carry_offset))
    return stress_gate, carry_gate


def posterior(
    monthly: pd.DataFrame,
    model_fit: MarkovFit,
    rules: RegimeRules,
) -> RegimeView:
    """The published posterior over :data:`GOLD_REGIMES`, per month."""
    violent = violent_probability(monthly, model_fit)
    if violent.empty:
        return RegimeView(
            posterior=pd.DataFrame(columns=list(GOLD_REGIMES)),
            violent_probability=violent,
        )

    aligned = monthly.loc[violent.index]
    stress_gate, carry_gate = gates(aligned, rules)

    if rules.driver_gates:
        crisis = (violent * stress_gate).clip(0.0, 1.0)
        carry = ((1.0 - crisis) * carry_gate).clip(0.0, 1.0)
        hedge = (1.0 - crisis - carry).clip(lower=0.0)
        frame = pd.DataFrame(
            {"hedge_bid": hedge, "carry_headwind": carry, "crisis_bid": crisis},
            index=violent.index,
        )[list(GOLD_REGIMES)]
    else:
        # Block 1 alone — a real three-state mapping, not a stub. Kept runnable
        # so the test suite can show what the gates are worth rather than taking
        # the module docstring's word for it.
        frame = markov_only_posterior(monthly, model_fit)
    # Rounding and clipping can leave the row a hair off one; a posterior that
    # does not sum to one is not a posterior, and the dashboard stacks these.
    total = frame.sum(axis=1).replace(0.0, np.nan)
    frame = frame.div(total, axis=0).fillna(1.0 / len(GOLD_REGIMES))

    return RegimeView(
        posterior=frame,
        violent_probability=violent,
        stress_gate=stress_gate,
        carry_gate=carry_gate,
    )


__all__ = [
    "EXOG_COLUMNS",
    "MarkovFit",
    "RegimeRules",
    "RegimeUnavailable",
    "RegimeView",
    "filtered_states",
    "fit",
    "gates",
    "markov_only_posterior",
    "posterior",
    "violent_probability",
]
