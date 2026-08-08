"""Five-state Gaussian HMM over the standardized features (§9 layer L2).

The HMM is unsupervised: it finds five persistent clusters in feature space and
a transition matrix between them, and has no idea which one deserves to be
called a crisis. §9 requires the mapping to named regimes to be **documented
rules over (mean return, vol)**, not eyeballed, and asserts label stability
across refits — because a refit that permutes two labels rewrites every regime
history the engine has ever published without changing a single number.

The labelling rule
------------------

1. Assign every observation to a state (Viterbi), and compute that state's mean
   return and volatility **in the series' own units** — not the dimensionless
   model inputs. Those were built to be index-independent, which is precisely
   what makes them unable to answer "is this a bull market".
2. Sort the five states by **mean return**, descending, and read the vocabulary
   onto them in order: ``bull_expansion``, ``normal_expansion``, ``late_cycle``,
   ``bear``, ``crisis``. Ties break on lower volatility first, so the order is
   total and a refit cannot swap two states that the data does not separate.

One rule, no special cases — and it is the third rule tried, because the two
more sophisticated ones each fail on a real series:

* **"crisis is the highest-volatility state"** labelled the *COVID rebound* a
  crisis. On the 2016-2026 S&P window the most violent stretch is the recovery,
  and that state's mean return is **+27.6%**. Volatility says how violent a
  period was, not which way it went.
* **"sort by return per unit of volatility"** survives that, and then fails on
  the 1927+ S&P: it made a state returning −6.1% at **11.1%** volatility the
  ``crisis`` — a slow drift — while a −12.7% state at 26.2% volatility was
  labelled ``bear``. Dividing by a small denominator makes a mild decline look
  catastrophic.

Mean return is the crude rule and it is the one that works on all three series.
The refinements were attempts to encode "risk-adjusted", and both ended up
encoding an artefact of their own denominator. That volatility rises as the
label worsens is left as a *consequence* to be checked — ``crisis`` does come
out above the median volatility on every series tested — rather than as an input
that can be gamed by a quiet state.

Determinism is a requirement, not a nicety: the replay test recomputes fits, and
an HMM that lands somewhere different each time would fail it for reasons that
have nothing to do with lookahead. The seed is configuration and travels in the
artifact.

**The seed alone does not buy it.** ``GaussianHMM`` initializes its means with
scikit-learn's k-means, which is parallelized with OpenMP, and a threaded
floating-point reduction sums its partial results in whatever order the threads
finish. That is a difference in the last bit or two of the initial means, which
200 EM iterations then amplify — measurably, to a relative ~1e-9 in the stored
parameters. Two refits of the same data under the same seed therefore produced
*almost* the same model and not the same bytes, and since fitted artifacts are
addressed by (model_version, fit date) and compared by content, "almost" is a
409 (issue #6). Worse, it did not stop at the HMM: the transition classifiers are
fitted on these posteriors, so a 1e-9 wobble moved a split threshold and
re-serialized a 340 kB booster that differed from its predecessor.

:func:`fit_hmm` therefore pins the thread pools to one thread for the duration of
the fit. It is the cheapest of the available fixes and the only one that makes
the fit genuinely reproducible rather than reproducible-to-a-tolerance — and it
is measurably free here, because the design matrix is four columns wide and
there was never any parallelism worth having in it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from findynamics.engines.equity.domain import REGIMES
from findynamics.engines.equity.regime.design import RegimeDesign

log = logging.getLogger("findynamics.engines.equity.regime.hmm")

#: §9 L2 — five states, and the count is not a free parameter. The vocabulary in
#: domain.py has five names and they are a published contract.
N_STATES = len(REGIMES)

#: Fixed so a refit on the same data lands in the same place. Changing it is a
#: model change and belongs in a version bump, not in a config tweak.
DEFAULT_SEED = 20260731

DEFAULT_N_ITER = 200

#: Full covariance: four features whose correlations are the point (velocity
#: down *while* vol is up is what a bear looks like, and a diagonal model cannot
#: represent that). With thousands of observations per state there is ample data
#: for the 10 free parameters per state this costs.
DEFAULT_COVARIANCE = "full"

#: A state holding less than this share of observations is not a regime, it is a
#: handful of outliers wearing one. Reported rather than merged: the honest
#: response is to say the fit is degenerate, not to quietly renumber.
MIN_STATE_SHARE = 0.005

#: Dirichlet pseudo-counts added to the transition matrix diagonal, quoted for a
#: daily series and scaled to the actual frequency in :func:`transition_prior`.
#:
#: Without it the fit is unusable, and not subtly: on the calibration series an
#: unconstrained HMM produced a **median regime duration of one day** and 3,029
#: regime changes over 39 years. That is a volatility classifier being read as a
#: regime model — and it would have put a different regime badge on the dashboard
#: most mornings.
#:
#: The prior is a real belief, not a smoothing hack: market regimes are things
#: that last months. At this strength the median run becomes 37 trading days and
#: regime changes fall to about four a year, which is what the word describes.
#: Episode detection *improves* rather than degrading — 2022 goes from 86% to 96%
#: of sessions flagged bear-or-crisis — because the flickering was noise, not
#: signal.
DEFAULT_TRANSITION_PRIOR = 1000.0

#: The frequency the prior above is quoted at.
PRIOR_REFERENCE_PERIODS = 252.0


class RegimeFitError(ValueError):
    """Raised when a usable five-state fit cannot be produced."""


@dataclass(frozen=True)
class StateStats:
    """What one fitted state turned out to be, in the units of the real series."""

    state: int
    label: str
    #: Annualized mean log return of the observations assigned to this state.
    mean_return: float
    #: Annualized standard deviation of those returns.
    volatility: float
    #: Fraction of observations assigned to this state.
    share: float
    #: Mean run length in observations — how persistent the state is.
    persistence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "label": self.label,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "share": self.share,
            "persistence": self.persistence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StateStats:
        return cls(
            state=int(raw["state"]),
            label=str(raw["label"]),
            mean_return=float(raw["mean_return"]),
            volatility=float(raw["volatility"]),
            share=float(raw["share"]),
            persistence=float(raw.get("persistence", 0.0)),
        )


@dataclass(frozen=True)
class HmmFit:
    """Everything needed to reconstruct the fitted model, JSON-serializable."""

    feature_names: tuple[str, ...]
    startprob: tuple[float, ...]
    transmat: tuple[tuple[float, ...], ...]
    #: State means in **standardized** space, one row per state.
    means: tuple[tuple[float, ...], ...]
    covars: tuple[tuple[tuple[float, ...], ...], ...]
    covariance_type: str
    #: Raw state index -> regime label, from the rules in the module docstring.
    labels: tuple[str, ...]
    stats: tuple[StateStats, ...]
    #: The series the parameters came from. A reader must never have to guess.
    fitted_on: str
    #: True when that series is not the published index.
    fitted_on_proxy: bool
    observations: int
    converged: bool
    seed: int
    #: Newest observation date in the fitting window.
    fitted_through: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "startprob": list(self.startprob),
            "transmat": [list(row) for row in self.transmat],
            "means": [list(row) for row in self.means],
            "covars": [[list(row) for row in matrix] for matrix in self.covars],
            "covariance_type": self.covariance_type,
            "labels": list(self.labels),
            "stats": [s.as_dict() for s in self.stats],
            "fitted_on": self.fitted_on,
            "fitted_on_proxy": self.fitted_on_proxy,
            "observations": self.observations,
            "converged": self.converged,
            "seed": self.seed,
            "fitted_through": self.fitted_through,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HmmFit:
        return cls(
            feature_names=tuple(raw["feature_names"]),
            startprob=tuple(float(p) for p in raw["startprob"]),
            transmat=tuple(tuple(float(v) for v in row) for row in raw["transmat"]),
            means=tuple(tuple(float(v) for v in row) for row in raw["means"]),
            covars=tuple(
                tuple(tuple(float(v) for v in row) for row in matrix) for matrix in raw["covars"]
            ),
            covariance_type=str(raw.get("covariance_type", DEFAULT_COVARIANCE)),
            labels=tuple(raw["labels"]),
            stats=tuple(StateStats.from_dict(s) for s in raw["stats"]),
            fitted_on=str(raw["fitted_on"]),
            fitted_on_proxy=bool(raw.get("fitted_on_proxy", False)),
            observations=int(raw["observations"]),
            converged=bool(raw.get("converged", True)),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            fitted_through=str(raw.get("fitted_through", "")),
        )

    def stats_for(self, label: str) -> StateStats | None:
        return next((s for s in self.stats if s.label == label), None)

    @property
    def degenerate_states(self) -> tuple[str, ...]:
        """Labels whose state holds too few observations to be a regime."""
        return tuple(s.label for s in self.stats if s.share < MIN_STATE_SHARE)


def _build(fit: HmmFit) -> GaussianHMM:
    """Reconstruct an hmmlearn model from stored parameters, fitting nothing."""
    model = GaussianHMM(
        n_components=len(fit.labels),
        covariance_type=fit.covariance_type,
        init_params="",
        params="",
    )
    model.startprob_ = np.asarray(fit.startprob, dtype=float)
    model.transmat_ = np.asarray(fit.transmat, dtype=float)
    model.means_ = np.asarray(fit.means, dtype=float)
    model.covars_ = np.asarray(fit.covars, dtype=float)
    return model


def _state_stats(
    states: np.ndarray,
    returns: pd.Series,
    periods_per_year: float,
    n_states: int,
) -> list[tuple[int, float, float, float, float]]:
    """(state, annualized mean return, annualized vol, share, persistence)."""
    values = returns.to_numpy(dtype=float)
    out: list[tuple[int, float, float, float, float]] = []

    # Mean run length: total observations in the state over the number of runs.
    boundaries = np.flatnonzero(np.diff(states)) + 1
    runs = np.split(states, boundaries)

    for state in range(n_states):
        mask = states == state
        share = float(mask.mean())
        selected = values[mask]
        selected = selected[np.isfinite(selected)]
        if selected.size < 2:
            out.append((state, 0.0, 0.0, share, 0.0))
            continue
        run_count = sum(1 for run in runs if run.size and run[0] == state)
        out.append(
            (
                state,
                float(selected.mean() * periods_per_year),
                float(selected.std(ddof=1) * np.sqrt(periods_per_year)),
                share,
                float(mask.sum() / run_count) if run_count else 0.0,
            )
        )
    return out


def reward_to_risk(mean_return: float, volatility: float) -> float:
    """Annualized return per unit of annualized volatility.

    No longer the labelling key — see the module docstring for why it was
    replaced — but kept because it is the natural way to *describe* a fitted
    state and the backtest report quotes it.
    """
    return mean_return / volatility if volatility > 1e-12 else -np.inf


def label_states(
    raw_stats: list[tuple[int, float, float, float, float]],
) -> dict[int, str]:
    """Apply the documented rule. Returns state -> label.

    Pure and separately testable, because this is the step that decides what
    every published regime *means* — and the one whose stability across refits
    §9 asks to be asserted.

    Sorted on ``(mean_return, -volatility)`` so the order is total. Two states
    with identical mean returns would otherwise be ordered by whichever the
    optimizer happened to number first, and a refit could swap their labels
    without a single number in the data having changed.
    """
    if len(raw_stats) != N_STATES:
        raise RegimeFitError(f"expected {N_STATES} states to label, got {len(raw_stats)}")

    ordered = sorted(raw_stats, key=lambda s: (s[1], -s[2]), reverse=True)
    return {entry[0]: label for entry, label in zip(ordered, REGIMES, strict=True)}


def transition_prior(
    n_states: int,
    periods_per_year: float,
    strength: float = DEFAULT_TRANSITION_PRIOR,
) -> np.ndarray:
    """Dirichlet pseudo-counts favouring self-transition, scaled to the frequency.

    ``strength`` is quoted for a daily series and scaled by the actual
    observation frequency, so "a regime lasts months" costs the monthly
    deep-history path proportionally fewer pseudo-counts than the daily paths. A
    prior fixed in absolute counts would be a mild nudge on 9,825 daily
    observations and an overwhelming one on 1,816 monthly ones.

    Off-diagonal entries are 1 — the uninformative Dirichlet — so the prior says
    only "states persist" and nothing at all about which transitions are likely.
    """
    scaled = strength * periods_per_year / PRIOR_REFERENCE_PERIODS
    return np.ones((n_states, n_states)) + np.eye(n_states) * max(scaled, 0.0)


def fit_hmm(
    design: RegimeDesign,
    *,
    n_states: int = N_STATES,
    seed: int = DEFAULT_SEED,
    n_iter: int = DEFAULT_N_ITER,
    covariance_type: str = DEFAULT_COVARIANCE,
    prior_strength: float = DEFAULT_TRANSITION_PRIOR,
    is_proxy: bool = False,
) -> HmmFit:
    """Fit and label a Gaussian HMM on one standardized design matrix.

    ``design`` must already be standardized per series (:mod:`.design`) — this
    function has no way to check that and every downstream guarantee depends
    on it.
    """
    if len(design) < n_states * 100:
        raise RegimeFitError(
            f"{design.series.series_id}: {len(design)} rows is too few to fit "
            f"{n_states} states; at least {n_states * 100} are needed for the "
            "state covariances to mean anything"
        )

    matrix = design.matrix
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=seed,
        transmat_prior=transition_prior(n_states, design.periods_per_year, prior_strength),
        # Reproducibility: k-means init from a fixed seed, so the same data
        # returns the same fit rather than the same fit up to a permutation.
        init_params="stmc",
    )

    with warnings.catch_warnings(), threadpool_limits(limits=1):
        # Not converging in n_iter is reported below with the iteration count
        # attached; hmmlearn's own warning says only that it happened.
        warnings.simplefilter("ignore")
        # One thread, for reproducibility rather than for thrift — see the module
        # docstring. Scoped to the fit so that nothing else in the process pays
        # for it, and covering `predict` too because the Viterbi path is what the
        # state labels are read off and it reads the same parameters.
        model.fit(matrix)
        states = model.predict(matrix)

    converged = bool(model.monitor_.converged)
    if not converged:
        log.warning(
            "hmm: %s did not converge in %d iterations; using the last parameters",
            design.series.series_id,
            n_iter,
        )

    raw_stats = _state_stats(states, design.returns, design.series.periods_per_year, n_states)
    labels = label_states(raw_stats)

    stats = tuple(
        StateStats(
            state=state,
            label=labels[state],
            mean_return=mean_return,
            volatility=vol,
            share=share,
            persistence=persistence,
        )
        for state, mean_return, vol, share, persistence in raw_stats
    )

    covars = np.asarray(model.covars_, dtype=float)
    fit = HmmFit(
        feature_names=tuple(design.frame.columns),
        startprob=tuple(float(p) for p in model.startprob_),
        transmat=tuple(tuple(float(v) for v in row) for row in model.transmat_),
        means=tuple(tuple(float(v) for v in row) for row in model.means_),
        covars=tuple(tuple(tuple(float(v) for v in row) for row in m) for m in covars),
        covariance_type=covariance_type,
        labels=tuple(labels[i] for i in range(n_states)),
        stats=stats,
        fitted_on=design.series.series_id,
        fitted_on_proxy=is_proxy,
        observations=len(design),
        converged=converged,
        seed=seed,
        fitted_through=design.frame.index[-1].date().isoformat(),
    )

    log.info(
        "hmm: fitted on %s (%d rows through %s)%s",
        fit.fitted_on,
        fit.observations,
        fit.fitted_through,
        " — PROXY, not the published index" if is_proxy else "",
    )
    for state in sorted(fit.stats, key=lambda s: REGIMES.index(s.label)):
        log.info(
            "  %-17s return=%+7.2f%%  vol=%5.2f%%  share=%5.1f%%  persistence=%.0f obs",
            state.label,
            state.mean_return * 100,
            state.volatility * 100,
            state.share * 100,
            state.persistence,
        )
    if fit.degenerate_states:
        log.warning(
            "hmm: %s hold under %.1f%% of observations — those are outliers, not regimes",
            ", ".join(fit.degenerate_states),
            MIN_STATE_SHARE * 100,
        )
    return fit


@dataclass(frozen=True)
class RegimeModel:
    """A fitted HMM applied to whatever design matrix it is handed.

    The transfer, made concrete: this object is fitted on ``calibration`` and
    called on ``publication``. It carries its own provenance so nothing
    downstream has to remember where its parameters came from.
    """

    fit: HmmFit

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RegimeModel:
        return cls(fit=HmmFit.from_dict(raw))

    def _check(self, design: RegimeDesign) -> None:
        columns = tuple(design.frame.columns)
        if columns != self.fit.feature_names:
            raise RegimeFitError(
                f"design columns {columns} do not match the fitted feature order "
                f"{self.fit.feature_names}; a Gaussian HMM's means are a vector and "
                "reordering them silently relabels every state"
            )

    def posteriors(self, design: RegimeDesign) -> pd.DataFrame:
        """P(regime | observations up to and including t), one column per regime.

        ``predict_proba`` in hmmlearn runs the forward-backward recursion, whose
        backward pass conditions on the whole sample — which is the smoothed
        answer, not the filtered one, and is banned from the feature path for
        exactly that reason. The forward pass alone is what a live state may use,
        so that is what this computes.
        """
        self._check(design)
        model = _build(self.fit)
        matrix = design.matrix

        # `_fit_log` gives the per-observation log likelihoods; `_do_forward_log_pass`
        # returns the forward lattice, from which alpha normalized per row is
        # P(state_t | y_1..y_t) — the filtered posterior.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_likelihood = model._compute_log_likelihood(matrix)
        forward = _forward_filtered(
            np.asarray(self.fit.startprob),
            np.asarray(self.fit.transmat),
            log_likelihood,
        )

        frame = pd.DataFrame(forward, index=design.frame.index, columns=list(self.fit.labels))
        # Columns come back in the published vocabulary order, so a chart stacks
        # them from most to least risk-on without knowing the state numbering.
        return frame[[label for label in REGIMES if label in frame.columns]]

    def states(self, design: RegimeDesign) -> pd.Series:
        """Most likely regime per date, by filtered posterior (argmax)."""
        posteriors = self.posteriors(design)
        return posteriors.idxmax(axis=1).rename("regime")

    def entropy(self, design: RegimeDesign) -> pd.Series:
        """Posterior entropy per date — the RII's uncertainty term (§3.2)."""
        posteriors = self.posteriors(design).to_numpy(dtype=float)
        safe = np.clip(posteriors, 1e-12, 1.0)
        values = -(safe * np.log(safe)).sum(axis=1)
        return pd.Series(values, index=self.posteriors(design).index, name="posterior_entropy")


def _forward_filtered(
    startprob: np.ndarray,
    transmat: np.ndarray,
    log_likelihood: np.ndarray,
) -> np.ndarray:
    """Filtered state posteriors from the forward recursion alone.

    Written out rather than taken from ``predict_proba`` because that method
    smooths: its backward pass conditions state *t* on observations after *t*,
    which would give the live dashboard a regime probability informed by the
    future. Normalizing each forward step in place also keeps the recursion
    numerically stable over ten thousand observations without logs.
    """
    n_obs, n_states = log_likelihood.shape
    out = np.empty((n_obs, n_states), dtype=float)

    # Subtract the row max before exponentiating: the log likelihoods run to a
    # few hundred in magnitude and exp() of that underflows to zero.
    scaled = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))

    alpha = startprob * scaled[0]
    total = alpha.sum()
    alpha = alpha / total if total > 0 else np.full(n_states, 1.0 / n_states)
    out[0] = alpha

    for t in range(1, n_obs):
        alpha = (alpha @ transmat) * scaled[t]
        total = alpha.sum()
        alpha = alpha / total if total > 0 else np.full(n_states, 1.0 / n_states)
        out[t] = alpha

    return out


__all__ = [
    "DEFAULT_COVARIANCE",
    "DEFAULT_N_ITER",
    "DEFAULT_SEED",
    "MIN_STATE_SHARE",
    "N_STATES",
    "HmmFit",
    "RegimeFitError",
    "RegimeModel",
    "StateStats",
    "fit_hmm",
    "label_states",
]
