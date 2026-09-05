"""``K`` — a documented instability composite.

    K_t = w1·Z(a_t) + w2·Z(j_t) + w3·Z(Ω̇_t) + w4·Z(Ω̈_t) + w5·Z(C_t)

**This is not Riemann curvature.** There is no connection here and no second
derivative of a metric; ``curvature`` is a label on a composite that stands for
regime instability. The precedent is exact: ``FINDYN_V1_SPEC.md`` §3.1 replaced
*snap* — the fourth derivative of price — with the RII, because a fourth
derivative of a daily index is essentially all microstructure, and the honest
response was to build a composite of individually measurable things rather than
a derivative nobody can estimate.

``K`` inherits that obligation and one more: it has to be **checked against the
RII**. Both are equal-weighted composites of expanding-scored instability
measures over overlapping observables. If they correlate above 0.9, ``K`` is a
re-derivation of an index this repository already publishes, computed a more
expensive way, and the redundancy panel in :mod:`.diagnostics` is required to say
so in one sentence at the top. That sentence is a legitimate result.

Every ``Z`` is an expanding z-score
-----------------------------------

With the same minimum-baseline convention ``jerk_z`` is held to
(``features/kinematics.py::expanding_z`` and ``baseline_window``, the same
functions). A full-sample z here would put the whole future into every
historical curvature value, and it would be invisible: the series would look
entirely plausible and be slightly too good everywhere.

Every term enters as a magnitude, and here is why for each
----------------------------------------------------------

``rii.py`` states the rule for jerk — "jerk enters as a magnitude. Direction is
the market's business; instability is about the size of the change in trend,
either way" — and every term here takes it, but the argument is not identical
for all five and is worth writing down per term rather than asserted once:

``acceleration``
    The same argument as jerk, one derivative down. A trend that is turning
    hard is unstable whichever way it turns; the *direction* is what
    ``velocity`` is for and the engine already publishes it signed.
``jerk``
    ``rii.py``'s rule, verbatim. The engine's published form is already
    ``jerk_z``, an expanding z-score, so the term here is the expanding z-score
    of its magnitude — a second scoring that puts it on the same axis as the
    other four rather than leaving one term in different units.
``omega_velocity``
    Latent-state motion in either direction means the system is not settled.
    The rejected alternative — signed, so that ``K`` is high only when Ω is
    moving *toward* stress — was not taken because it turns an instability
    reading into a directional stress forecast, and because Ω already carries
    the level: a signed Ω̇ term would make ``K`` partly a second copy of Ω.
``omega_acceleration``
    As above, one derivative up.
``coupling``
    ``C`` is a sensitivity. A strongly negative coupling is as coupled a system
    as a strongly positive one, and the sign is genuinely interesting — it is
    published, with its own t-statistic, in the coupling panel. ``K`` asks how
    strongly the two blocks are tied together, which is a magnitude question.

Weights are equal by default and are never fitted on the full sample
--------------------------------------------------------------------

Config offers exactly two modes and there is no third:

``equal``
    ``w_i = 1/5``, the documented rule. Equal is a real choice and not a
    placeholder, for the reason ``rii.py::DEFAULT_WEIGHTS`` gives: nothing in
    the record justifies asserting that one term leads another by a factor of
    1.4, and a weight vector fitted on eight episodes is fitted on eight
    observations.
``fitted``
    The magnitudes of the first principal component of the five standardized
    terms, over the **first** ``fit_observations`` complete rows only, frozen
    for the entire path thereafter. Unsupervised on purpose: there is no target
    anywhere in this module, so there is no target to leak. The window is *the
    first N rows* rather than a fraction of the sample precisely so that
    appending later data cannot move it — which is the property the KK2 test
    asserts.

    The sign of a PC1 loading carries no interpretation on an all-magnitude
    block, so the weight is the loading's magnitude. A *negative* loading would
    mean one instability measure moves opposite to the other four, which is a
    finding for the diagnostics panel rather than a reason to subtract it from
    a composite — the raw loadings are therefore reported alongside.

Anything else is lookahead. Wanting to optimize the weights over the whole
history is the specific failure this rule exists to prevent.

Missing terms degrade with **renormalized** weights and the surviving list is
reported — ``compute_rii``'s pattern, "six components, with the seventh named".
The renormalization sums with ``math.fsum``: CPython 3.12 gave ``sum`` Neumaier
compensation and 3.11 sums naively, so the same five weights land on exactly 1.0
on one interpreter and 0.9999999999999999 on the other, which is a CI failure
rather than a rounding nicety (commit ``bdf5e2d``).

Per-date coverage, and where this departs from the RII
------------------------------------------------------

``compute_rii`` also renormalizes *per date* over the components that have a
value there, "so the warm-up of a slow component does not drag the whole index
toward zero while it fills". That is right for the RII, whose seven components
warm up within months of each other.

It is wrong here, and measurably so. The price terms come off a record that
starts in 1927 and the Ω terms off one that starts in 2000, so a per-date
renormalization with no floor publishes a **two-term** composite for the first
seventy years — and the largest reading in the whole series is then 1987-10-19,
built from ``acceleration`` and ``jerk`` alone. That is a price-instability
index wearing a curvature's name, and nothing in the output says so.

So ``min_coverage`` (config, default 0.999 — every term present) blanks any date
whose available weight falls short, and the per-date ``coverage`` series is
published so a reader can see exactly where and why.

The cost is real and belongs in the report rather than in a footnote: every term
is scored against a ten-year expanding baseline, so on a 2000-start Ω the
composite does not begin until roughly 2010 and **2008 is outside it**. Shorten
``zscore_min_years`` and it reaches further back on a noisier scale; that is a
KK5 sensitivity, not a default.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from findynamics.engines.equity.features.kinematics import baseline_window, expanding_z
from findynamics.factors.compute import score_series
from findynamics.research.omega.config import OmegaConfig
from findynamics.research.omega.estimator import DEFAULT_SEED

log = logging.getLogger("findynamics.research.omega.curvature")

#: The five terms, in the order the module docstring and every table list them.
CURVATURE_TERMS: tuple[str, ...] = (
    "acceleration",
    "jerk",
    "omega_velocity",
    "omega_acceleration",
    "coupling",
)

#: Weighting modes. There is no third, and adding one is a lookahead decision
#: rather than a configuration one.
WEIGHT_MODES: tuple[str, ...] = ("equal", "fitted")


class OmegaCurvatureError(ValueError):
    """Raised when the curvature composite is configured or called incoherently."""


@dataclass(frozen=True)
class CurvatureParams:
    """The curvature half of ``omega.yaml``."""

    weights: str = "equal"
    #: Complete rows the ``fitted`` mode estimates over, counted from the start.
    fit_observations: int = 1260
    zscore_min_years: float = 10.0
    #: A term with fewer scored rows than this is dropped and named.
    min_term_observations: int = 252
    #: Share of the total weight that must be present on a date for ``K`` to be
    #: published there. 0.999 means "every surviving term", written as a
    #: fraction rather than a flag so a deliberate partial composite is
    #: expressible and visible.
    min_coverage: float = 0.999

    @classmethod
    def from_config(cls, config: OmegaConfig) -> CurvatureParams:
        block = config.block("curvature")
        defaults = cls()

        mode = str(block.get("weights", defaults.weights))
        if mode not in WEIGHT_MODES:
            raise OmegaCurvatureError(
                f"curvature.weights must be one of {list(WEIGHT_MODES)}, got {mode!r}. "
                "There is deliberately no third mode: fitting weights on the whole "
                "history is lookahead (docs/research/kk-omega-design.md §6)"
            )

        fit_observations = int(block.get("fit_observations", defaults.fit_observations))
        if fit_observations < 2:
            raise OmegaCurvatureError(
                f"curvature.fit_observations must be >= 2, got {fit_observations}"
            )
        zscore_min_years = float(block.get("zscore_min_years", defaults.zscore_min_years))
        if zscore_min_years <= 0.0:
            raise OmegaCurvatureError(
                f"curvature.zscore_min_years must be > 0, got {zscore_min_years}"
            )

        min_coverage = float(block.get("min_coverage", defaults.min_coverage))
        if not 0.0 < min_coverage <= 1.0:
            raise OmegaCurvatureError(
                f"curvature.min_coverage must be within (0, 1], got {min_coverage}"
            )

        return cls(
            weights=mode,
            fit_observations=fit_observations,
            zscore_min_years=zscore_min_years,
            min_term_observations=int(
                block.get("min_term_observations", defaults.min_term_observations)
            ),
            min_coverage=min_coverage,
        )


@dataclass(frozen=True)
class CurvatureResult:
    """``K``, its parts, its weights, and what it could not see."""

    #: The composite. Same index as the scored terms.
    curvature: pd.Series
    #: Expanding percentile of the composite, 0-100.
    percentile: pd.Series
    #: Term name -> its expanding z-score. The inputs, after orientation.
    components: dict[str, pd.Series]
    #: Weights actually used, renormalized over the terms present. Sums to 1.
    weights: dict[str, float]
    #: Share of the total weight present on each date, before the coverage
    #: floor is applied. What makes a partial composite visible instead of
    #: implied.
    coverage: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: Terms the caller could not supply, or which were too short to score.
    missing: tuple[str, ...] = ()
    #: Set only in ``fitted`` mode: the window the weights were estimated on.
    fit_start: date | None = None
    fit_end: date | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def latest(self) -> float | None:
        usable = self.curvature.dropna()
        return None if usable.empty else float(usable.iloc[-1])

    def contributions(self, when: pd.Timestamp) -> dict[str, float]:
        """Each term's weighted contribution to ``K`` on one date.

        What the diagnostics panel decomposes the five largest ``K`` dates with.
        A composite whose biggest readings all come from one term is a composite
        with one component and four decorations, and this is how that becomes
        visible.
        """
        out: dict[str, float] = {}
        for name, series in self.components.items():
            if when in series.index and np.isfinite(series.loc[when]):
                out[name] = float(series.loc[when]) * self.weights.get(name, 0.0)
        return out


def _oriented(name: str, series: pd.Series) -> pd.Series:
    """Every term enters as a magnitude — see the module docstring, per term."""
    return series.abs()


def _fitted_weights(
    scored: pd.DataFrame,
    params: CurvatureParams,
) -> tuple[dict[str, float], date, date, dict[str, float]]:
    """PC1 magnitudes over the first ``fit_observations`` complete rows.

    Returns the weights, the window they were fitted on, and the raw loadings —
    the last so that a negative one, which would mean a term moving opposite to
    the others, is visible in the diagnostics rather than absorbed by the
    absolute value.
    """
    complete = scored.dropna()
    if len(complete) < params.fit_observations:
        raise OmegaCurvatureError(
            f"the fitted weighting needs {params.fit_observations} complete row(s) and has "
            f"{len(complete)}; either shorten curvature.fit_observations or use "
            "curvature.weights: equal"
        )
    window = complete.iloc[: params.fit_observations]

    mean = window.mean()
    scale = window.std(ddof=0)
    constant = sorted(c for c in window.columns if not scale[c] > 0.0)
    if constant:
        raise OmegaCurvatureError(
            f"term(s) {constant} are constant over the weighting window; a principal "
            "component cannot describe a column with no variation"
        )

    with threadpool_limits(limits=1):
        pca = PCA(n_components=1, svd_solver="full", random_state=DEFAULT_SEED)
        pca.fit(((window - mean) / scale).to_numpy(dtype=float))

    loadings = {
        name: float(value) for name, value in zip(window.columns, pca.components_[0], strict=True)
    }
    magnitudes = {name: abs(value) for name, value in loadings.items()}
    total = math.fsum(magnitudes.values())
    if total <= 0.0:
        raise OmegaCurvatureError("PC1 loads on nothing; the weighting window is degenerate")

    weights = {name: value / total for name, value in magnitudes.items()}
    negative = sorted(name for name, value in loadings.items() if value < 0.0)
    if negative:
        log.info(
            "omega curvature: PC1 loads negatively on %s over the weighting window — that "
            "term moves opposite to the others, which is a finding for the diagnostics "
            "panel and not a reason to subtract it from an instability composite",
            ", ".join(negative),
        )
    return weights, window.index[0].date(), window.index[-1].date(), loadings


def curvature_weights(
    scored: pd.DataFrame,
    *,
    params: CurvatureParams,
) -> tuple[dict[str, float], date | None, date | None, dict[str, float]]:
    """The weights actually used, renormalized with ``math.fsum`` to sum to 1."""
    if not len(scored.columns):
        raise OmegaCurvatureError("no terms to weight")

    if params.weights == "equal":
        share = 1.0 / len(scored.columns)
        weights = dict.fromkeys(scored.columns, share)
        # fsum rather than sum: the renormalization has to land on exactly 1.0
        # on 3.11 and on 3.13 alike (commit bdf5e2d).
        total = math.fsum(weights.values())
        return {name: value / total for name, value in weights.items()}, None, None, {}

    return _fitted_weights(scored, params)


def financial_curvature(
    terms: dict[str, pd.Series],
    *,
    params: CurvatureParams,
    periods_per_year: float,
) -> CurvatureResult:
    """``K`` and its parts, with weights renormalized over surviving terms.

    ``terms`` maps a name in :data:`CURVATURE_TERMS` to its raw series. A term
    the caller cannot supply is simply absent from the mapping; one that is
    present but too short to score is dropped here. Either way it is **named**
    in ``missing`` and never zero-filled — zero on an instability axis means
    "maximally stable", which is a much stronger claim than "we could not
    measure it".
    """
    unknown = sorted(set(terms) - set(CURVATURE_TERMS))
    if unknown:
        raise OmegaCurvatureError(
            f"unknown curvature term(s) {unknown}; expected a subset of {list(CURVATURE_TERMS)}"
        )

    scored: dict[str, pd.Series] = {}
    missing: list[str] = []
    baselines: dict[str, int] = {}

    for name in CURVATURE_TERMS:
        series = terms.get(name)
        if series is None or series.dropna().empty:
            missing.append(name)
            continue
        oriented = _oriented(name, series).dropna()
        if len(oriented) < params.min_term_observations:
            log.info(
                "omega curvature: dropping %s — %d scored row(s), below the %d required",
                name,
                len(oriented),
                params.min_term_observations,
            )
            missing.append(name)
            continue
        baseline, _ = baseline_window(
            len(oriented), periods_per_year, min_years=params.zscore_min_years
        )
        baselines[name] = baseline
        scored[name] = expanding_z(oriented, baseline).rename(name)

    if not scored:
        raise OmegaCurvatureError(
            "the curvature composite needs at least one usable term and has none"
        )

    frame = pd.DataFrame(scored)
    frame = frame[[name for name in CURVATURE_TERMS if name in frame.columns]]

    weights, fit_start, fit_end, loadings = curvature_weights(frame, params=params)

    # Renormalized per date over the terms that actually have a value there, so
    # a slow term's warm-up does not drag the composite toward zero while it
    # fills — `compute_rii`'s treatment, for the same reason.
    weight_row = pd.Series(weights)
    present = frame.notna()
    weighted = (frame.fillna(0.0) * weight_row).sum(axis=1)
    coverage = (present * weight_row).sum(axis=1).rename("coverage")
    curvature = (weighted / coverage.replace(0.0, np.nan)).rename("financial_curvature")
    # The coverage floor. Without it the composite publishes a two-term reading
    # for every pre-Ω date and calls it curvature — see the module docstring.
    curvature = curvature.where(coverage >= params.min_coverage)

    percentile = score_series(curvature.dropna(), 1).reindex(curvature.index)
    percentile = percentile.rename("financial_curvature_percentile")

    diagnostics: dict[str, float] = {
        "curvature_terms_used": float(len(scored)),
        "curvature_terms_missing": float(len(missing)),
        "curvature_defined": float(curvature.notna().sum()),
        "curvature_weight_sum": math.fsum(weights.values()),
        "curvature_min_coverage": params.min_coverage,
        "curvature_blanked_by_coverage": float(
            int((coverage > 0.0).sum()) - int(curvature.notna().sum())
        ),
    }
    diagnostics.update({f"curvature_weight_{name}": value for name, value in weights.items()})
    diagnostics.update({f"curvature_baseline_{name}": float(v) for name, v in baselines.items()})
    diagnostics.update({f"curvature_loading_{name}": value for name, value in loadings.items()})

    if missing:
        log.info(
            "omega curvature: built from %d of %d terms; missing %s",
            len(scored),
            len(CURVATURE_TERMS),
            ", ".join(sorted(missing)),
        )

    defined = curvature.dropna()
    log.info(
        "omega curvature: %d of %d date(s) reach the %.1f%% coverage floor%s",
        len(defined),
        int((coverage > 0.0).sum()),
        params.min_coverage * 100.0,
        f"; K spans {defined.index[0].date()} → {defined.index[-1].date()}"
        if not defined.empty
        else "",
    )

    return CurvatureResult(
        curvature=curvature,
        percentile=percentile,
        components=scored,
        weights=weights,
        coverage=coverage,
        missing=tuple(sorted(missing)),
        fit_start=fit_start,
        fit_end=fit_end,
        diagnostics=diagnostics,
    )


__all__ = [
    "CURVATURE_TERMS",
    "WEIGHT_MODES",
    "CurvatureParams",
    "CurvatureResult",
    "OmegaCurvatureError",
    "curvature_weights",
    "financial_curvature",
]
