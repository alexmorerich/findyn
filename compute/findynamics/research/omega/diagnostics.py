"""The markdown panels the KK5 report is assembled from.

A ``render_report``-style renderer following ``engines/equity/diagnostics.py``:
it takes the objects the module produces and returns markdown, so that the
report is regenerable from stored artifacts rather than hand-assembled from a
terminal.

Five panels, and one of them is the point of the whole track.

**Ω panel.** Explained variance, the loading table, the columns dropped, the
sign-reference column. A loading table is the only way to see that "Ω" meant a
different linear combination in 2008 than it did in 2020.

**Coupling panel.** For each of the three estimators: defined share, the ``β1``
path with its Newey-West t-statistics, ``R²``, the winsorized share, and whether
it declined.

**Curvature panel.** The component correlation matrix, the weight table, and the
contribution decomposition on the five largest ``K`` dates. A composite whose
biggest readings all come from one term is a composite with one component and
four decorations, and the decomposition is how that becomes visible.

**The redundancy panel — required, and the reason this module exists.**
Correlation and rank correlation of Ω, Ω̇, ``C`` and ``K`` against **each
existing published quantity**: ``velocity``, ``acceleration``, ``jerk_z``, the
RII, and each RII component from ``rii.py``. If ``K`` correlates above
:data:`REDUNDANT_RHO` with the RII, the panel opens with the sentence *"K is a
re-derivation of the existing instability index."* That sentence is emitted by a
threshold rather than by a judgement call, precisely so that nobody has to
decide on the day whether to write it, and the renderer must not soften it.

**Stationarity panel.** An augmented Dickey-Fuller reading and a variance ratio
on Ω and ``C``, so the KK3 regressions are not run on something with an obvious
unit root without the report saying so. This is not a formality: the
``velocity`` specification's in-sample ``R²`` is around 0.45, which on two
persistent regressors is exactly the shape a spurious regression takes.

Building the comparison set
---------------------------

:func:`equity_reference` runs the engine's own stack — ``compute_features`` →
``build_design`` → ``fit_hmm`` → ``compute_rii`` — on the publication path, so
the quantities Ω is compared against are the ones the engine actually publishes
rather than re-derivations of them. It costs about ten seconds and is the
expensive part of any diagnostics run.

It uses **one** price path where the engine uses two (it computes the RII on the
calibration path and reindexes onto publication). On the shipped configuration
those are the same index over nearly the same span, and this panel reports
*correlations* rather than reproducing published values, so the simplification
is safe — but it does mean a number here will not match a number on the
dashboard to the last decimal, and the report says so rather than implying
otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.pit import PITAccessor
from findynamics.engines.equity import rii as rii_mod
from findynamics.engines.equity.features.pipeline import FeatureSet, compute_features
from findynamics.engines.equity.prices import publication_path, resolve_from
from findynamics.engines.equity.regime.design import RegimeDesign, build_design
from findynamics.engines.equity.regime.hmm import RegimeModel, fit_hmm
from findynamics.research.omega.contracts import OmegaPath
from findynamics.research.omega.coupling import (
    ESTIMATOR_NAMES,
    REGRESSION_SPECIFICATIONS,
    CouplingResult,
)
from findynamics.research.omega.curvature import CurvatureResult

log = logging.getLogger("findynamics.research.omega.diagnostics")

#: ADF p-value above which the panel reports that a unit root cannot be ruled
#: out. The conventional 5%: a threshold that moves after the data arrives is
#: not a threshold, and this one is quoted in the KK5 report.
UNIT_ROOT_P = 0.05

#: Rank correlation at or above which one series is reported as a re-derivation
#: of another. 0.9 is the design note's number (§8, Q4) and is deliberately not
#: a tunable: a threshold that moves after the data arrives is not a threshold.
REDUNDANT_RHO = 0.9

#: Minimum overlapping observations before a correlation is printed at all. A
#: rank correlation over eighty dates is a number, not evidence.
MIN_OVERLAP = 250

#: Series ids the RII needs beyond the price path.
RII_SERIES: tuple[str, ...] = ("FRED:DGS10", "FRED:BAMLH0A0HYM2", "FRED:NFCI")


@dataclass(frozen=True)
class EquityReference:
    """What the equity engine publishes, for Ω to be compared against."""

    features: FeatureSet
    design: RegimeDesign
    rii: rii_mod.RiiResult
    posteriors: pd.DataFrame

    def published(self) -> dict[str, pd.Series]:
        """Name -> series, in the order the redundancy panel prints them."""
        out: dict[str, pd.Series] = {
            "velocity": self.features.frame["velocity"],
            "acceleration": self.features.frame["acceleration"],
            "jerk_z": self.features.frame["jerk_z"],
            "rii": self.rii.index,
        }
        out.update({f"rii_{name}": series for name, series in self.rii.components.items()})
        return out


def equity_reference(
    accessor: PITAccessor,
    config: SeriesConfig | None = None,
) -> EquityReference:
    """Run the engine's own stack so the comparison is against real outputs.

    Expensive — a Kalman MLE, an FFD search and an HMM fit, about ten seconds on
    the committed fixture — and deliberately not cached here: a cache keyed on
    nothing is how two panels in one report come to describe two different
    information sets.
    """
    resolved = config or get_series_config()
    roles = resolve_from(accessor, resolved)
    path, series = publication_path(accessor, roles)
    features = compute_features(path, series)
    design = build_design(features)
    model = RegimeModel(fit_hmm(design))
    posteriors = model.posteriors(design)

    wide = accessor.wide(list(RII_SERIES))
    rii = rii_mod.compute_rii(
        posteriors,
        jerk_z=features.frame.get("jerk_z"),
        realized_vol=design.realized_vol,
        equity_returns=features.log_price.diff(),
        bond_yield=wide.get("FRED:DGS10"),
        credit_spread=wide.get("FRED:BAMLH0A0HYM2"),
        liquidity=wide.get("FRED:NFCI"),
        periods_per_year=series.periods_per_year,
    )
    log.info(
        "omega diagnostics: equity reference built on %s over %d row(s); "
        "RII from %d component(s), missing %s",
        series.series_id,
        len(features.frame),
        len(rii.components),
        ", ".join(rii.missing) or "nothing",
    )
    return EquityReference(features=features, design=design, rii=rii, posteriors=posteriors)


def _pair(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    """The two series on their common dates.

    ``sort=True`` explicitly. The two rarely share a calendar exactly, pandas is
    deprecating the implicit sort, and an unsorted union would put a rank
    correlation out of date order — the same call ``rii.correlation_breakdown``
    makes, for the same reason.
    """
    return pd.concat([left.rename("left"), right.rename("right")], axis=1, sort=True).dropna()


def correlations(
    latent: dict[str, pd.Series],
    published: dict[str, pd.Series],
) -> pd.DataFrame:
    """Rank correlation of every latent quantity against every published one.

    Spearman rather than Pearson as the headline: the question is whether one
    series is a *re-derivation* of another, and a monotone reparameterization is
    still a re-derivation. Pearson is reported beside it because a high Spearman
    with a low Pearson is itself informative — it says the two agree on the
    ordering and disagree on the scale.
    """
    rows: list[dict[str, object]] = []
    for published_name, published_series in published.items():
        row: dict[str, object] = {"published": published_name}
        for latent_name, latent_series in latent.items():
            pair = _pair(latent_series, published_series)
            if len(pair) < MIN_OVERLAP:
                row[f"{latent_name}_rho"] = np.nan
                row[f"{latent_name}_r"] = np.nan
                row[f"{latent_name}_n"] = float(len(pair))
                continue
            row[f"{latent_name}_rho"] = float(pair["left"].corr(pair["right"], method="spearman"))
            row[f"{latent_name}_r"] = float(pair["left"].corr(pair["right"]))
            row[f"{latent_name}_n"] = float(len(pair))
        rows.append(row)
    return pd.DataFrame(rows).set_index("published")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    if isinstance(value, int | float):
        return f"{value:+.{digits}f}"
    return str(value)


def _span(series: pd.Series) -> str:
    usable = series.dropna()
    if usable.empty:
        return "empty"
    return f"{usable.index[0].date()} → {usable.index[-1].date()} ({len(usable):,} rows)"


def omega_panel(path: OmegaPath) -> list[str]:
    spec = path.spec
    lines = [
        "## Ω",
        "",
        f"- Estimator: `{spec.estimator}`, model version `{spec.model_version}`",
        f"- Fitted on {spec.n_observations:,} rows, {spec.fit_start} → {spec.fit_end}",
        f"- PC1 explains **{spec.explained_variance_ratio:.1%}** of the standardized block",
        f"- Sign pinned on `{spec.sign_reference}`",
        f"- Columns dropped: {', '.join(f'`{c}`' for c in spec.dropped) or 'none'}",
        "",
        "| Column | Loading | Scaler mean | Scaler scale |",
        "|---|---:|---:|---:|",
    ]
    for column in spec.columns:
        index = spec.columns.index(column)
        lines.append(
            f"| `{column}` | {_fmt(spec.loading(column))} | "
            f"{_fmt(spec.mean[index], 4)} | {_fmt(spec.scale[index], 4)} |"
        )
    scored = int(path.omega_regime.notna().sum())
    lines += ["", "| Latent state | Share of scored dates |", "|---|---:|"]
    for label in ("latent_calm", "latent_transition", "latent_stress"):
        share = float((path.omega_regime == label).sum()) / scored if scored else 0.0
        lines.append(f"| `{label}` | {share:.1%} |")
    lines += [
        "",
        f"Ω spans {_span(path.omega)}; Ω̇ spans {_span(path.omega_velocity)} after "
        f"{int(path.diagnostics.get('burn_in_periods', 0)):,} rows of filter start-up.",
        "",
    ]
    return lines


def coupling_panel(results: dict[str, CouplingResult]) -> list[str]:
    lines = ["## Coupling `C`", ""]
    for name in ESTIMATOR_NAMES:
        result = results.get(name)
        if result is None:
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        if result.declined:
            lines += [f"**Declined.** {result.decline_reason}", ""]
            continue
        lines += [
            f"- Span: {_span(result.coupling)}",
            f"- Defined on {result.defined_share:.1%} of its dates",
        ]
        for key in ("coupling_undefined_share", "coupling_winsorized_share"):
            if key in result.diagnostics:
                lines.append(f"- `{key}`: {result.diagnostics[key]:.1%}")
        latest = result.latest()
        lines.append(f"- Latest: {_fmt(latest, 6) if latest is not None else 'n/a'}")
        lines.append("")

        if name == "regression":
            lines += [
                "| Specification | β1 on Ω̇ | Newey–West t | R² | Dates |",
                "|---|---:|---:|---:|---:|",
            ]
            for specification in REGRESSION_SPECIFICATIONS:
                diagnostics = result.diagnostics
                lines.append(
                    f"| `{specification}` "
                    f"| {_fmt(diagnostics.get(f'regression_{specification}_beta_latest'), 6)} "
                    f"| {_fmt(diagnostics.get(f'regression_{specification}_tstat_latest'), 2)} "
                    f"| {_fmt(diagnostics.get(f'regression_{specification}_r2_latest'))} "
                    f"| {int(diagnostics.get(f'regression_{specification}_rows', 0)):,} |"
                )
            lines += [
                "",
                "The t-statistic is Newey–West and is evaluated on a cadence, not daily; "
                "an exact daily path is quadratic in the sample length because the "
                "residuals are refitted at every date.",
                "",
                "A high in-sample `R²` here is **not** evidence. Both regressors are "
                "persistent, which is the shape a spurious regression takes — see the "
                "stationarity panel before reading it as anything.",
                "",
            ]
    return lines


def curvature_panel(result: CurvatureResult) -> list[str]:
    lines = [
        "## Curvature `K`",
        "",
        f"- Span: {_span(result.curvature)}",
        f"- Terms used: {', '.join(f'`{t}`' for t in result.components)}",
        f"- Terms missing: {', '.join(f'`{t}`' for t in result.missing) or 'none'}",
        f"- Dates blanked by the coverage floor: "
        f"{int(result.diagnostics.get('curvature_blanked_by_coverage', 0)):,}",
    ]
    if result.fit_start is not None:
        lines.append(f"- Weights fitted on {result.fit_start} → {result.fit_end}, then frozen")
    lines += ["", "| Term | Weight | PC1 loading |", "|---|---:|---:|"]
    for term, weight in result.weights.items():
        loading = result.diagnostics.get(f"curvature_loading_{term}")
        lines.append(f"| `{term}` | {weight:.4f} | {_fmt(loading) if loading else '—'} |")
    lines.append("")

    frame = pd.DataFrame(result.components).dropna()
    if len(frame) >= MIN_OVERLAP:
        correlation = frame.corr(method="spearman")
        lines += [
            "Component rank correlations:",
            "",
            "| | " + " | ".join(f"`{c}`" for c in correlation.columns) + " |",
            "|---" * (len(correlation.columns) + 1) + "|",
        ]
        for term in correlation.index:
            cells = " | ".join(_fmt(correlation.loc[term, c], 2) for c in correlation.columns)
            lines.append(f"| `{term}` | {cells} |")
        lines.append("")

    largest = result.curvature.dropna().nlargest(5)
    if not largest.empty:
        terms = list(result.components)
        lines += [
            "Contribution decomposition on the five largest `K` dates — a composite "
            "whose biggest readings all come from one term is a composite with one "
            "component and four decorations:",
            "",
            "| Date | K | " + " | ".join(f"`{t}`" for t in terms) + " |",
            "|---" * (len(terms) + 2) + "|",
        ]
        for when, value in largest.items():
            contributions = result.contributions(when)
            cells = " | ".join(_fmt(contributions.get(t), 2) for t in terms)
            lines.append(f"| {when.date()} | {value:+.2f} | {cells} |")
        lines.append("")
    return lines


def redundancy_panel(
    latent: dict[str, pd.Series],
    published: dict[str, pd.Series],
) -> list[str]:
    """Ω, Ω̇, `C`, `K` against every quantity the engine already publishes.

    The verdict sentence at the top is emitted by :data:`REDUNDANT_RHO` rather
    than by a judgement call. That is the whole design: the person running this
    on the day the numbers arrive does not get to decide whether to write it.
    """
    table = correlations(latent, published)
    lines = ["## Redundancy against what the engine already publishes", ""]

    verdicts: list[str] = []
    for latent_name in latent:
        column = f"{latent_name}_rho"
        if column not in table.columns:
            continue
        magnitudes = table[column].abs().dropna()
        if magnitudes.empty:
            continue
        worst = magnitudes.idxmax()
        value = table.loc[worst, column]
        if abs(value) >= REDUNDANT_RHO:
            if latent_name == "curvature" and worst == "rii":
                verdicts.append("**K is a re-derivation of the existing instability index.**")
            else:
                verdicts.append(
                    f"**`{latent_name}` is a re-derivation of `{worst}`** (ρ_s = {value:+.2f})."
                )
    if verdicts:
        lines += [*verdicts, ""]
    else:
        lines += [
            f"No latent quantity reaches |ρ_s| ≥ {REDUNDANT_RHO} against any published "
            "quantity, so none of them is a re-derivation of something the engine already "
            "ships. That is a statement about redundancy and **not** about usefulness: "
            "carrying different information and carrying *useful* information are "
            "different claims, and only KK3 can test the second.",
            "",
        ]

    header = "| Published | " + " | ".join(f"`{name}` ρ_s / r" for name in latent) + " |"
    lines += [header, "|---" * (len(latent) + 1) + "|"]
    for published_name in table.index:
        cells = []
        for latent_name in latent:
            rho = table.loc[published_name, f"{latent_name}_rho"]
            pearson = table.loc[published_name, f"{latent_name}_r"]
            cells.append(f"{_fmt(rho, 2)} / {_fmt(pearson, 2)}")
        lines.append(f"| `{published_name}` | " + " | ".join(cells) + " |")

    overlaps = [int(table[f"{name}_n"].min()) for name in latent if f"{name}_n" in table.columns]
    lines += [
        "",
        f"Smallest overlap in the table: {min(overlaps):,} dates." if overlaps else "",
        "",
    ]
    return [line for line in lines if line is not None]


def _adf(series: pd.Series, name: str) -> dict[str, float]:
    """Augmented Dickey-Fuller with a fixed lag, so two runs agree.

    ``autolag`` selects the lag from the data, which is deterministic but makes
    the reported statistic depend on a choice nothing else in the report can
    see. The Schwert rule gives a lag from the sample size alone, which is the
    same kind of pure function as the Newey-West rule beside it.
    """
    clean = series.dropna()
    if len(clean) < MIN_OVERLAP:
        return {}
    lag = int(np.floor(12.0 * (len(clean) / 100.0) ** 0.25))
    statistic, pvalue, *_ = adfuller(clean.to_numpy(dtype=float), maxlag=lag, autolag=None)
    return {"statistic": float(statistic), "p_value": float(pvalue), "lag": float(lag)}


def _variance_ratio(series: pd.Series, horizon: int = 21) -> float | None:
    """``Var(x_t − x_{t−h}) / (h · Var(x_t − x_{t−1}))``.

    1 for a random walk, below 1 for mean reversion, above for trending. A
    second reading beside the ADF because the two fail in different directions,
    and a unit root that only one of them sees is worth knowing about.
    """
    clean = series.dropna()
    if len(clean) < MIN_OVERLAP:
        return None
    short = clean.diff().dropna()
    long = clean.diff(horizon).dropna()
    if short.var() <= 0.0:
        return None
    return float(long.var() / (horizon * short.var()))


def stationarity_panel(series: dict[str, pd.Series]) -> list[str]:
    rows: list[str] = []
    suspect: list[tuple[str, float]] = []
    for name, values in series.items():
        adf = _adf(values, name)
        ratio = _variance_ratio(values)
        if not adf:
            rows.append(f"| `{name}` | n/a | n/a | n/a | n/a |")
            continue
        if adf["p_value"] > UNIT_ROOT_P:
            suspect.append((name, adf["p_value"]))
        rows.append(
            f"| `{name}` | {adf['statistic']:+.2f} | {adf['p_value']:.4f} | "
            f"{int(adf['lag'])} | {_fmt(ratio)} |"
        )

    lines = [
        "## Stationarity",
        "",
        "The KK3 regressions are run on these. A unit root in a regressor and a "
        "persistent target is how a regression reports a large `R²` and means nothing "
        "by it.",
        "",
    ]
    # Emitted by the threshold, like the redundancy verdict above it, so that
    # nobody has to decide on the day whether the caveat is worth writing.
    if suspect:
        named = ", ".join(f"`{name}` (p = {p:.2f})" for name, p in suspect)
        lines += [
            f"**A unit root cannot be ruled out in {named}.** Any KK3 arm that uses one "
            "of these as a regressor has to say what it did about that — first-difference "
            "it, or report the result as conditional on the level being stationary. An "
            "expanding-window coefficient path is autocorrelated by construction, so this "
            "is an expected reading rather than a surprising one, and it is exactly the "
            "kind of expected reading that gets skipped over.",
            "",
        ]
    else:
        lines += [
            f"Every series rejects a unit root at p < {UNIT_ROOT_P}.",
            "",
        ]

    lines += [
        "| Series | ADF statistic | p | lag | Variance ratio (21d) |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    return lines


@dataclass(frozen=True)
class DiagnosticsInput:
    """Everything ``render_report`` needs, gathered by the caller.

    A record rather than a long signature: KK5 regenerates this report from
    stored artifacts and a positional call with eight arguments is how the
    coupling results and the curvature result end up swapped.
    """

    path: OmegaPath
    couplings: dict[str, CouplingResult]
    curvature: CurvatureResult
    reference: EquityReference | None = None
    notes: list[str] = field(default_factory=list)


def render_report(data: DiagnosticsInput) -> str:
    """The full markdown body, assembled from the objects the module produced."""
    lines = [
        "# KK-Ω diagnostics",
        "",
        "A fifth-dimensional latent-state representation inspired by Kaluza–Klein "
        "geometry, tested as a quantitative modelling hypothesis.",
        "",
        "**Everything below is in-sample.** Each quantity here was computed from a "
        "projection fitted on the whole record, so this document describes what the "
        "construction looks like and cannot say whether any of it predicts anything. "
        "KK3's walk-forward is the only thing that may produce a reported number.",
        "",
    ]
    lines += data.notes + ([""] if data.notes else [])
    lines += omega_panel(data.path)
    lines += coupling_panel(data.couplings)
    lines += curvature_panel(data.curvature)

    latent = {
        "omega": data.path.omega,
        "omega_velocity": data.path.omega_velocity,
        "coupling": data.couplings[
            "regression" if "regression" in data.couplings else next(iter(data.couplings))
        ].coupling,
        "curvature": data.curvature.curvature,
    }
    if data.reference is not None:
        lines += redundancy_panel(latent, data.reference.published())
    lines += stationarity_panel(
        {
            "omega": data.path.omega,
            "omega_velocity": data.path.omega_velocity,
            "coupling": latent["coupling"],
            "curvature": data.curvature.curvature,
        }
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "MIN_OVERLAP",
    "REDUNDANT_RHO",
    "UNIT_ROOT_P",
    "RII_SERIES",
    "DiagnosticsInput",
    "EquityReference",
    "correlations",
    "coupling_panel",
    "curvature_panel",
    "equity_reference",
    "omega_panel",
    "redundancy_panel",
    "render_report",
    "stationarity_panel",
]
