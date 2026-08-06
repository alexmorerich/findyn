"""Shared vocabulary — names owned by no single engine.

These constants are fixed by FINDYN_V1_SPEC.md and mirrored in
serving/src/domain.ts. Changing one is a schema change, not a refactor;
tests/test_domain.py guards the two copies against drift.

Engine-private vocabulary does not belong here: the five HMM regimes and the
kinematic feature names live in :mod:`findynamics.engines.equity.domain`.
"""

from __future__ import annotations

from typing import Final

#: The five asset engines (``01-target-architecture.md`` §1). Registry keys,
#: `asset_state.asset` values, and the `ASSETS` array in serving/src/domain.ts.
ASSETS: Final[tuple[str, ...]] = ("money", "rates", "equity", "gold", "crypto")

#: §2.2 — the shared risk factors (v1 called them "forces"). Layer 0: computed
#: once per run and handed to every engine through ``WorldState``.
#: Re-exported by :mod:`findynamics.factors.definitions` as the initial factor set.
#:
#: Every score runs on one axis: 100 is maximally risk-supportive, 0 maximally
#: hostile. That is why ``real_rate`` scores high when real rates are *low* and
#: ``usd_strength`` scores high when the dollar is *weak* — the name says which
#: variable is being read, the axis says what the number means. Consumers that
#: want the level rather than the reading take it from ``components``.
FACTORS: Final[tuple[str, ...]] = (
    "valuation",
    "earnings",
    "liquidity",
    "rates",
    "credit",
    "inflation",
    "labor",
    "risk_appetite",
    "sentiment",
    # P1 additions (03-contracts.md §4): inputs the rates, gold and crypto
    # engines share, promoted to Layer 0 rather than recomputed per engine.
    "real_rate",
    "usd_strength",
    # P5. The *quantity* of money, which is not what `liquidity` measures.
    # `liquidity` blends M2 and the Fed's balance sheet with NFCI and overnight
    # RRP take-up, so it reads as financial conditions — tight or loose. This one
    # is the stock of central-bank and broad money on its own, which is the
    # variable FinCrypto regresses against, and mixing a conditions index into a
    # regressor whose coefficient is published as a beta would make that beta
    # uninterpretable.
    "global_liquidity",
)

#: Standard discount horizons — the tenors ``D(t, h)`` is published for.
#:
#: Distinct from :data:`HORIZONS`, which are *forecast* horizons (how far ahead a
#: model is asked to speak). These are points on a curve: what a dollar at that
#: tenor is worth today. FinMoney publishes a factor per entry; any engine that
#: discounts a cash flow uses the same grid, so two engines cannot disagree about
#: what "the 3y discount factor" means.
DISCOUNT_HORIZONS: Final[tuple[str, ...]] = (
    "1m",
    "3m",
    "6m",
    "1y",
    "2y",
    "3y",
    "5y",
    "7y",
    "10y",
    "20y",
    "30y",
)

#: Each discount horizon in years. ``1m`` is 1/12 of a year, not 30/360 — the
#: horizons are curve tenors, and a curve is parameterised in years.
DISCOUNT_HORIZON_YEARS: Final[dict[str, float]] = {
    "1m": 1.0 / 12.0,
    "3m": 0.25,
    "6m": 0.5,
    "1y": 1.0,
    "2y": 2.0,
    "3y": 3.0,
    "5y": 5.0,
    "7y": 7.0,
    "10y": 10.0,
    "20y": 20.0,
    "30y": 30.0,
}

#: Prefix marking a series that is another engine's *published output* read back
#: through the point-in-time gateway, e.g. ``ENGINE:rates.ns_level``.
#:
#: This is how an engine consumes another engine's work without importing it
#: (``01-target-architecture.md`` §3 rule 2, CI-enforced). The row travels the
#: same path as any observation — provider, ``macro_series`` shape, ``pit_join``
#: — so it is subject to the same release-date filter, and a consumer physically
#: cannot see an output that had not been published yet.
ENGINE_SERIES_PREFIX: Final[str] = "ENGINE:"


def engine_series_id(asset: str, metric: str) -> str:
    """Series id under which ``asset``'s ``metric`` is read back."""
    if not asset or not metric:
        raise ValueError(f"engine_series_id needs both asset and metric, got {asset!r}/{metric!r}")
    if asset not in ASSETS:
        raise ValueError(f"{asset!r} is not one of {list(ASSETS)}")
    return f"{ENGINE_SERIES_PREFIX}{asset}.{metric}"


def parse_engine_series_id(series_id: str) -> tuple[str, str] | None:
    """``('rates', 'ns_level')`` for an ``ENGINE:`` id, else ``None``.

    ``None`` rather than an exception: callers use this to *classify* an id, and
    "this is an ordinary provider series" is an answer, not a failure.
    """
    if not series_id.startswith(ENGINE_SERIES_PREFIX):
        return None
    body = series_id[len(ENGINE_SERIES_PREFIX) :]
    asset, _, metric = body.partition(".")
    if not asset or not metric:
        raise ValueError(
            f"malformed engine series id {series_id!r}; expected "
            f"{ENGINE_SERIES_PREFIX}<asset>.<metric>"
        )
    if asset not in ASSETS:
        raise ValueError(f"engine series id {series_id!r} names unknown asset {asset!r}")
    return asset, metric


#: §10 — forecast horizons.
HORIZONS: Final[tuple[str, ...]] = (
    "tactical",
    "strategic",
    "generational",
    "educational_30y",
    "educational_50y",
)

#: §10 — excluded from every accuracy evaluation; scenario simulation only.
EDUCATIONAL_HORIZONS: Final[frozenset[str]] = frozenset({"educational_30y", "educational_50y"})

#: §10 — quantiles stored for every forecast distribution.
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: §5.2 — observation cadences a configured series may declare.
FREQUENCIES: Final[frozenset[str]] = frozenset({"daily", "weekly", "monthly", "quarterly"})
