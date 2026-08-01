"""Series that are arithmetic over other series (``DERIVED:*``).

Two of the v1 factor inputs are not fetched from anywhere — they are computed
from series that already exist:

``DERIVED:EXCESS_CAPE_YIELD``
    ``1 / CAPE − (nominal 10y − 10y breakeven)``. Shiller's own excess CAPE
    yield: what equities earn over the real risk-free rate, which is the
    comparison a raw CAPE cannot make. A CAPE of 30 means something very
    different at a 0% real rate than at 3%.

``DERIVED:HY_IG_DIFFERENTIAL``
    ``HY OAS − IG OAS``. The *credit-quality* spread rather than the credit
    spread: both widen when rates move, and the difference isolates how much
    extra compensation the market demands for default risk specifically.

**The release date is the maximum over the inputs, never a synthesized lag.**
This is the whole reason the module exists rather than the arithmetic living in
a factor. ``series.yaml`` gives ``DERIVED:EXCESS_CAPE_YIELD`` a 30-day lag, and
synthesizing a release date from that would claim the January value became
knowable on 31 January — but it depends on CAPE, which Shiller publishes weeks
later. Every date in between would be a lookahead the PIT machinery could not
see, because the row would carry a release date that *looked* legitimate.
Composing point-in-time series means composing their release dates, and the
composition is a maximum: a derived value is knowable only once its slowest
input is.

Inputs are read back through the serving plane rather than re-fetched from FRED,
for the same reason :mod:`.published` does it: whatever the rest of the system
is using for CAPE is what this must use, or the derived series and its own
inputs disagree about history. That read is also where the real release dates
come from — a fresh FRED fetch would only give today's vintage.

Two questions get answered separately, and conflating them is the mistake this
docstring exists to prevent:

*Which value pairs with which?* — by **observation date**. January's CAPE belongs
beside January's real rate, because the excess CAPE yield is a statement about
one period. Pairing on release dates instead would put mid-February's rate next
to January's valuation, since that is what had been published by the time CAPE
landed. Economically wrong, and invisible in the output.

*When does the pair become knowable?* — the **maximum release date** over the
observations actually used. That is the composition rule above.

So mixed frequencies compose by as-of join on the observation date, and the
release date is computed afterwards from whatever that join selected. Neither
step can be skipped: the first alone reaches forward past the release, the
second alone mismatches the periods.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from urllib.parse import quote

from findynamics.data.providers.base import (
    Frequency,
    NotFoundError,
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.published import (
    ADMIN_URL_ENV,
    API_URL_ENV,
    CACHE_TTL,
    resolve_api_base,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.derived")

#: Newest N observations per input series. The derived series can be no longer
#: than its shortest input, and every one of these is a factor input already
#: capped at a comparable window.
DEFAULT_LIMIT = 5000

PREFIX = "DERIVED:"


@dataclass(frozen=True)
class Recipe:
    """One derived series: its inputs, how to combine them, and what it means."""

    series_id: str
    frequency: Frequency
    unit: str
    title: str
    notes: str
    #: Input series ids, in the order ``combine`` expects them. The first is the
    #: **driver**: its observation dates become the derived series' dates, and
    #: the others are joined as-of onto it.
    inputs: tuple[str, ...]
    combine: object  # Callable[[tuple[float, ...]], float]; typed loosely for the table


def _excess_cape_yield(values: tuple[float, ...]) -> float:
    cape, nominal_10y, breakeven_10y = values
    if cape <= 0:
        raise ValueError(f"CAPE must be positive to invert, got {cape}")
    # Percent throughout: 1/CAPE is a fraction, the rate legs are already in
    # percent. Getting this wrong scales the factor by 100 and it still looks
    # like a plausible time series, which is why it is stated rather than
    # inferred from the arithmetic.
    return (1.0 / cape) * 100.0 - (nominal_10y - breakeven_10y)


def _hy_ig_differential(values: tuple[float, ...]) -> float:
    high_yield, investment_grade = values
    return high_yield - investment_grade


#: Every derived series this provider knows how to build. Adding one is a table
#: entry, not a code path.
RECIPES: dict[str, Recipe] = {
    "DERIVED:EXCESS_CAPE_YIELD": Recipe(
        series_id="DERIVED:EXCESS_CAPE_YIELD",
        frequency="monthly",
        unit="percent",
        title="Excess CAPE yield (1/CAPE minus the 10y real rate)",
        notes=(
            "1/SHILLER:CAPE as a percent, minus (FRED:DGS10 - FRED:T10YIE). "
            "Released when the last of its inputs was released, which is "
            "governed by CAPE."
        ),
        inputs=("SHILLER:CAPE", "FRED:DGS10", "FRED:T10YIE"),
        combine=_excess_cape_yield,
    ),
    "DERIVED:HY_IG_DIFFERENTIAL": Recipe(
        series_id="DERIVED:HY_IG_DIFFERENTIAL",
        frequency="daily",
        unit="percent",
        title="High-yield minus investment-grade OAS",
        notes=(
            "FRED:BAMLH0A0HYM2 - FRED:BAMLC0A0CM. Isolates compensation for "
            "default risk from the rate move both spreads share."
        ),
        inputs=("FRED:BAMLH0A0HYM2", "FRED:BAMLC0A0CM"),
        combine=_hy_ig_differential,
    ),
}


@dataclass(frozen=True)
class _Point:
    """One input observation, reduced to what the composition needs."""

    observation_date: date
    release_date: date
    value: float


class DerivedProvider(Provider):
    """Computes ``DERIVED:*`` series from observations already in the store."""

    id = "derived"
    requires_api_key = False

    def __init__(
        self,
        transport: Transport,
        *,
        base_url: str | None,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/") if base_url else None
        self.limit = limit

    def available_series(self) -> list[str]:
        return sorted(RECIPES)

    def _recipe(self, series_id: str) -> Recipe:
        recipe = RECIPES.get(series_id)
        if recipe is None:
            raise NotFoundError(
                self.id,
                f"unknown derived series {series_id!r}; available: {', '.join(sorted(RECIPES))}",
            )
        return recipe

    def _require_base(self) -> str:
        if not self.base_url:
            raise NotFoundError(
                self.id,
                f"neither {API_URL_ENV} nor {ADMIN_URL_ENV} is set, so there is no "
                "serving plane to read this series' inputs from",
            )
        return self.base_url

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        recipe = self._recipe(series_id)
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=recipe.title,
            frequency=recipe.frequency,
            unit=recipe.unit,
            notes=recipe.notes,
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        recipe = self._recipe(series_id)
        base = self._require_base()

        loaded = {name: self._load_input(base, name, start, end) for name in recipe.inputs}

        missing = [name for name, points in loaded.items() if not points]
        if missing:
            # Loud, and empty rather than partial. A derived series computed from
            # some of its inputs is not a shorter version of itself, it is a
            # different quantity — an excess CAPE yield without the rate legs is
            # just an earnings yield, and it would publish under a name that
            # promises otherwise.
            raise NotFoundError(
                self.id,
                f"{series_id}: no observations for {', '.join(missing)}; "
                "backfill the inputs before deriving from them",
            )

        driver = loaded[recipe.inputs[0]]
        others = [loaded[name] for name in recipe.inputs[1:]]

        observations: list[Observation] = []
        skipped = 0
        for point in driver:
            # Paired on the OBSERVATION date — same period, not same vintage.
            aligned = [_asof(series, point.observation_date) for series in others]
            if any(match is None for match in aligned):
                # No input observation exists for that period at all. Dropped
                # rather than back-filled, which is what trims a derived series
                # to where all of its inputs really existed: carrying the first
                # available breakeven backwards would invent TIPS pricing for
                # 1990, when TIPS did not trade.
                skipped += 1
                continue

            values = (point.value, *(match.value for match in aligned))  # type: ignore[union-attr]
            try:
                value = recipe.combine(values)  # type: ignore[operator]
            except (ValueError, ZeroDivisionError) as err:
                log.warning("%s: skipping %s (%s)", series_id, point.observation_date, err)
                continue

            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency=recipe.frequency,
                    unit=recipe.unit,
                    observation_date=point.observation_date,
                    # The composition rule, applied to the observations the
                    # period-join actually selected. Not a synthesized lag: see
                    # the module docstring for why that would be a lookahead the
                    # PIT layer cannot detect.
                    release_date=max(
                        point.release_date,
                        *(match.release_date for match in aligned),  # type: ignore[union-attr]
                    ),
                    value=value,
                )
            )

        log.info(
            "%s: derived %d observations from %s (%d dates dropped for unreleased inputs)",
            series_id,
            len(observations),
            ", ".join(recipe.inputs),
            skipped,
        )
        return observations

    def _load_input(
        self,
        base: str,
        series_id: str,
        start: date | None,
        end: date | None,
    ) -> list[_Point]:
        """One input series, oldest first, with its real release dates."""
        params: dict[str, str] = {"limit": str(self.limit)}
        if start is not None:
            params["from"] = start.isoformat()
        if end is not None:
            params["to"] = end.isoformat()

        response = self.transport.get(
            f"{base}/api/v1/series/{quote(series_id, safe='')}",
            params=params,
            cache_ttl=CACHE_TTL,
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response for {series_id}: {err}") from err

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ParseError(self.id, f"{series_id}: response is not a FinDyn envelope")

        rows = data.get("observations")
        if not isinstance(rows, list):
            raise ParseError(self.id, f"{series_id}: response carries no observations array")

        points: list[_Point] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            observation_date = _date(row.get("obs_date"))
            release_date = _date(row.get("release_date"))
            value = _float(row.get("value"))
            if observation_date is None or value is None:
                continue
            # A row with no release date cannot be composed: the maximum below
            # would silently ignore it and the result would claim to be knowable
            # earlier than it was.
            if release_date is None:
                raise ParseError(
                    self.id,
                    f"{series_id}: observation {observation_date} carries no release_date, "
                    "so it cannot be composed into a point-in-time derived series",
                )
            points.append(
                _Point(observation_date=observation_date, release_date=release_date, value=value)
            )

        # Sorted by observation date: that is the key the period-join walks.
        points.sort(key=lambda p: (p.observation_date, p.release_date))
        return points


def _asof(points: list[_Point], observation_date: date) -> _Point | None:
    """Newest point *observed* on or before ``observation_date``.

    Deliberately not keyed on the release date. The question here is which
    values describe the same period, and January's CAPE belongs beside January's
    real rate. Whether the pair was knowable — and when — is answered separately,
    by taking the maximum release date over exactly the points this returns.
    """
    match: _Point | None = None
    for point in points:  # sorted by observation_date
        if point.observation_date > observation_date:
            break
        match = point
    return match


def _date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed  # NaN check


def build_derived_provider(
    transport: Transport,
    env: Mapping[str, str] | None = None,
) -> DerivedProvider:
    import os

    resolved = env if env is not None else os.environ
    return DerivedProvider(transport, base_url=resolve_api_base(resolved))


__all__ = [
    "DEFAULT_LIMIT",
    "PREFIX",
    "RECIPES",
    "DerivedProvider",
    "Recipe",
    "build_derived_provider",
]
