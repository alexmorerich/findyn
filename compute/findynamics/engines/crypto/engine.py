"""FinCrypto — network scarcity. **Experimental, research-only** (Phase 5).

Pipeline: price panel -> supply schedule -> expanding liquidity beta -> jump
intensity -> vol/drawdown regime -> speculation index -> :class:`AssetState`.

Read this before reading anything else in the package
-----------------------------------------------------

**This engine publishes no expected return, and that is a modelling result
rather than an omission.** ``AssetState.expected_return`` is ``None``, set
explicitly and asserted in the tests. Every other engine here can point at
something that generates the number: FinMoney integrates an observable short
rate, FinRates fits a curve of traded yields, FinEquity discounts an earnings
stream, and FinGold — which has no cash flow either — publishes a
regime-conditional historical mean over a six-hundred-month sample and labels it
loudly as the past tense.

Bitcoin has none of those. There is no cash flow, no issuer, no coupon, and no
multi-century record of being treated as money; the whole tradeable history is
about fifteen years and four cycles. A regime-conditional mean over four cycles
is not a weak estimate of an expected return, it is a description of four
events. Publishing it as ``expected_return`` would put a number into the field
the portfolio layer optimizes against, and the portfolio layer would then be
allocating on the strength of 2017 and 2021 having happened. ``None`` is the
honest value and the schema was built to carry it (``03-contracts.md`` §1:
"``None`` where meaningless (crypto)").

What it *does* claim
--------------------

Three things, in descending order of how much they are worth:

* ``regime`` — ``winter | normal | frenzy`` from the trailing price path. Two
  thresholds you can state in a sentence; see ``regime.py`` on why a fitted
  switching model would be worse here despite being better statistics.
* ``speculation_index`` — 0-100, how much of the price is momentum. A
  conjunction of volatility, on-chain volume and distance above the
  liquidity-implied band.
* ``liquidity_beta`` — bitcoin's monthly co-movement with the money stock,
  estimated on an expanding window and published with units.

The price it reads is **stitched** from a daily close and a longer daily-average
history, validated and flagged per date by ``prices.py``; three of the confidence
terms below are priced off that provenance. See
``docs/design/crypto-price-record.md``.

``confidence`` is capped at 0.5 by construction — half of what FinGold's ceiling
allows itself, and FinGold is the other engine with nothing to discount. See
:meth:`CryptoEngine._confidence` for why that specific number.

Quarantine
----------

Nothing outside this package may import it (``01-target-architecture.md`` §3
rule 5, enforced by the ``Crypto is quarantined`` contract in
``compute/pyproject.toml``), the portfolio layer excludes it because
``experimental`` is ``True``, and ``config/engines/crypto.yaml`` ships
``enabled: false``. Three independent gates, because any one of them can be
switched off by someone who has not read this docstring.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.state import (
    AssetState,
    EngineOutput,
    Signal,
    WorldState,
)
from findynamics.core.engine import AssetEngine, StateUnavailable
from findynamics.core.registry import register_engine
from findynamics.engines.crypto import jumps as jumps_mod
from findynamics.engines.crypto import liquidity_beta as beta_mod
from findynamics.engines.crypto import prices as prices_mod
from findynamics.engines.crypto import regime as regime_mod
from findynamics.engines.crypto import scarcity as scarcity_mod
from findynamics.engines.crypto import speculation as speculation_mod
from findynamics.engines.crypto.domain import CRYPTO_METRICS, CRYPTO_REGIMES, regime_code

log = logging.getLogger("findynamics.engines.crypto")

MODEL_VERSION = "crypto-0.1.0"

#: Bitcoin has no exchange calendar. Every annualization in this engine uses 365,
#: and getting it wrong understates volatility and jump intensity by ~20% and
#: ~31% respectively — in the reassuring direction, silently.
CALENDAR_DAYS = 365

#: Roles without which there is no engine.
REQUIRED_ROLES = ("price",)

#: Roles the engine degrades without, and says so.
OPTIONAL_ROLES = (
    "price_fallback",
    "price_history",
    "m2",
    "central_bank_assets",
    "tx_volume_usd",
    "active_addresses",
    "transactions",
    "hash_rate",
)

#: On-chain roles carried through to ``engine_output`` as published series.
#:
#: Only ``tx_volume_usd`` feeds the model — it is the volume leg of the
#: speculation index. The other three are published because they are the
#: network measurements a page about network scarcity is *for*, and because a
#: configured series nothing reads is a series that quietly rots: nobody notices
#: when it stops arriving. Charting them means a broken feed is visible.
ONCHAIN_ROLES = ("tx_volume_usd", "active_addresses", "transactions", "hash_rate")


@dataclass(frozen=True)
class CryptoRules:
    """The four rule blocks, loaded once per run."""

    beta: beta_mod.BetaRules
    jumps: jumps_mod.JumpRules
    regime: regime_mod.RegimeRules
    speculation: speculation_mod.SpeculationRules

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> CryptoRules:
        return cls(
            beta=beta_mod.BetaRules.from_params(params),
            jumps=jumps_mod.JumpRules.from_params(params, where="engines/crypto.yaml"),
            regime=regime_mod.RegimeRules.from_params(params),
            speculation=speculation_mod.SpeculationRules.from_params(params),
        )


@dataclass(frozen=True)
class CryptoAnalysis:
    """Everything one run derives from the information set, computed once."""

    #: Daily panel on the price's own index: price, log_price, returns, the
    #: regime's three deciding quantities, and the supply schedule.
    daily: pd.DataFrame
    #: Month-end panel with ``ret`` — the monthly log return the beta reads.
    monthly: pd.DataFrame
    jumps: jumps_mod.JumpResult
    beta: beta_mod.BetaResult
    regime: regime_mod.RegimeView
    speculation: speculation_mod.SpeculationResult
    #: Role -> whether its input arrived at all.
    available: dict[str, bool]
    #: Where the price came from, per date and in aggregate. Carried rather than
    #: reduced to a flag: the record is stitched from a close and a daily
    #: average, and which one supplied a date changes what a single-day figure
    #: means (see prices.py).
    prices: prices_mod.PriceRecord
    rules: CryptoRules
    as_of: date

    @property
    def latest_key(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of)

    @property
    def has_regime(self) -> bool:
        return not self.regime.empty


@register_engine
class CryptoEngine(AssetEngine):
    """Regime, speculation index and liquidity beta for bitcoin. Research only."""

    name: ClassVar[str] = "crypto"
    version: ClassVar[str] = MODEL_VERSION

    #: Quarantines this engine from the portfolio layer (§3 rule 5). Not a label:
    #: ``core.registry.portfolio_engines`` filters on it, and the API publishes it
    #: so a consumer cannot read a crypto state without being told what it is.
    experimental: ClassVar[bool] = True

    def __init__(
        self,
        config: SeriesConfig | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._config = config or get_series_config()
        self._artifacts = artifacts or ArtifactStore()
        self._cache: tuple[object, CryptoAnalysis | None] | None = None

    # -- configuration ----------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        engine = self._config.engines.get(self.name)
        return dict(engine.params) if engine else {}

    @property
    def series_ids(self) -> dict[str, str]:
        """Role -> series id, from ``engines.crypto.series`` in series.yaml."""
        configured = self._config.engine_series(self.name)
        missing = [role for role in REQUIRED_ROLES if role not in configured]
        if missing:
            raise ValueError(
                f"series.yaml engines.crypto.series is missing required role(s) {missing}; "
                f"expected all of {list(REQUIRED_ROLES)}"
            )
        known = set(REQUIRED_ROLES) | set(OPTIONAL_ROLES)
        return {role: spec.id for role, spec in configured.items() if role in known}

    def required_series(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.series_ids.values())))

    @property
    def rules(self) -> CryptoRules:
        return CryptoRules.from_params(self.params)

    def _block(self, name: str) -> dict[str, Any]:
        raw = self.params.get(name) or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def history_days(self) -> int:
        outputs = self._block("outputs")
        key = "backfill_history_days" if self.full_history else "history_days"
        default = 40000 if self.full_history else 3650
        return int(outputs.get(key, default))

    # -- analysis ---------------------------------------------------------

    def analyze(self, world: WorldState) -> CryptoAnalysis | None:
        """One information set's full derivation, memoized on accessor identity.

        ``predict`` and ``outputs`` are called back to back with the same world,
        and running the expanding regression twice per night is waste.
        """
        if self._cache is not None and self._cache[0] is world.series:
            return self._cache[1]
        analysis = self._analyze(world)
        self._cache = (world.series, analysis)
        return analysis

    # Price resolution moved to prices.py in the P5 backfill: three roles, a
    # close/average distinction that changes what some downstream numbers mean,
    # and a splice that has to be validated rather than assumed. That is more
    # than a `combine_first` and more than belongs inline in the engine.

    def _analyze(self, world: WorldState) -> CryptoAnalysis | None:
        rules = self.rules
        ids = self.series_ids

        frame = world.series.wide(self.required_series())
        if frame.empty:
            log.warning("crypto: no observations knowable at %s", world.as_of)
            return None

        try:
            record = prices_mod.build(frame, ids)
        except prices_mod.PriceUnavailable as err:
            log.warning("crypto: no bitcoin price knowable at %s — %s", world.as_of, err)
            return None
        price = record.price
        if price.empty:
            log.warning("crypto: no bitcoin price knowable at %s", world.as_of)
            return None

        # The price is the spine: this engine speaks about dates bitcoin traded
        # on, which — unlike every other engine here — is all of them.
        index = price.index
        aligned = frame.reindex(index).ffill()

        daily = pd.DataFrame(index=index)
        daily["price"] = price
        daily["log_price"] = np.log(price)
        daily["log_return"] = daily["log_price"].diff()
        # Provenance per date, published as a series. 1.0 marks a date the
        # daily-average history role supplied rather than a close — a different
        # statistic on any single day, which is why it travels with the price
        # instead of being summarised away (same reasoning as gold's
        # `real_rate_is_ex_post`).
        daily["price_is_daily_average"] = record.is_daily_average.astype(float)

        regime_view = regime_mod.classify(price, rules.regime)
        daily["drawdown"] = regime_view.drawdown
        daily["return_12m"] = regime_view.return_12m
        daily["realized_vol"] = regime_view.realized_vol

        # Pure calendar arithmetic — no market data, so no PIT treatment needed.
        supply = scarcity_mod.schedule(index)
        for column in ("issued_supply", "issuance_rate", "stock_to_flow"):
            daily[column] = supply[column]

        detected = jumps_mod.detect(daily["log_return"], rules.jumps, label="crypto jumps")

        # The regression is monthly because its regressor is. See liquidity_beta.
        monthly = daily[["log_price"]].resample("ME").last()
        monthly["ret"] = monthly["log_price"].diff()
        monthly = monthly.dropna(subset=["ret"])

        liquidity, legs = beta_mod.composite_liquidity(frame, ids, rules.beta)
        monthly_liquidity = (
            liquidity.reindex(monthly.index, method="ffill")
            if not liquidity.empty
            else pd.Series(dtype=float)
        )
        beta_result = beta_mod.estimate(monthly["ret"], monthly_liquidity, rules.beta, legs=legs)

        # Month-end values carried forward onto the daily index: a coefficient
        # published at month end applies from then until the next one, which is
        # what a run on any day in between would have used.
        band_excess = (
            beta_mod.excess_over_band(beta_result, rules.beta).reindex(index, method="ffill")
            if not beta_result.empty
            else None
        )

        volume_id = ids.get("tx_volume_usd")
        volume = aligned[volume_id] if volume_id and volume_id in aligned else None
        speculation = speculation_mod.compute(
            daily["realized_vol"], volume, band_excess, rules.speculation, index=index
        )
        daily["speculation_index"] = speculation.index
        daily["liquidity_residual"] = (
            beta_mod.level_deviation(beta_result).reindex(index, method="ffill")
            if not beta_result.empty
            else np.nan
        )
        daily["liquidity_beta"] = (
            beta_result.beta.reindex(index, method="ffill") if not beta_result.empty else np.nan
        )
        daily["liquidity_beta_r2"] = (
            beta_result.r_squared.reindex(index, method="ffill")
            if not beta_result.empty
            else np.nan
        )
        daily["jump_intensity"] = detected.intensity.reindex(index)
        volume_ratio = (
            speculation_mod.volume_trend(volume, rules.speculation).reindex(index)
            if volume is not None
            else pd.Series(np.nan, index=index)
        )
        daily["tx_volume_trend"] = volume_ratio

        # The network measurements themselves, carried onto the price spine so
        # they can be charted. Only tx_volume_usd feeds the model; the rest are
        # published so that a feed which stops arriving is visible rather than
        # silently absent from a series nobody looks at.
        for role in ONCHAIN_ROLES:
            series_id = ids.get(role)
            daily[role] = (
                aligned[series_id]
                if series_id and series_id in aligned
                else pd.Series(np.nan, index=index)
            )

        available = {
            role: bool(series_id in frame and frame[series_id].notna().any())
            for role, series_id in ids.items()
        }
        missing = [role for role, present in available.items() if not present]
        if missing:
            log.info("crypto inputs absent from this information set: %s", ", ".join(missing))

        return CryptoAnalysis(
            daily=daily,
            monthly=monthly,
            jumps=detected,
            beta=beta_result,
            regime=regime_view,
            speculation=speculation,
            available=available,
            prices=record,
            rules=rules,
            as_of=index[-1].date(),
        )

    # -- AssetEngine ------------------------------------------------------

    def fit(self, world: WorldState) -> None:
        """Nothing is fitted between runs, and that is the design.

        Every estimator here is an expanding-window closed form recomputed from
        the information set each run: the OLS slope is five running sums, the
        jump threshold is a function of the sample size, and the regime is two
        comparisons. There is no likelihood to maximize and therefore no
        parameter that could drift between a refit and the run that consumes it.

        The method records what it saw so ``monthly_refit`` leaves a trace and so
        an operator can tell "no fit was needed" from "the refit never ran" —
        which are the same silence otherwise.
        """
        analysis = self.analyze(world)
        if analysis is None:
            log.warning("crypto.fit: nothing knowable at %s", world.as_of)
            return

        beta = analysis.beta.latest()
        self._artifacts.save(
            self.name,
            {
                "fitted_as_of": world.as_of.isoformat(),
                "model_version": self.version,
                "note": (
                    "FinCrypto holds no fitted parameters: every estimator is an "
                    "expanding-window closed form recomputed per run. This record "
                    "exists so a missing refit is distinguishable from an unneeded one."
                ),
                "observed": {
                    "months": int(len(analysis.monthly)),
                    "liquidity_beta": None if beta is None else round(beta, 6),
                },
            },
        )
        log.info(
            "crypto.fit: %d months through %s (nothing to fit; beta %s)",
            len(analysis.monthly),
            analysis.as_of,
            "unavailable" if beta is None else f"{beta:.2f}",
        )

    def predict(self, world: WorldState) -> AssetState:
        """Today's crypto state. **``expected_return`` is None** — see the module docstring."""
        analysis = self.analyze(world)
        if analysis is None:
            raise StateUnavailable(
                f"crypto: no bitcoin price knowable at {world.as_of}; backfill "
                "STOOQ:BTCUSD (or its YAHOO:BTC-USD fallback) before running the engine"
            )
        label = analysis.regime.latest()
        if label is None:
            raise StateUnavailable(
                "crypto: the price history knowable here is shorter than the regime "
                "model's window, so there is no regime to publish. The per-date "
                "outputs are still published."
            )

        return AssetState(
            asset=self.name,
            as_of=analysis.as_of,
            regime=label,
            # Explicit, not forgotten. The whole first section of this module's
            # docstring is about this None; a test asserts it stays one.
            expected_return=None,
            risk_score=self._risk_score(analysis),
            confidence=self._confidence(analysis),
            signals=self._signals(analysis, label),
            model_version=self.version,
            components=self._components(world, analysis, label),
        )

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Per-date price, regime inputs, supply schedule, beta and speculation."""
        analysis = self.analyze(world)
        if analysis is None:
            return ()

        cutoff = analysis.latest_key - pd.Timedelta(days=self.history_days)
        rows: list[EngineOutput] = []

        daily = analysis.daily
        for metric in CRYPTO_METRICS:
            if metric == "regime_code":
                continue
            if metric in daily:
                rows.extend(self._rows(metric, daily[metric], cutoff))

        # engine_output stores REALs, so the label travels as its index in the
        # vocabulary with the name itself in `meta`.
        labels = analysis.regime.label.loc[cutoff:].dropna()
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="regime_code",
                as_of=key.date(),
                value=float(regime_code(str(value))),
                meta={"regime": str(value)},
            )
            for key, value in labels.items()
        )
        return tuple(rows)

    # -- internals --------------------------------------------------------

    def _rows(self, metric: str, series: pd.Series, cutoff: pd.Timestamp) -> list[EngineOutput]:
        if series is None or series.empty:
            return []
        window = series.loc[cutoff:].dropna()
        return [
            EngineOutput(
                asset=self.name,
                metric=metric,
                as_of=key.date(),
                value=round(float(value), 8),
            )
            for key, value in window.items()
            if np.isfinite(value)
        ]

    def _risk_score(self, analysis: CryptoAnalysis) -> float:
        """0-100 from realized volatility and jump intensity.

        The same two-term shape FinGold uses and for the same reason — a
        volatility number alone understates an asset whose losses arrive in three
        sessions — but on a scale of its own. ``vol_reference_pct`` is 150%
        rather than gold's 30%, because a 0-100 axis calibrated on gold would put
        bitcoin's *quiet* years at 100 and say nothing about the difference
        between 2019 and 2021.

        A consequence worth stating plainly: this score is comparable across
        dates for bitcoin and is **not** comparable with another engine's risk
        score, because the two are percentages of different references. The
        portfolio layer never sees it — this engine is experimental — which is
        the only reason that is tolerable.
        """
        params = self._block("risk")
        vol_reference = float(params.get("vol_reference_pct", 150.0))
        jump_reference = float(params.get("jump_intensity_reference", 12.0))
        vol_weight = float(params.get("vol_weight", 0.7))
        jump_weight = float(params.get("jump_weight", 0.3))

        vol = analysis.daily["realized_vol"].dropna()
        vol_term = (
            min(float(vol.iloc[-1]) / max(vol_reference, 1e-9), 1.0)
            if not vol.empty
            else float(params.get("vol_fallback", 0.6))
        )
        intensity = analysis.jumps.latest_intensity()
        jump_term = min((intensity or 0.0) / max(jump_reference, 1e-9), 1.0)

        score = 100.0 * (vol_weight * vol_term + jump_weight * jump_term)
        return round(min(max(score, 0.0), 100.0), 4)

    def _confidence(self, analysis: CryptoAnalysis) -> float:
        """0-1, capped at 0.5 by construction. The cap is the honest part.

        Why 0.5 and not the 0.7 FinGold allows itself, when both engines model an
        asset with no cash flow:

        * **Sample.** Gold's regime model sees six hundred months across four
          distinct monetary eras. This engine sees about a hundred and eighty
          months and four cycles, and two of those cycles are most of the
          variance. Nothing estimated on four events deserves to be more than
          half believed.
        * **Stationarity.** Gold in 1975 and gold in 2025 are the same asset
          doing the same job. Bitcoin's holder base, market structure,
          derivatives, custody and regulatory status have all changed
          discontinuously inside the sample, so an expanding window is averaging
          over regimes of the *market*, not just of the price.
        * **The liquidity beta is a co-movement, not a mechanism.** Gold's real
          rate driver is an arithmetic cost of carry. This engine's macro
          relationship is two trending series that have trended together, which
          is a weaker kind of claim and is stated as one.

        The ceiling is a ceiling, not a floor: penalties below take it down from
        there, and no branch can raise it. The clamp at the end is what makes
        that structural rather than a convention — a future edit that adds a
        bonus term still cannot publish 0.6.
        """
        params = self._block("confidence")
        ceiling = float(params.get("ceiling", 0.5))
        confidence = ceiling

        if analysis.beta.latest() is None:
            confidence -= float(params.get("no_liquidity_beta_penalty", 0.1))
        elif not all(analysis.beta.legs.values()):
            # One leg of the money composite is missing, so the regressor is not
            # the quantity the beta is documented to be against.
            confidence -= float(params.get("partial_liquidity_penalty", 0.05))

        missing_terms = len(analysis.speculation.missing_terms)
        confidence -= missing_terms * float(params.get("missing_speculation_term_penalty", 0.05))

        if analysis.jumps.empty:
            confidence -= float(params.get("no_jump_detector_penalty", 0.05))
        if analysis.prices.from_fallback:
            confidence -= float(params.get("fallback_price_penalty", 0.02))
        # The average-based prefix is real history and worth having; it is also
        # not the same statistic as a close, so the more of the window it
        # carries, the less the single-day figures mean. Scaled by share rather
        # than charged as a flat fee, because a record that is 5% average and one
        # that is 100% average are different claims.
        confidence -= (
            float(params.get("daily_average_price_penalty", 0.05)) * analysis.prices.average_share
        )
        if analysis.prices.declined_reason is not None:
            confidence -= float(params.get("declined_splice_penalty", 0.03))

        # Clamped to the ceiling, not to 1.0. See the docstring: this is what
        # makes the cap a property of the engine rather than of today's config.
        return round(min(max(confidence, 0.0), ceiling), 4)

    def _signals(self, analysis: CryptoAnalysis, label: str) -> tuple[Signal, ...]:
        """Directional reads. ``direction`` is +1 supportive of the asset, -1 adverse.

        Note what ``speculation_index`` does with its sign: a high reading is
        ``-1``. That is not a view about where the price goes next — this engine
        has none — it is the statement that a price which is mostly momentum is a
        worse risk than one which is not, which is the question ``direction``
        asks.
        """
        signals: list[Signal] = []
        thresholds = self._block("signals")

        speculation = analysis.speculation.latest()
        if speculation is not None:
            elevated = float(thresholds.get("speculation_elevated", 60.0))
            subdued = float(thresholds.get("speculation_subdued", 20.0))
            terms = analysis.speculation.term_count
            signals.append(
                Signal(
                    name="speculation_index",
                    value=round(speculation, 4),
                    direction=-1
                    if speculation >= elevated
                    else (1 if speculation < subdued else 0),
                    note=(
                        f"0-100 from {terms} of 3 terms (volatility x on-chain volume trend x "
                        "distance above the liquidity-implied band), combined geometrically so "
                        "that all three must be present. Not a price view: it says how much of "
                        "the price is momentum, which is a risk statement."
                    ),
                )
            )

        beta = analysis.beta.latest()
        if beta is not None:
            r_squared = analysis.beta.latest_r_squared()
            months = analysis.beta.n_observations.dropna()
            fit = "R2 unavailable" if r_squared is None else f"R2 {r_squared:.3f}"
            signals.append(
                Signal(
                    name="liquidity_beta",
                    value=round(beta, 6),
                    # Sensitivity, not direction. A high beta is neither good nor
                    # bad on its own — it says the asset moves with the money
                    # supply, and the money supply moves both ways.
                    direction=0,
                    note=(
                        "Monthly log return per unit log change in the money stock "
                        "(US M2 + Fed assets), expanding window over "
                        f"{int(months.iloc[-1]) if not months.empty else 0} months, {fit}. "
                        "A co-movement in a sample, not a mechanism and not a forecast — and "
                        "read the R2 before reading the coefficient, because on this data it "
                        "is small enough that the slope is mostly describing noise."
                    ),
                )
            )

        signals.append(
            Signal(
                name="regime",
                value=float(regime_code(label)),
                # `normal` is the only neutral reading. Winter and frenzy are
                # both -1 and for opposite-looking reasons: one is a market that
                # has already fallen a long way, the other is one whose price is
                # mostly momentum. `direction` asks about risk, and both are more
                # of it than the middle.
                direction=0 if label == "normal" else -1,
                note=(
                    f"{label}: from the trailing price path alone — drawdown from the "
                    "trailing-year peak, 12-month return and realized volatility. Winter is "
                    "checked first, because a market that is both up on the year and 45% off "
                    "its peak is a top rather than a boom."
                ),
            )
        )

        intensity = analysis.jumps.latest_intensity()
        if intensity is not None:
            signals.append(
                Signal(
                    name="jump_intensity",
                    value=round(intensity, 4),
                    direction=-1
                    if intensity >= float(thresholds.get("jump_intensity_elevated", 12.0))
                    else 0,
                    note=(
                        f"{intensity:.1f} Lee-Mykland detections per year, annualized on a "
                        "365-day calendar. The threshold scales with each date's own trailing "
                        "volatility, so this reads 'disorderly for bitcoin lately' rather than "
                        "'volatile' — bitcoin is volatile throughout."
                    ),
                )
            )

        record = analysis.prices
        if record.spliced:
            signals.append(
                Signal(
                    name="price_record_spliced",
                    value=round(record.average_share, 6),
                    # Not a market read. 0 because it says something about the
                    # measurement rather than about the asset, and `direction`
                    # is reserved for the latter.
                    direction=0,
                    note=(
                        f"{record.series_id}: {record.average_share:.0%} of the record is a "
                        "volume-weighted daily AVERAGE across exchanges rather than a close. "
                        "The two agree on the level and on volatility in aggregate, and differ "
                        "on any single high-range day — 2020-03-12 is a 4,971 close against a "
                        "7,937 average. Jump dates either side of the seam are not strictly the "
                        "same measurement."
                    ),
                )
            )
        elif record.declined_reason is not None:
            signals.append(
                Signal(
                    name="price_history_declined",
                    value=1.0,
                    direction=0,
                    note=(
                        f"The deep-history extension was refused: {record.declined_reason}. "
                        "The record starts where the closes do, so the sample is shorter than "
                        "configured and the early cycles are absent."
                    ),
                )
            )

        # The experimental status is a signal, not only a flag on the envelope:
        # anything that reads signals and not the envelope must still be told.
        signals.append(
            Signal(
                name="experimental",
                value=1.0,
                direction=0,
                note=(
                    "Research only. This engine publishes no expected return by design, "
                    "its confidence is capped at 0.5, and the portfolio layer excludes it. "
                    "See engines/crypto/engine.py for why each of those is a result rather "
                    "than a limitation."
                ),
            )
        )

        absent = [role for role, present in analysis.available.items() if not present]
        if absent:
            signals.append(
                Signal(
                    name="inputs_absent",
                    value=float(len(absent)),
                    direction=0,
                    note=(
                        f"{len(absent)} configured input(s) are not in this information set: "
                        f"{', '.join(sorted(absent))}. Shown as absent rather than as zero. "
                        "The paid on-chain vendors are deliberately not configured at all — "
                        "see data/providers/registry.py."
                    ),
                )
            )
        return tuple(signals)

    #: Layer 0 factors reading the same forces this engine's inputs do. Named
    #: here rather than in ``core`` because *which* shared factors are relevant to
    #: bitcoin is a fact about bitcoin.
    SHARED_FACTORS: ClassVar[tuple[str, ...]] = (
        "global_liquidity",
        "liquidity",
        "risk_appetite",
        "real_rate",
    )

    @staticmethod
    def _shared_factors(world: WorldState) -> dict[str, float]:
        """Layer 0's reading of the same forces, published beside the engine's own.

        Not an input to anything, deliberately. ``global_liquidity`` is a 0-100
        expanding percentile on the risk-supportive axis; the liquidity beta is a
        regression coefficient in log points per log point. Feeding the score into
        the regression would produce a slope with no statable units, which is why
        ``liquidity_beta.py`` regresses on the level change and this method exists
        only so the page can show the two side by side. Disagreement between them
        is the interesting case and is invisible if they are averaged.
        """
        out: dict[str, float] = {}
        for name in CryptoEngine.SHARED_FACTORS:
            score = world.factor_score(name)
            if score is not None and math.isfinite(score):
                out[f"factor_{name}"] = round(float(score), 4)
        return out

    def _components(
        self, world: WorldState, analysis: CryptoAnalysis, label: str
    ) -> dict[str, float]:
        """The explainability trace: regime inputs, supply, beta, speculation terms."""
        components: dict[str, float] = {"regime_code": float(regime_code(label))}

        row = analysis.daily.iloc[-1]
        for key in (
            "price",
            "drawdown",
            "return_12m",
            "realized_vol",
            "issued_supply",
            "issuance_rate",
            "stock_to_flow",
            "tx_volume_trend",
            "liquidity_residual",
            "jump_intensity",
        ):
            value = row.get(key)
            if value is not None and np.isfinite(value):
                components[key] = round(float(value), 6)

        beta = analysis.beta.latest()
        if beta is not None:
            components["liquidity_beta"] = round(beta, 6)
        r_squared = analysis.beta.latest_r_squared()
        if r_squared is not None:
            components["liquidity_beta_r2"] = round(r_squared, 6)
        months = analysis.beta.n_observations.dropna()
        if not months.empty:
            components["liquidity_beta_months"] = float(int(months.iloc[-1]))

        speculation = analysis.speculation.latest()
        if speculation is not None:
            components["speculation_index"] = round(speculation, 4)
        components["speculation_terms"] = float(analysis.speculation.term_count)
        components.update(analysis.speculation.latest_terms())

        days, projected = scarcity_mod.days_to_next_halving(analysis.as_of)
        components["days_to_next_halving"] = days
        # 1.0 means the date above is a projection at the nominal block time, not
        # an observed halving. Carried as a number because components are numeric.
        components["next_halving_is_projected"] = 1.0 if projected else 0.0

        # Loud, and in the components rather than only in a docstring: something
        # rendering this state needs to be able to see the claim being declined.
        components["expected_return_is_deliberately_absent"] = 1.0
        components["confidence_ceiling"] = float(self._block("confidence").get("ceiling", 0.5))
        components["price_from_fallback_source"] = 1.0 if analysis.prices.from_fallback else 0.0
        components["price_average_share"] = round(analysis.prices.average_share, 6)
        components["price_is_spliced"] = 1.0 if analysis.prices.spliced else 0.0
        components["price_history_declined"] = (
            1.0 if analysis.prices.declined_reason is not None else 0.0
        )

        components.update(self._shared_factors(world))
        return components


__all__ = [
    "CALENDAR_DAYS",
    "CRYPTO_METRICS",
    "CRYPTO_REGIMES",
    "MODEL_VERSION",
    "OPTIONAL_ROLES",
    "REQUIRED_ROLES",
    "CryptoAnalysis",
    "CryptoEngine",
    "CryptoRules",
]
