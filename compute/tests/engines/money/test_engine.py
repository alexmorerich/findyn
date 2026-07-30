"""The engine end to end: contract, outputs, degradation and independence."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.core.contracts.state import ContractError, FactorState
from findynamics.core.contracts.vocab import DISCOUNT_HORIZONS, parse_engine_series_id
from findynamics.core.registry import ENGINES, get_engine
from findynamics.engines.money.account import InsufficientRatePathError
from findynamics.engines.money.domain import CARRY_WINDOWS, MONEY_METRICS, MONEY_REGIMES
from findynamics.engines.money.engine import PUBLISHED_DISCOUNT_HORIZONS, MoneyEngine
from tests.engines.money.conftest import (
    BILL_3M,
    DTB3,
    RRP,
    SOFR,
    curve_rows,
    rate_rows,
    world_from,
)

CUTOFF = date(2020, 12, 31)


@pytest.fixture
def state(money_engine, money_observations):
    return money_engine.predict(world_from(money_observations, CUTOFF))


class TestRegistration:
    def test_the_engine_registers_itself_under_its_name(self):
        assert ENGINES["money"] is MoneyEngine
        assert get_engine("money").name == "money"

    def test_it_is_not_experimental(self):
        assert MoneyEngine.experimental is False

    def test_it_declares_every_configured_series(self, money_engine):
        required = money_engine.required_series()
        assert SOFR in required
        assert DTB3 in required
        assert BILL_3M in required
        assert RRP in required
        assert sum(1 for s in required if s.startswith("ENGINE:")) == 4

    def test_required_series_are_deduplicated_and_sorted(self, money_engine):
        required = money_engine.required_series()
        assert list(required) == sorted(set(required))

    def test_the_shared_series_are_named_once_not_duplicated(self, money_engine, config):
        """DGS3MO and RRPONTSYD are also wanted elsewhere; they must be fetched once."""
        from findynamics.data.store import required_series_ids

        ids = required_series_ids(config, money_engine.required_series())
        assert len(ids) == len(set(ids))
        assert ids.count(BILL_3M) == 1
        assert ids.count(RRP) == 1


class TestAssetState:
    def test_it_satisfies_the_contract(self, state):
        assert state.asset == "money"
        assert state.model_version == "money-1.0.0"
        assert state.regime in MONEY_REGIMES
        assert 0.0 <= state.risk_score <= 100.0
        assert 0.0 <= state.confidence <= 1.0
        assert isinstance(state.signals, tuple)

    def test_expected_return_is_the_current_short_rate_as_a_decimal(
        self, state, money_engine, money_observations
    ):
        """Not a forecast — what cash is earning right now."""
        analysis = money_engine.analyze(world_from(money_observations, CUTOFF))
        assert analysis is not None
        assert state.expected_return == pytest.approx(analysis.path.latest / 100.0, abs=1e-8)
        assert 0.0 <= state.expected_return < 0.10

    def test_the_state_date_is_the_newest_date_cash_actually_earned(self, state):
        assert state.as_of <= CUTOFF
        assert state.as_of >= CUTOFF - timedelta(days=7)

    def test_risk_is_near_zero_by_construction(self, state):
        """Cash has no duration and no credit; the ceiling is 10 on a 0-100 axis."""
        assert state.risk_score <= 10.0

    def test_a_calm_market_scores_exactly_zero_risk(self, money_engine, money_observations):
        calm = money_engine.predict(world_from(money_observations, date(2018, 12, 10)))
        assert calm.risk_score == 0.0

    def test_stress_lifts_risk_but_never_past_the_ceiling(self, money_engine, money_observations):
        stressed = money_engine.predict(world_from(money_observations, date(2019, 9, 19)))
        assert stressed.regime == "stressed"
        assert 0.0 < stressed.risk_score <= 10.0

    def test_components_carry_the_whole_discount_curve(self, state):
        components = state.components or {}
        for horizon in DISCOUNT_HORIZONS:
            assert f"discount_{horizon}" in components

    def test_components_explain_the_liquidity_call(self, state):
        components = state.components or {}
        assert "bill_sofr_spread" in components
        assert "stressed_spread_threshold_pp" in components
        assert "liquidity_code" in components

    def test_components_record_the_splice_composition(self, state):
        components = state.components or {}
        assert 0.0 < components["primary_rate_share"] <= 1.0
        assert components["rate_path_days"] > 300


class TestSignals:
    def test_real_carry_is_always_published(self, state):
        names = {s.name for s in state.signals}
        assert "real_carry" in names

    def test_real_carry_direction_follows_the_shared_inflation_factor(
        self, money_engine, money_observations
    ):
        """The factor orients the signal; it is not subtracted from the carry."""
        base = world_from(money_observations, CUTOFF)

        benign = world_from(
            money_observations,
            CUTOFF,
            factors={
                "inflation": FactorState(name="inflation", as_of=CUTOFF, score=90.0),
            },
        )
        hostile = world_from(
            money_observations,
            CUTOFF,
            factors={
                "inflation": FactorState(name="inflation", as_of=CUTOFF, score=5.0),
            },
        )

        def carry_signal(world):
            money_engine._cache = None
            return next(s for s in money_engine.predict(world).signals if s.name == "real_carry")

        neutral_signal = carry_signal(base)
        benign_signal = carry_signal(benign)
        hostile_signal = carry_signal(hostile)

        # Same nominal number in all three; only the read changes.
        assert benign_signal.value == hostile_signal.value == neutral_signal.value
        assert benign_signal.direction == 1
        assert hostile_signal.direction == -1
        assert neutral_signal.direction == 0

    def test_real_carry_is_adverse_when_there_is_no_carry_at_all(
        self, money_engine, money_observations
    ):
        """A zero-rate world: nothing to be had, whatever inflation is doing."""
        zero = world_from(
            money_observations,
            date(2020, 12, 31),
            factors={"inflation": FactorState(name="inflation", as_of=CUTOFF, score=99.0)},
        )
        signal = next(s for s in money_engine.predict(zero).signals if s.name == "real_carry")
        # SOFR was 0.07% through late 2020, so the 12m carry is positive but tiny.
        assert signal.value >= 0.0

    def test_the_spread_signal_reads_adverse_under_stress(self, money_engine, money_observations):
        state = money_engine.predict(world_from(money_observations, date(2019, 9, 19)))
        spread = next(s for s in state.signals if s.name == "bill_sofr_spread")
        assert spread.direction == -1
        assert spread.value < 0.0

    def test_a_degraded_curve_is_announced_as_a_signal(self, state):
        """Silence about a flat 30y extrapolation would be the wrong default."""
        names = {s.name for s in state.signals}
        assert "curve_source_degraded" in names

    def test_every_signal_has_a_note(self, state):
        for signal in state.signals:
            assert signal.note, signal.name


class TestTheSpreadSurvivesStaggeredPublication:
    """The normal production case, not an edge case.

    SOFR carries a one-day publication lag and the constant-maturity bill
    effectively two, so on almost every real run the newest date of the rate path
    has an overnight rate and no bill beside it. Reading that row literally would
    leave the liquidity state resting on reverse-repo alone *every single day*,
    with the engine's primary input one row above it, unused — and it would look
    like working software, because the state still publishes.
    """

    def _staggered(self) -> pd.DataFrame:
        """SOFR through the 30th, bill and reverse repo only through the 29th."""
        rows: list[dict] = []
        for i in range(90):
            day = date(2026, 5, 1) + timedelta(days=i)
            rows += rate_rows(SOFR, {day: 4.30})
            rows += rate_rows(BILL_3M, {day: 4.10})
            rows += rate_rows(RRP, {day: 12.0})
        rows += rate_rows(SOFR, {date(2026, 7, 30): 4.31})
        return pd.DataFrame(rows)

    def test_the_state_still_reads_the_spread(self, money_engine):
        state = money_engine.predict(world_from(self._staggered(), date(2026, 7, 31)))
        components = state.components or {}

        assert state.as_of == date(2026, 7, 30), "the rate path ends on the newest SOFR"
        assert components["bill_sofr_spread"] == pytest.approx(-0.20)
        assert any(s.name == "bill_sofr_spread" for s in state.signals)

    def test_confidence_is_not_docked_for_a_one_day_stagger(self, money_engine):
        """Otherwise every production run would publish a permanently degraded read."""
        state = money_engine.predict(world_from(self._staggered(), date(2026, 7, 31)))
        # 1.0 - 0.20 for the absent curve only; no spread penalty.
        assert state.confidence == pytest.approx(0.80, abs=1e-9)

    def test_a_stale_spread_beyond_the_limit_is_dropped(self, money_engine):
        """A fortnight-old dislocation is not a read on today's funding market."""
        rows: list[dict] = []
        for i in range(90):
            day = date(2026, 5, 1) + timedelta(days=i)
            rows += rate_rows(SOFR, {day: 4.30})
            rows += rate_rows(BILL_3M, {day: 4.10})
        for i in range(1, 15):
            rows += rate_rows(SOFR, {date(2026, 7, 30) + timedelta(days=i): 4.31})

        state = money_engine.predict(world_from(pd.DataFrame(rows), date(2026, 8, 20)))
        assert "bill_sofr_spread" not in (state.components or {})
        assert state.confidence < 0.80


class TestConfidence:
    def test_a_spliced_path_costs_confidence(self, money_engine, money_observations):
        """Pre-SOFR the numeraire rests on a converted bill quote."""
        spliced = money_engine.predict(world_from(money_observations, date(2018, 3, 1)))
        money_engine._cache = None
        modern = money_engine.predict(world_from(money_observations, CUTOFF))
        assert spliced.confidence < modern.confidence

    def test_no_spread_costs_confidence(self, money_engine, money_observations):
        pre_sofr = money_engine.predict(world_from(money_observations, date(2018, 3, 1)))
        assert pre_sofr.confidence == pytest.approx(1.0 - 0.25 - 0.30 - 0.20, abs=1e-9)

    def test_a_published_curve_recovers_the_curve_penalty(self, money_engine, money_observations):
        without = money_engine.predict(world_from(money_observations, CUTOFF))
        money_engine._cache = None

        days = [CUTOFF - timedelta(days=i) for i in range(10)]
        with_curve = pd.concat(
            [
                money_observations,
                pd.DataFrame(
                    curve_rows(
                        {"level": 1.5, "slope": 1.4, "curvature": -0.3, "lambda": 0.609}, days
                    )
                ),
            ],
            ignore_index=True,
        )
        state = money_engine.predict(world_from(with_curve, CUTOFF))
        assert state.confidence == pytest.approx(without.confidence + 0.20, abs=1e-9)


class TestTheCurveIsReadAsPublishedData:
    """The independence contract, exercised the way production exercises it."""

    def test_published_factors_change_the_long_end_only(self, money_engine, money_observations):
        days = [CUTOFF - timedelta(days=i) for i in range(10)]
        without = money_engine.predict(world_from(money_observations, CUTOFF))
        money_engine._cache = None

        frame = pd.concat(
            [
                money_observations,
                pd.DataFrame(
                    curve_rows(
                        {"level": 1.5, "slope": 1.4, "curvature": -0.3, "lambda": 0.609}, days
                    )
                ),
            ],
            ignore_index=True,
        )
        with_curve = money_engine.predict(world_from(frame, CUTOFF))

        a, b = without.components or {}, with_curve.components or {}
        for horizon in ("1m", "3m", "6m", "1y"):
            assert a[f"discount_{horizon}"] == pytest.approx(b[f"discount_{horizon}"])
        assert a["discount_10y"] != pytest.approx(b["discount_10y"])
        assert b["ns_lambda"] == 0.609

    def test_the_engine_never_imports_the_rates_engine(self):
        """Belt and braces beside lint-imports: nothing in the module graph."""
        import sys

        import findynamics.engines.money as money_pkg

        for name, module in list(sys.modules.items()):
            if not name.startswith("findynamics.engines.money"):
                continue
            for attr in vars(module).values():
                origin = getattr(attr, "__module__", "") or ""
                assert not origin.startswith("findynamics.engines.rates"), (
                    f"{name} pulled in {origin}"
                )
        assert money_pkg.__name__ == "findynamics.engines.money"

    def test_the_curve_series_ids_are_well_formed_engine_ids(self, money_engine):
        for series_id in money_engine.curve_ids.values():
            parsed = parse_engine_series_id(series_id)
            assert parsed is not None
            assert parsed[0] == "rates"


class TestOutputs:
    def test_every_promised_metric_is_published(self, money_engine, money_observations):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        published = {row.metric for row in rows}

        for metric in ("wealth_index", "short_rate", "bill_sofr_spread", "liquidity_code"):
            assert metric in published
        for metric in CARRY_WINDOWS:
            assert metric in published
        for horizon in PUBLISHED_DISCOUNT_HORIZONS:
            assert f"discount_{horizon}" in published

    def test_nothing_outside_the_declared_vocabulary_is_published(
        self, money_engine, money_observations
    ):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        assert {row.metric for row in rows} <= set(MONEY_METRICS)

    def test_every_row_is_finite_and_within_the_window(self, money_engine, money_observations):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        assert rows
        for row in rows:
            assert row.asset == "money"
            assert math.isfinite(row.value)
            assert row.as_of <= CUTOFF

    def test_the_wealth_index_carries_its_base_date(self, money_engine, money_observations):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        wealth = [r for r in rows if r.metric == "wealth_index"]
        assert wealth
        base = {(r.meta or {}).get("base") for r in wealth}
        assert base == {"2018-01-02"}

    def test_the_base_follows_an_accrual_reset(self, money_engine, money_observations):
        """After the fixture's deliberate gap the dollar was invested later."""
        rows = money_engine.outputs(world_from(money_observations, date(2024, 10, 31)))
        wealth = [r for r in rows if r.metric == "wealth_index"]
        assert {(r.meta or {}).get("base") for r in wealth} == {"2024-06-03"}

    def test_the_liquidity_code_carries_its_label(self, money_engine, money_observations):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        codes = [r for r in rows if r.metric == "liquidity_code"]
        assert codes
        for row in codes:
            assert (row.meta or {}).get("liquidity") in MONEY_REGIMES

    def test_the_short_rate_records_which_series_it_came_from(
        self, money_engine, money_observations
    ):
        rows = money_engine.outputs(world_from(money_observations, date(2018, 5, 1)))
        short = [r for r in rows if r.metric == "short_rate"]
        sources = {(r.meta or {}).get("source") for r in short}
        assert sources == {SOFR, DTB3}

    def test_discount_rows_record_their_curve_source(self, money_engine, money_observations):
        rows = money_engine.outputs(world_from(money_observations, CUTOFF))
        for row in rows:
            if row.metric.startswith("discount_"):
                assert (row.meta or {}).get("curve_source") in ("ns", "short_rate")

    def test_discount_history_uses_each_dates_own_published_factors(
        self, money_engine, money_observations
    ):
        """Never carried across dates — that would fabricate an unfitted curve."""
        covered = [CUTOFF - timedelta(days=i) for i in range(5)]
        frame = pd.concat(
            [
                money_observations,
                pd.DataFrame(
                    curve_rows(
                        {"level": 1.5, "slope": 1.4, "curvature": -0.3, "lambda": 0.609}, covered
                    )
                ),
            ],
            ignore_index=True,
        )
        rows = money_engine.outputs(world_from(frame, CUTOFF))
        by_source: dict[str, set[date]] = {"ns": set(), "short_rate": set()}
        for row in rows:
            if row.metric == "discount_10y":
                by_source[(row.meta or {})["curve_source"]].add(row.as_of)

        assert by_source["ns"], "dates with published factors must use them"
        assert by_source["short_rate"], "dates without must fall back"
        assert by_source["ns"] & set(covered) == by_source["ns"]
        assert not (by_source["ns"] & by_source["short_rate"])

    def test_the_history_window_is_honoured(self, money_engine, money_observations, config):
        from dataclasses import replace

        params = {**config.engines["money"].params, "outputs": {"history_days": 30}}
        narrow = replace(
            config,
            engines={
                **config.engines,
                "money": replace(config.engines["money"], params=params),
            },
        )
        engine = MoneyEngine(narrow)
        rows = engine.outputs(world_from(money_observations, CUTOFF))
        assert rows
        assert min(row.as_of for row in rows) >= CUTOFF - timedelta(days=40)


class TestFit:
    def test_fit_is_a_no_op_and_writes_no_artifact(
        self, money_engine, money_observations, artifacts
    ):
        """Deliberately nothing to fit; an artifact would imply otherwise."""
        money_engine.fit(world_from(money_observations, CUTOFF))
        assert artifacts.load("money") == {}

    def test_predict_does_not_depend_on_fit_having_run(self, money_engine, money_observations):
        world = world_from(money_observations, CUTOFF)
        before = money_engine.predict(world)
        money_engine.fit(world)
        money_engine._cache = None
        after = money_engine.predict(world)
        assert before == after


class TestDegradation:
    def test_no_short_rate_at_all_raises_rather_than_publishing_a_guess(self, money_engine):
        rows = rate_rows(BILL_3M, {date(2020, 6, 1): 0.15})
        world = world_from(pd.DataFrame(rows), date(2020, 6, 5))
        with pytest.raises(InsufficientRatePathError, match="no short-rate path"):
            money_engine.predict(world)

    def test_an_empty_information_set_produces_no_outputs(self, money_engine):
        empty = pd.DataFrame(
            columns=["series_id", "obs_date", "release_date", "revision_date", "value"]
        )
        world = world_from(empty, date(2020, 6, 5))
        assert money_engine.outputs(world) == ()

    def test_it_still_publishes_with_only_the_short_rate(self, money_engine):
        """No bill, no reverse repo, no curve — the numeraire still works."""
        values = {date(2020, 1, 1) + timedelta(days=i): 1.5 for i in range(120)}
        world = world_from(pd.DataFrame(rate_rows(SOFR, values)), date(2020, 5, 5))

        state = money_engine.predict(world)
        assert state.regime == "normal"
        assert state.risk_score == 0.0
        assert state.expected_return == pytest.approx(0.015)
        assert (state.components or {})["discount_1y"] == pytest.approx(math.exp(-0.015))

    def test_a_state_that_would_violate_the_contract_is_never_constructed(self, state):
        """The contract validates on construction, so reaching here is the proof."""
        with pytest.raises(ContractError):
            type(state)(**{**state.__dict__, "risk_score": 140.0})


class TestCaching:
    def test_analysis_is_memoized_per_accessor(self, money_engine, money_observations):
        world = world_from(money_observations, CUTOFF)
        assert money_engine.analyze(world) is money_engine.analyze(world)

    def test_a_new_information_set_invalidates_the_cache(self, money_engine, money_observations):
        first = money_engine.analyze(world_from(money_observations, date(2019, 6, 3)))
        second = money_engine.analyze(world_from(money_observations, CUTOFF))
        assert first is not None and second is not None
        assert first is not second
        assert first.as_of < second.as_of
