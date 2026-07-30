"""Liquidity regimes, asserted against the funding events that actually happened.

Thresholds are the one thing a synthetic fixture cannot validate: you can always
invent a series that crosses whichever line you wrote. So the classifier is run
over the real snapshot and asked about three dates the record already has an
opinion on — September 2019, March 2020, and September 2024, which looked similar
on the headline spread and was not a crisis.

The negative control is the load-bearing test here. A rule that catches both
crises and also flags every easing cycle has not identified anything.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.money import liquidity as liquidity_mod
from findynamics.engines.money.domain import MONEY_REGIMES, liquidity_code
from tests.engines.money.conftest import BILL_3M, RRP, SOFR


@pytest.fixture
def rules(config) -> liquidity_mod.LiquidityRules:
    """The shipped thresholds, not test-local ones — that is the point."""
    return liquidity_mod.LiquidityRules.from_params(config.engines["money"].params)


def history(
    observations: pd.DataFrame, as_of: date, rules: liquidity_mod.LiquidityRules
) -> pd.Series:
    """Every date classified, as of one information-set cutoff."""
    accessor = PandasPITAccessor(observations, as_of)
    frame = liquidity_mod.build_inputs(
        accessor.wide([SOFR, BILL_3M, RRP]),
        rules,
        bill_id=BILL_3M,
        overnight_id=SOFR,
        rrp_id=RRP,
    )
    return liquidity_mod.classify_history(frame, rules)


def between(labels: pd.Series, start: str, end: str) -> list[str]:
    window = labels.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    return list(window)


class TestSeptember2019RepoStress:
    """17 September 2019: SOFR printed 5.25% against a 1.99% bill."""

    def test_the_blowup_reads_stressed(self, money_observations, rules):
        labels = history(money_observations, date(2019, 9, 30), rules)
        assert labels.loc[pd.Timestamp("2019-09-17")] == "stressed"
        assert labels.loc[pd.Timestamp("2019-09-18")] == "stressed"

    def test_the_single_session_spike_alone_is_enough(self, money_observations, rules):
        """A -3.26 spread needs no confirmation from anything else."""
        accessor = PandasPITAccessor(money_observations, date(2019, 9, 30))
        frame = liquidity_mod.build_inputs(
            accessor.wide([SOFR, BILL_3M, RRP]),
            rules,
            bill_id=BILL_3M,
            overnight_id=SOFR,
            rrp_id=RRP,
        )
        key = pd.Timestamp("2019-09-17")
        inputs = liquidity_mod.inputs_on(frame.loc[key], key)

        assert inputs.spread is not None
        assert inputs.spread <= rules.stressed_spike_pp
        assert liquidity_mod.is_stressed(inputs, rules)

    def test_the_days_before_the_blowup_are_not_yet_stressed(self, money_observations, rules):
        """The stress began on the 17th. A rule that calls the 16th has hindsight."""
        labels = history(money_observations, date(2019, 9, 30), rules)
        assert labels.loc[pd.Timestamp("2019-09-16")] != "stressed"
        assert labels.loc[pd.Timestamp("2019-09-13")] != "stressed"


class TestMarch2020DashForCash:
    """Bills collapsed 1.2pp in ten sessions while the overnight rate barely moved."""

    def test_the_dislocation_reads_stressed(self, money_observations, rules):
        labels = history(money_observations, date(2020, 3, 31), rules)
        window = between(labels, "2020-03-04", "2020-03-16")
        assert window, "the fixture must cover March 2020"
        assert all(label == "stressed" for label in window), window

    def test_it_is_caught_by_the_collapsing_bill_leg_not_by_a_spike(
        self, money_observations, rules
    ):
        """The mechanism is the opposite of 2019's and must be detected as such.

        No single session breaches the spike threshold; the overnight rate is
        *falling*, not climbing. What identifies it is bills being scrambled for.
        """
        accessor = PandasPITAccessor(money_observations, date(2020, 3, 31))
        frame = liquidity_mod.build_inputs(
            accessor.wide([SOFR, BILL_3M, RRP]),
            rules,
            bill_id=BILL_3M,
            overnight_id=SOFR,
            rrp_id=RRP,
        )
        key = pd.Timestamp("2020-03-12")
        inputs = liquidity_mod.inputs_on(frame.loc[key], key)

        assert inputs.spread is not None and inputs.spread > rules.stressed_spike_pp
        assert inputs.overnight_change is not None and inputs.overnight_change < 0.0
        assert inputs.bill_change is not None
        assert inputs.bill_change <= rules.stressed_bill_drop_pp
        assert liquidity_mod.is_stressed(inputs, rules)

    def test_the_stress_clears_once_the_market_is_repaired(self, money_observations, rules):
        """Same crisis, and by end-March the funding market is working again.

        Bills and repo are back within a basis point of each other. If this still
        read `stressed` the engine would be describing the news rather than the
        money market — the recession had barely started.

        Note what is *not* asserted: `abundant`. Reverse-repo take-up touched
        $285bn on 31 March, but only for a day — the five-session average peaked
        near $194bn and fell back. A one-day quarter-end blip is not a regime, and
        the shipped $250bn threshold correctly declines to call it one. The
        genuinely abundant era is 2021-2023, covered below.
        """
        labels = history(money_observations, date(2020, 4, 30), rules)
        late_march = between(labels, "2020-03-24", "2020-03-31")
        assert late_march
        assert "stressed" not in late_march, late_march


class TestSeptember2024NegativeControl:
    """The test that gives the other two their meaning."""

    def test_an_ordinary_easing_cycle_is_not_stressed(self, money_observations, rules):
        """The bill ran 0.49pp below SOFR into the September 2024 cut.

        On the headline spread that is two-thirds of the way to March 2020. It was
        not a crisis, and the classifier must not say it was — on any session.
        """
        labels = history(money_observations, date(2024, 10, 31), rules)
        window = between(labels, "2024-08-01", "2024-10-31")
        assert window
        assert "stressed" not in window, [
            (str(k.date()), v) for k, v in labels.items() if v == "stressed"
        ]

    def test_it_reads_tightening_instead(self, money_observations, rules):
        """Which is the correct answer: QT was on and reserves were draining."""
        labels = history(money_observations, date(2024, 10, 31), rules)
        assert labels.loc[pd.Timestamp("2024-09-18")] == "tightening"

    def test_both_confirming_legs_fail_independently(self, money_observations, rules):
        """Not a near miss on one threshold — a miss on both, separately."""
        accessor = PandasPITAccessor(money_observations, date(2024, 10, 31))
        frame = liquidity_mod.build_inputs(
            accessor.wide([SOFR, BILL_3M, RRP]),
            rules,
            bill_id=BILL_3M,
            overnight_id=SOFR,
            rrp_id=RRP,
        )
        key = pd.Timestamp("2024-09-18")
        inputs = liquidity_mod.inputs_on(frame.loc[key], key)

        assert inputs.bill_change is not None
        assert inputs.bill_change > rules.stressed_bill_drop_pp
        assert inputs.overnight_change is not None
        assert inputs.overnight_change < rules.stressed_sofr_jump_pp


class TestStressIsRareOverTheWholeSnapshot:
    """A classifier that fires often is a thermometer, not a stress detector."""

    def test_stress_is_confined_to_the_two_known_episodes(self, money_observations, rules):
        labels = history(money_observations, date(2024, 10, 31), rules)
        stressed = [pd.Timestamp(k).date() for k, v in labels.items() if v == "stressed"]

        assert stressed, "the snapshot contains two real stress episodes"
        for day in stressed:
            in_2019 = date(2019, 9, 16) <= day <= date(2019, 9, 20)
            in_2020 = date(2020, 3, 2) <= day <= date(2020, 3, 20)
            assert in_2019 or in_2020, f"unexpected stress call on {day}"

    def test_it_is_a_small_share_of_all_sessions(self, money_observations, rules):
        labels = history(money_observations, date(2024, 10, 31), rules)
        share = sum(1 for v in labels if v == "stressed") / len(labels)
        assert 0.0 < share < 0.02, share

    def test_the_2018_turn_of_year_squeeze_is_a_documented_near_miss(
        self, money_observations, rules
    ):
        """Recorded because it sits three basis points from the threshold.

        31 Dec 2018 / 2 Jan 2019 was a genuine year-end funding squeeze. It reads
        `tightening`, not `stressed`, and a future recalibration should know that
        widening `stressed_spread_pp` by 0.05 flips it. Asserted so the margin is
        visible in the suite rather than only in a comment.
        """
        labels = history(money_observations, date(2019, 1, 31), rules)
        assert labels.loc[pd.Timestamp("2019-01-02")] == "tightening"

        accessor = PandasPITAccessor(money_observations, date(2019, 1, 31))
        frame = liquidity_mod.build_inputs(
            accessor.wide([SOFR, BILL_3M, RRP]),
            rules,
            bill_id=BILL_3M,
            overnight_id=SOFR,
            rrp_id=RRP,
        )
        key = pd.Timestamp("2019-01-03")
        inputs = liquidity_mod.inputs_on(frame.loc[key], key)
        assert inputs.spread_mean is not None
        assert (
            rules.stressed_spread_pp - 0.05 < inputs.spread_mean <= rules.stressed_spread_pp + 0.05
        )


class TestAbundantAndTightening:
    def test_the_2021_style_buffer_reads_abundant(self, rules):
        frame = pd.DataFrame(
            {
                SOFR: [0.05] * 80,
                BILL_3M: [0.06] * 80,
                RRP: [1800.0] * 80,
            },
            index=pd.date_range("2021-06-01", periods=80, freq="D"),
        )
        labels = liquidity_mod.classify_history(
            liquidity_mod.build_inputs(
                frame, rules, bill_id=BILL_3M, overnight_id=SOFR, rrp_id=RRP
            ),
            rules,
        )
        assert labels.iloc[-1] == "abundant"

    def test_a_draining_buffer_reads_tightening(self, rules):
        """RRP halving off its own trailing average, with a benign spread."""
        rrp = [2000.0] * 63 + [900.0] * 10
        frame = pd.DataFrame(
            {SOFR: [5.3] * 73, BILL_3M: [5.35] * 73, RRP: rrp},
            index=pd.date_range("2023-06-01", periods=73, freq="D"),
        )
        labels = liquidity_mod.classify_history(
            liquidity_mod.build_inputs(
                frame, rules, bill_id=BILL_3M, overnight_id=SOFR, rrp_id=RRP
            ),
            rules,
        )
        assert labels.iloc[-1] == "tightening"

    def test_a_tiny_rrp_balance_carries_no_signal(self, rules):
        """2019 ran at $3bn, where a 60% swing is one counterparty."""
        rrp = [3.0] * 63 + [1.0] * 10
        frame = pd.DataFrame(
            {SOFR: [2.4] * 73, BILL_3M: [2.42] * 73, RRP: rrp},
            index=pd.date_range("2019-04-01", periods=73, freq="D"),
        )
        labels = liquidity_mod.classify_history(
            liquidity_mod.build_inputs(
                frame, rules, bill_id=BILL_3M, overnight_id=SOFR, rrp_id=RRP
            ),
            rules,
        )
        assert labels.iloc[-1] == "normal"

    def test_an_unremarkable_market_reads_normal(self, rules):
        frame = pd.DataFrame(
            {SOFR: [2.40] * 70, BILL_3M: [2.42] * 70, RRP: [5.0] * 70},
            index=pd.date_range("2018-06-01", periods=70, freq="D"),
        )
        labels = liquidity_mod.classify_history(
            liquidity_mod.build_inputs(
                frame, rules, bill_id=BILL_3M, overnight_id=SOFR, rrp_id=RRP
            ),
            rules,
        )
        assert set(labels) == {"normal"}


class TestDegradation:
    def test_without_sofr_the_spread_cannot_be_formed(self, money_observations, rules):
        """Pre-2018 the state rests on reverse repo alone; it must not guess."""
        labels = history(money_observations, date(2018, 3, 1), rules)
        assert set(labels) <= set(MONEY_REGIMES)
        assert "stressed" not in set(labels)

        accessor = PandasPITAccessor(money_observations, date(2018, 3, 1))
        frame = liquidity_mod.build_inputs(
            accessor.wide([SOFR, BILL_3M, RRP]),
            rules,
            bill_id=BILL_3M,
            overnight_id=SOFR,
            rrp_id=RRP,
        )
        inputs = liquidity_mod.inputs_on(frame.iloc[-1], frame.index[-1])
        assert not inputs.has_spread

    def test_an_empty_frame_classifies_nothing(self, rules):
        assert liquidity_mod.build_inputs(
            pd.DataFrame(), rules, bill_id="a", overnight_id="b", rrp_id="c"
        ).empty
        assert liquidity_mod.classify_history(pd.DataFrame(), rules).empty

    def test_a_read_with_no_inputs_at_all_is_normal(self, rules):
        inputs = liquidity_mod.LiquidityInputs(
            as_of=pd.Timestamp("2026-07-29"),
            spread=None,
            spread_mean=None,
            bill_change=None,
            overnight_change=None,
            rrp_level=None,
            rrp_ratio=None,
        )
        assert liquidity_mod.classify(inputs, rules) == "normal"


class TestRulesAndVocabulary:
    def test_the_shipped_config_parses(self, config):
        rules = liquidity_mod.LiquidityRules.from_params(config.engines["money"].params)
        assert rules.stressed_spread_pp < rules.tightening_spread_pp < 0.0
        assert rules.abundant_rrp_bn >= rules.rrp_material_bn

    def test_an_unknown_threshold_is_rejected_rather_than_ignored(self):
        """A typo in yaml must not silently leave the default in place."""
        with pytest.raises(ValueError, match="unknown threshold"):
            liquidity_mod.LiquidityRules.from_params({"liquidity": {"stresed_spread_pp": -1.0}})

    def test_a_non_mapping_liquidity_block_is_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            liquidity_mod.LiquidityRules.from_params({"liquidity": [1, 2]})

    def test_every_state_has_a_code_and_codes_are_ordered_by_tightness(self):
        codes = [liquidity_code(state) for state in MONEY_REGIMES]
        assert codes == sorted(codes)
        assert liquidity_code("abundant") < liquidity_code("stressed")

    def test_an_unknown_state_has_no_code(self):
        with pytest.raises(ValueError, match="unknown liquidity state"):
            liquidity_code("plentiful")

    def test_classification_is_always_in_the_vocabulary(self, money_observations, rules):
        labels = history(money_observations, date(2024, 10, 31), rules)
        assert set(labels) <= set(MONEY_REGIMES)


class TestCausality:
    """Every transform is trailing, so a date's label cannot move later."""

    def test_a_label_does_not_change_when_more_future_data_arrives(self, money_observations, rules):
        """The regression test for a centred window sneaking in."""
        early = history(money_observations, date(2019, 10, 15), rules)
        late = history(money_observations, date(2024, 10, 31), rules)

        shared = early.index.intersection(late.index)
        assert len(shared) > 300
        mismatches = [
            (str(pd.Timestamp(k).date()), early.loc[k], late.loc[k])
            for k in shared
            if early.loc[k] != late.loc[k]
        ]
        assert not mismatches, mismatches
