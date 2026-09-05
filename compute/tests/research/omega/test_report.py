"""The report's language rules, its structure, and its byte-identity.

Two of these enforce things that would otherwise survive on good intentions.

**The language rules.** The track forbids a specific list of phrases anywhere in
the committed report. A rule enforced only by review is a rule that survives
exactly one busy afternoon, so it is a test — and it greps the *committed file*,
not a string built in the test, because the document is the artifact.

**The quoted verdict rule.** §G must reproduce §9 of the design note verbatim.
The renderer reads it out of that file, so this test checks the extraction still
finds it and that the text in the report matches the note character for
character. If someone edits §9 after seeing the data, the report changes with it
and the diff is visible in one commit rather than hidden in prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from findynamics.research.omega.report import (
    DEFAULT_ARTIFACTS,
    DEFAULT_REPORT,
    OmegaReportError,
    load_artifacts,
    render_report,
    sub_period_hits,
    verdict_from,
    verdict_rules,
)

#: Phrases the track forbids anywhere in the document. Each asserts more than
#: the evidence can carry: a physical claim, a proof, or a cause.
#:
#: Matched on **word boundaries**, not as substrings, and that narrowing is a
#: real finding rather than a convenience: the first version banned the substring
#: ``proves`` and fired on "improves out-of-sample R²", which is the plainest
#: possible statement of the actual result. A guard that flags the sentence you
#: are required to write is a guard that gets deleted. The KK0 guard's docstring
#: gives the rule — narrow the pattern, say so, never delete it.
FORBIDDEN: tuple[tuple[str, str], ...] = (
    (r"markets have a fifth dimension", "asserts a physical reality the track does not claim"),
    (r"fifth dimension of markets", "the same assertion, reordered"),
    (r"\bproves\b", "a walk-forward on one index does not prove anything"),
    (r"\bproven\b", "same"),
    (r"\bproof\b", "same"),
    (r"confirms the hypothesis", "a pre-registered test is not confirmation"),
    (r"the market's true state", "Ω is an inferred coordinate, not a hidden truth"),
    (r"Ω drives", "a causal verb applied to a correlation"),
    (r"\bomega drives\b", "same"),
    (r"Ω causes", "same"),
    (r"driven by Ω", "same"),
)

#: The framing the track requires, once, in this form.
REQUIRED_FRAMING = (
    "A fifth-dimensional latent-state representation inspired by Kaluza-Klein geometry\n"
    "was tested as a quantitative modelling hypothesis."
)

SECTIONS: tuple[str, ...] = (
    "## A. Mathematical formulation",
    "## B. Data",
    "## C. Model comparison",
    "## D. Performance",
    "## E. Predictive statistics",
    "## F. Failure analysis",
    "## G. Verdict",
    "## H. Reproduction",
)


@pytest.fixture(scope="module")
def report_text() -> str:
    if not DEFAULT_REPORT.exists():
        pytest.skip("kk-omega-report.md not generated in this working tree")
    return DEFAULT_REPORT.read_text()


@pytest.mark.parametrize("pattern,why", FORBIDDEN)
def test_the_report_avoids_the_forbidden_phrases(report_text: str, pattern: str, why: str):
    hit = re.search(pattern, report_text, re.I)
    assert hit is None, f"{pattern!r}: {why} — found {hit.group(0)!r}"


@pytest.mark.parametrize("pattern,why", FORBIDDEN)
def test_every_forbidden_pattern_matches_its_own_planted_violation(pattern: str, why: str):
    """A regex that has never matched is a regex that might not match.

    The word-boundary narrowing above is exactly the kind of edit that can turn
    a live ban into a dead one, so each pattern is checked against a sentence it
    must catch.
    """
    planted = {
        r"markets have a fifth dimension": "We found that markets have a fifth dimension.",
        r"fifth dimension of markets": "The fifth dimension of markets is now measurable.",
        r"\bproves\b": "This proves the model works.",
        r"\bproven\b": "The hypothesis is proven.",
        r"\bproof\b": "This is proof of the mechanism.",
        r"confirms the hypothesis": "The result confirms the hypothesis.",
        r"the market's true state": "Ω recovers the market's true state.",
        r"Ω drives": "Ω drives realized volatility.",
        r"\bomega drives\b": "Omega drives volatility.",
        r"Ω causes": "Ω causes the drawdown.",
        r"driven by Ω": "Volatility is driven by Ω.",
    }[pattern]
    assert re.search(pattern, planted, re.I) is not None


def test_the_narrowed_ban_still_allows_the_sentence_the_result_requires():
    """ "improves out-of-sample R²" is the finding. It must not trip the guard."""
    legitimate = "Arm B improves out-of-sample R² over A′ at three horizons."
    for pattern, _ in FORBIDDEN:
        assert re.search(pattern, legitimate, re.I) is None


def test_the_report_carries_the_required_framing(report_text: str):
    assert REQUIRED_FRAMING in report_text


def test_the_report_has_every_section_in_order(report_text: str):
    for heading in SECTIONS:
        assert heading in report_text
    positions = [report_text.index(h) for h in SECTIONS]
    assert positions == sorted(positions)


def test_the_verdict_rule_is_quoted_verbatim_from_the_design_note(report_text: str):
    """The KK5 acceptance check, as a test rather than as a manual diff."""
    rules = verdict_rules()
    quoted = "> " + rules.replace("\n", "\n> ")
    assert quoted in report_text, "§G does not quote the design note's §9 verbatim"

    for marker in ("PARTIALLY SUPPORTED", "NO INCREMENTAL INFORMATION", "UNSTABLE", "REJECTED"):
        assert marker in rules
        assert marker in report_text


def test_the_report_states_a_verdict_from_the_pre_registered_vocabulary(report_text: str):
    allowed = {
        "SUPPORTED",
        "PARTIALLY SUPPORTED",
        "NO INCREMENTAL INFORMATION",
        "UNSTABLE",
        "REJECTED",
        "NO VERDICT REACHED",
    }
    match = re.search(r"^### `([A-Z ]+)`$", report_text, re.M)
    assert match is not None, "§G has no verdict heading"
    assert match.group(1) in allowed


def test_the_report_says_its_numbers_are_out_of_sample(report_text: str):
    assert "out-of-sample" in report_text
    assert "No causal relationship is" in report_text


def test_regenerating_the_report_is_byte_identical():
    """The property that makes ``--report`` safe to run twice."""
    artifacts = load_artifacts(DEFAULT_ARTIFACTS)
    first = render_report(artifacts, config_hash="x", model_version="y")
    second = render_report(artifacts, config_hash="x", model_version="y")
    assert first == second


def test_a_missing_artifact_is_named_rather_than_skipped(tmp_path: Path):
    with pytest.raises(OmegaReportError, match="missing artifact"):
        load_artifacts(tmp_path)


def test_the_design_note_must_actually_contain_the_section(tmp_path: Path):
    note = tmp_path / "design.md"
    note.write_text("# nothing here\n")
    with pytest.raises(OmegaReportError, match="Verdict vocabulary"):
        verdict_rules(note)


def test_no_verdict_is_reached_when_exactly_one_of_q2_q3_q4_passes():
    """The gap this run fell into, pinned so a §9 repair has to notice it.

    Q1 fails and one of Q2/Q3/Q4 passes: `SUPPORTED` needs Q1, `PARTIALLY
    SUPPORTED` needs two, `NO INCREMENTAL INFORMATION` needs zero passing
    against A′, `UNSTABLE` needs instability, `REJECTED` needs nothing passing.
    """
    verdict, trail = verdict_from(
        {"Q1": "FAILS", "Q2": "PASSES (arm B)", "Q3": "FAILS", "Q4": "FAILS", "Q5": "FAILS"},
        sub_period_hits=5,
        headline_flips=False,
        max_redundancy=0.76,
        control_shows_skill=True,
        control_cause_found=True,
        beats_a=False,
    )
    assert verdict == "NO VERDICT REACHED"
    assert [fired for _, fired, _ in trail] == [False] * 5


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ({"Q1": "FAILS", "Q2": "FAILS", "Q3": "FAILS", "Q4": "FAILS", "Q5": "FAILS"}, "REJECTED"),
        (
            {"Q1": "PASSES", "Q2": "PASSES", "Q3": "PASSES", "Q4": "FAILS", "Q5": "FAILS"},
            "SUPPORTED",
        ),
        (
            {"Q1": "FAILS", "Q2": "PASSES", "Q3": "PASSES", "Q4": "FAILS", "Q5": "FAILS"},
            "PARTIALLY SUPPORTED",
        ),
    ],
)
def test_the_other_rules_still_fire_when_their_conditions_hold(verdicts, expected):
    """The gap is a gap, not a broken evaluator."""
    verdict, _ = verdict_from(
        verdicts,
        sub_period_hits=5,
        headline_flips=False,
        max_redundancy=0.5,
        control_shows_skill=False,
        control_cause_found=True,
        beats_a=True,
    )
    assert verdict == expected


def test_an_unstable_result_is_not_promoted():
    verdict, _ = verdict_from(
        {"Q1": "FAILS", "Q2": "PASSES", "Q3": "PASSES", "Q4": "FAILS", "Q5": "FAILS"},
        sub_period_hits=1,
        headline_flips=False,
        max_redundancy=0.5,
        control_shows_skill=False,
        control_cause_found=True,
        beats_a=True,
    )
    assert verdict == "UNSTABLE"


def test_a_control_that_shows_unexplained_skill_rejects():
    """The acceptance gate, in the verdict layer as well as in the prose."""
    verdict, _ = verdict_from(
        {"Q1": "PASSES", "Q2": "PASSES", "Q3": "PASSES", "Q4": "PASSES", "Q5": "PASSES"},
        sub_period_hits=5,
        headline_flips=False,
        max_redundancy=0.5,
        control_shows_skill=True,
        control_cause_found=False,
        beats_a=True,
    )
    assert verdict == "REJECTED"


def test_a_latent_quantity_that_is_a_published_one_rejects():
    verdict, _ = verdict_from(
        {"Q1": "PASSES", "Q2": "PASSES", "Q3": "FAILS", "Q4": "FAILS", "Q5": "FAILS"},
        sub_period_hits=5,
        headline_flips=False,
        max_redundancy=0.97,
        control_shows_skill=False,
        control_cause_found=True,
        beats_a=True,
    )
    assert verdict == "REJECTED"


def test_sub_period_hits_counts_periods_not_rows():
    frame = pd.DataFrame(
        {
            "arm": ["B"] * 6,
            "target": ["forward_vol"] * 6,
            "sub_period": ["a", "a", "b", "b", "c", "c"],
            "horizon": [5, 21] * 3,
            "delta_r2_vs_a_prime": [0.1, -0.2, -0.1, -0.2, 0.3, 0.4],
        }
    )
    assert sub_period_hits(frame, "B", "forward_vol") == 2
