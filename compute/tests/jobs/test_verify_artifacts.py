"""The refit's self-check — it replaced a step that verified nothing.

`Publish fitted parameters` uploaded `compute/artifacts/`, which a production
refit never writes: with both admin secrets set the store is R2. The step found
an empty directory every month, warned into the void and passed. These tests
pin the replacement's judgement, because a verifier that cries wolf gets muted
and one that waves everything through is the step it replaced.
"""

from __future__ import annotations

from jobs.verify_artifacts import check

GOOD = {"model_version": "equity-1.2.0+cal.x", "as_of": "2026-07-31", "series": {}}


def test_a_usable_artifact_reports_nothing() -> None:
    assert check("equity", GOOD, as_of=None) == []
    assert check("equity", GOOD, as_of="2026-07-31") == []


def test_an_engine_that_fits_nothing_is_not_a_failure() -> None:
    """`money` has no parameters — the numeraire is arithmetic on observed rates.

    Failing it would make the step cry wolf every month, which is how a check
    stops being read.
    """
    assert check("money", {}, as_of="2026-07-31") == []


def test_either_fit_date_spelling_satisfies_the_address() -> None:
    """equity writes `as_of`; gold and rates write `fitted_as_of`.

    Requiring one spelling would fail three engines for a naming difference,
    which is the false alarm this test exists to prevent.
    """
    gold = {"model_version": "gold-1.0.0", "fitted_as_of": "2026-07-31"}
    assert check("gold", gold, as_of="2026-07-31") == []


def test_a_wall_clock_in_the_stored_bytes_is_a_failure() -> None:
    """The root of issue #6: `fitted_at` made every document differ from every
    other by construction, so an idempotent re-write became a 409."""
    problems = check("equity", {**GOOD, "fitted_at": "2026-08-08T01:22:11Z"}, as_of=None)
    assert len(problems) == 1
    assert "wall-clock" in problems[0]
    assert "issue #6" in problems[0]


def test_a_fit_date_that_is_not_the_run_s_cutoff_is_a_failure() -> None:
    """The artifact is addressed by (model_version, fit date). A mismatch means
    it is filed under a date it does not describe, and the next refit of the
    month will not collide with it the way it should."""
    problems = check("equity", GOOD, as_of="2026-06-30")
    assert len(problems) == 1
    assert "not the run's cutoff" in problems[0]


def test_a_document_with_no_address_or_version_is_a_failure() -> None:
    assert len(check("equity", {"series": {}}, as_of=None)) == 2
