"""Source-level bans on lookahead-shaped code under ``findynamics/research/``.

The guard ``tests/engines/equity/test_no_lookahead_guards.py`` runs over the
equity engine, pointed at the research layer, with three additional patterns
that only this layer can reach for — and with one repair, described below,
without which six of the eight patterns cannot match anything at all.

Duplicated rather than parameterized over two roots, deliberately: the two lists
are allowed to diverge — the scaler and PCA bans have no meaning for an engine
that fits neither — and a shared fixture would make adding a research-only ban
feel like editing an engine's contract.

The repair: reconstruct columns, not token order
------------------------------------------------

The engine guard's ``code_lines`` joins the tokens of a line with
``f"{line} {token.string}"``, which inserts a space between **every** pair of
tokens. ``result.smooth(params)`` comes back as ``result . smooth ( params )``,
and the pattern ``\\.smooth\\s*\\(`` — which requires ``.`` to be adjacent to
``smooth`` — cannot match it. Measured over that module's own five patterns,
only ``savgol|savitzky|Savitzky`` survives the transformation, because it is the
one pattern that is a bare identifier rather than an expression shape. Its
self-test exercises exactly that pattern, which is why the gap is not visible
from the test file.

So this version places each token at its **real start column** and leaves the
gaps where comments and strings were removed. The reconstruction then reads like
the source it came from, every pattern below matches the shape it names, and
:func:`test_every_banned_pattern_detects_its_own_violation` proves it for all
eight rather than for one.

This is a repair to *this* file only. The equity guard is not touched here: the
same repair applied there makes two legitimate call sites fail
(``engines/equity/backtest.py::false_alarm_rate`` and
``regime/calibrate.py::transition_labels``, both of which shift backwards to
build a **label** and must), and deciding how to express that exemption is an
engine-side change with its own review, not a line item in a research recon.

The three additions, and what each one protects
-----------------------------------------------

``StandardScaler().fit_transform`` / ``PCA(...).fit_transform``
    The whole anti-lookahead protocol of this track is *fit on the expanding
    window, transform forward only* (``docs/research/kk-omega-design.md`` §6). A
    combined ``fit_transform`` is the one call that makes that separation
    impossible to express, and it is also the call every scikit-learn tutorial
    opens with. Fitting a scaler or a projection on the whole sample puts every
    future observation into every historical Ω, and the result looks entirely
    plausible — just slightly too good, everywhere.

``.expanding(...).apply(...)``
    Causal in principle and trivially non-causal in practice, because the
    applied function may close over anything at all. ``expanding_z`` in
    ``features/kinematics.py`` shows the safe shape: ``.expanding().mean()`` and
    ``.expanding().std()``, which cannot see past their own window. This is the
    broadest of the three and the most likely to need narrowing; if it ever
    fires on legitimate causal code, narrow the **pattern** and say so in the
    phase report. Deleting it is not the fix.

The pressure point, decided in KK0 and reached in KK3
-----------------------------------------------------

KK3 grades arms against forward returns ``r_{t+h}`` and forward realized
volatility ``σ_{t+h}``. Building a **target** by looking forward is legitimate
and unavoidable — you cannot grade a call without knowing what happened next —
and the obvious way to write it is ``.shift(-h)``, which the fifth pattern bans.

The resolution was fixed in KK0, before there was a failing test creating
pressure to reach for the easy fix, and KK3 used it as written. ``§14.1`` rule 3
bans centred and forward-looking constructions **in the feature path**; the
label path is a different path. So label construction is confined to modules
named in :data:`TARGET_MODULES` — one entry, ``omega/targets.py``, added with a
comment naming the four functions that need it. That is a visible, reviewable
act. Loosening a pattern so that a target stops tripping it is not: it would
exempt the feature path at the same time, silently, and the guard would go on
passing.

The exemption is **per pattern, not per file**, and only
:data:`TARGET_EXEMPT` is exemptable. A module that builds labels has a reason
to shift backwards; it has no reason at all to fit a scaler on the whole sample
or to reach for the RTS smoother, and a file-level exemption would hand it
those for free. That is the same mistake as loosening the pattern, made one
directory further down.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

RESEARCH_ROOT = Path(__file__).resolve().parents[3] / "findynamics" / "research"

#: Modules permitted to construct forward-looking **labels**, as paths relative
#: to :data:`RESEARCH_ROOT`. A phase that adds one names the module *and the
#: functions*, here, in a comment.
#:
#: ``omega/targets.py`` (KK3) builds the four things the walk-forward grades
#: against: ``forward_return``, ``forward_volatility``, ``forward_drawdown`` and
#: ``transition_within``. Each looks forward by construction — grading a call
#: without knowing what happened next is not possible — and each is confined to
#: that file so the exemption stays one line wide.
TARGET_MODULES: frozenset[str] = frozenset({"omega/targets.py"})

#: The only patterns :data:`TARGET_MODULES` may trip. Building a label requires
#: looking forward; it does not require any of the other seven, so they stay
#: enforced everywhere without exception.
TARGET_EXEMPT: frozenset[str] = frozenset({r"\.shift\(\s*-\d"})

#: Patterns that must not appear anywhere under findynamics/research.
BANNED: dict[str, str] = {
    r"savgol|savitzky|Savitzky": (
        "Savitzky-Golay uses a centred window and is display-layer only (§8.1)"
    ),
    r"\.smooth\s*\(": "the RTS smoother conditions on the whole sample (§8.1)",
    r"smoothed_state|smoothed_forecasts": "smoothed state estimates read the future",
    r"\.rolling\([^)]*center\s*=\s*True": "a centred rolling window reads the future",
    r"\.shift\(\s*-\d": "a negative shift pulls a future value backwards",
    r"StandardScaler\(\)\.fit_transform": "fitting a scaler on the whole sample leaks the future",
    r"PCA\([^)]*\)\.fit_transform": "fitting PCA on the whole sample leaks the future",
    r"\.expanding\([^)]*\)\.apply": "expanding().apply is easy to make non-causal",
}

#: One planted violation per banned pattern, so that every regex in
#: :data:`BANNED` is known to match the shape it describes. Keys are the
#: pattern; values are the line of code that must trip it.
PLANTED: dict[str, str] = {
    r"savgol|savitzky|Savitzky": "smoothed = savgol_filter(omega, 21, 3)",
    r"\.smooth\s*\(": "state = model.smooth(params)",
    r"smoothed_state|smoothed_forecasts": "level = result.smoothed_state[0]",
    r"\.rolling\([^)]*center\s*=\s*True": "baseline = omega.rolling(window=63, center=True).mean()",
    r"\.shift\(\s*-\d": "target = returns.shift(-21)",
    r"StandardScaler\(\)\.fit_transform": "z = StandardScaler().fit_transform(frame)",
    r"PCA\([^)]*\)\.fit_transform": 'omega = PCA(n_components=1, svd_solver="full").fit_transform(z)',
    r"\.expanding\([^)]*\)\.apply": "scored = series.expanding(min_periods=60).apply(leaky)",
}


def source_files() -> list[Path]:
    """Every module under the research root. Nothing is excluded here.

    The :data:`TARGET_MODULES` exemption is applied per pattern in
    :func:`test_no_lookahead_shaped_call_appears_under_research`, not by
    dropping files from this list — a file dropped here would be exempt from
    all eight patterns at once.
    """
    return sorted(p for p in RESEARCH_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def code_lines(path: Path) -> dict[int, str]:
    """Line number -> executable code on it, with comments and strings removed.

    Tokens are written back at their original start column rather than joined
    with spaces, so the reconstruction preserves the shape the patterns are
    written against — see this module's docstring for what the joined form
    silently does to six of the eight.
    """
    return code_lines_of(path.read_text())


def code_lines_of(source: str) -> dict[int, str]:
    lines: dict[int, str] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE):
            continue
        row, column = token.start
        line = lines.get(row, "")
        if len(line) < column:
            line += " " * (column - len(line))
        lines[row] = line[:column] + token.string
    return lines


def test_the_guard_has_source_to_check():
    """A guard that silently passes on an empty directory guards nothing."""
    assert len(source_files()) >= 5


def test_comments_and_docstrings_are_stripped_before_matching():
    """These modules explain the bans at length; the prose must not trip them.

    The whole reason for tokenizing rather than grepping: a naive scan flags the
    documentation that exists to prevent the very thing it is looking for, and
    the obvious workaround — loosening the pattern — is how the guard stops
    catching real code.
    """
    lines = code_lines_of(
        '"""A docstring naming savgol and .smooth( must not count."""\n'
        "# neither must a comment naming StandardScaler().fit_transform\n"
        "value = savgol_filter(y)\n"
    )
    hits = sorted(n for n, line in lines.items() if "savgol" in line)
    assert hits == [3]


@pytest.mark.parametrize("pattern", sorted(BANNED))
def test_every_banned_pattern_detects_its_own_violation(pattern: str):
    """A regex that has never matched is a regex that might not match.

    The engine guard's self-test covers one of its five patterns. This covers
    all eight, which is how the token-joining defect described in the module
    docstring becomes visible instead of staying a silent pass.
    """
    planted = PLANTED[pattern]
    lines = code_lines_of(f"{planted}\n")
    assert re.compile(pattern).search(lines[1]), (
        f"{pattern!r} did not match its own planted violation {planted!r}; "
        f"the tokenizer reconstructed it as {lines[1]!r}"
    )


@pytest.mark.parametrize("pattern,why", sorted(BANNED.items()))
def test_no_lookahead_shaped_call_appears_under_research(pattern: str, why: str):
    exemptable = pattern in TARGET_EXEMPT
    offenders = []
    compiled = re.compile(pattern)
    for path in source_files():
        relative = str(path.relative_to(RESEARCH_ROOT))
        if exemptable and relative in TARGET_MODULES:
            continue
        offenders += [
            f"{relative}:{n}" for n, line in code_lines(path).items() if compiled.search(line)
        ]
    assert not offenders, f"{why} — found at {offenders}"


def test_the_target_exemption_cannot_be_widened_by_accident():
    """Only the label-shaped pattern is exemptable, and it must be a real ban.

    Two ways the exemption could quietly become a hole: a phase adds a pattern
    to :data:`TARGET_EXEMPT` that has nothing to do with labels, or it names one
    that is not in :data:`BANNED` at all — in which case the entry reads like an
    exemption and enforces nothing.
    """
    unenforced = sorted(TARGET_EXEMPT - set(BANNED))
    assert not unenforced, f"exemptions for patterns that are not banned: {unenforced}"
    assert sorted(TARGET_EXEMPT) == [r"\.shift\(\s*-\d"]
