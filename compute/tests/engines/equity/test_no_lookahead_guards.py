"""Source-level bans on lookahead-shaped code (FINDYN_V1_SPEC.md §8.1, §14.1).

Two filters read the future and are easy to reach for by accident, because both
are the *obvious* tool for their job and both are one import away:

* **Savitzky-Golay** uses a centred window. §8.1 permits it in the dashboard
  display layer and nowhere else, and asks for this to be enforced by module
  separation rather than by review.
* **The RTS smoother** conditions on the whole sample. ``kalman.py`` calls
  ``filter()`` for that reason, and a future edit to ``smooth()`` would be a
  one-word change with no visible symptom — every number would still look
  plausible, just slightly too good.

Greps rather than import checks, because the failure mode is a call added later
in a file that already imports the right things. A test that reads the source is
the only kind that catches "someone typed `.smooth(`".

Comments and docstrings are stripped before matching, with ``tokenize`` rather
than a "line starts with #" heuristic. These modules explain the bans at length,
so a naive grep flags the documentation that exists to prevent the very thing it
is looking for — and the obvious workaround, loosening the pattern, is how the
guard stops catching real code.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[3] / "findynamics" / "engines" / "equity"

#: Patterns that must not appear anywhere under engines/equity.
BANNED: dict[str, str] = {
    r"savgol|savitzky|Savitzky": (
        "Savitzky-Golay uses a centred window and is display-layer only (§8.1)"
    ),
    r"\.smooth\s*\(": "the RTS smoother conditions on the whole sample (§8.1)",
    r"smoothed_state|smoothed_forecasts": "smoothed state estimates read the future",
    r"\.rolling\([^)]*center\s*=\s*True": "a centred rolling window reads the future",
    r"\.shift\(\s*-\d": "a negative shift pulls a future value backwards",
}


def source_files() -> list[Path]:
    return sorted(p for p in ENGINE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def code_lines(path: Path) -> dict[int, str]:
    """Line number -> executable code on it, with comments and strings removed."""
    lines: dict[int, str] = {}
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    for token in tokens:
        if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE):
            continue
        row = token.start[0]
        lines[row] = f"{lines.get(row, '')} {token.string}"
    return lines


def test_the_guard_has_source_to_check():
    """A guard that silently passes on an empty directory guards nothing."""
    assert len(source_files()) >= 5


def test_the_guard_detects_a_planted_violation(tmp_path):
    """And a guard nobody has seen fail is a guard nobody should trust."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""A docstring mentioning savgol must not count."""\nx = savgol_filter(y)\n'
    )
    hits = [n for n, line in code_lines(planted).items() if "savgol" in line]
    assert hits == [2]


@pytest.mark.parametrize("pattern,why", sorted(BANNED.items()))
def test_no_lookahead_shaped_call_appears_under_engines_equity(pattern: str, why: str):
    compiled = re.compile(pattern)
    offenders = [
        f"{path.relative_to(ENGINE_ROOT)}:{n}"
        for path in source_files()
        for n, line in code_lines(path).items()
        if compiled.search(line)
    ]
    assert not offenders, f"{why} — found at {offenders}"
