"""FinDyn compute job: weekly.

Delivered in milestone M4 — FINDYN_V1_SPEC.md §11 (Monte Carlo + forecast distributions).
"""

from __future__ import annotations

import sys

from jobs._common import base_parser, configure_logging, not_yet


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__ or "weekly").parse_args(argv)
    configure_logging(args.verbose)
    return not_yet("M4", "§11 (Monte Carlo + forecast distributions)")


if __name__ == "__main__":
    sys.exit(main())
