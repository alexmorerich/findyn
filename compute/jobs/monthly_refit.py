"""FinDyn compute job: monthly_refit.

Delivered in milestone M3 — FINDYN_V1_SPEC.md §9 (expanding-window model refit).
"""

from __future__ import annotations

import sys

from jobs._common import base_parser, configure_logging, not_yet


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__ or "monthly_refit").parse_args(argv)
    configure_logging(args.verbose)
    return not_yet("M3", "§9 (expanding-window model refit)")


if __name__ == "__main__":
    sys.exit(main())
