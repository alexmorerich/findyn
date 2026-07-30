"""FinDyn compute job: backfill.

Delivered in milestone M1 — FINDYN_V1_SPEC.md §6 (historical backfill).
"""

from __future__ import annotations

import sys

from jobs._common import base_parser, configure_logging, not_yet


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__ or "backfill").parse_args(argv)
    configure_logging(args.verbose)
    return not_yet("M1", "§6 (historical backfill)")


if __name__ == "__main__":
    sys.exit(main())
