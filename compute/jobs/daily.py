"""FinDyn compute job: daily.

Delivered in milestone M2 — FINDYN_V1_SPEC.md §6 (daily feature + state run).
"""

from __future__ import annotations

import sys

from jobs._common import base_parser, configure_logging, not_yet


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__ or "daily").parse_args(argv)
    configure_logging(args.verbose)
    return not_yet("M2", "§6 (daily feature + state run)")


if __name__ == "__main__":
    sys.exit(main())
