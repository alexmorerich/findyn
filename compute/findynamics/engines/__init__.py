"""Asset engines — mutually isolated implementations of ``AssetEngine``.

This is the only module allowed to name the engines, and it imports each one
guarded by its config enable flag (``03-contracts.md`` §3): discovery elsewhere
is by registry name, so nothing outside this file depends on which engines
happen to exist. No engine imports another (CI-enforced).

All five subpackages are empty in P0 — the models arrive in P1-P5 — so there is
nothing to import yet and this file stays a docstring.
"""
