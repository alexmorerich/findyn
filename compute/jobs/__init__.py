"""Compute-plane job entrypoints (FINDYN_V1_SPEC.md §6).

Each job is a plain Python CLI with no assumptions about its host, so the
scheduler can be GitHub Actions today and Cloudflare Containers later without
touching job code.
"""
