"""Fitted-parameter storage for engines.

``AssetEngine.fit`` runs monthly and ``predict`` runs daily, in separate
processes: whatever the refit chose has to survive between them. This is that
handle — a JSON file per engine under ``compute/artifacts/``, which is
gitignored because fitted parameters are model output, not source.

Deliberately small. Engines need "write these numbers, read them back, and cope
when they are not there yet"; anything more (versioned blobs, remote object
storage) is framework built ahead of a user for it.

A missing or unreadable artifact is never fatal — ``load`` returns ``{}`` and the
engine falls back to its configured defaults. A daily run must not stop because
a refit has not happened yet, or because someone deleted the directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("findynamics.core.artifacts")

#: Overridable so a test or a CI run never writes into the working tree.
ENV_VAR = "FINDYN_ARTIFACT_DIR"

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def artifact_dir() -> Path:
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else DEFAULT_DIR


class ArtifactStore:
    """Named JSON documents, one per engine."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or artifact_dir()

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid artifact name: {name!r}")
        return self._dir / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        """Stored parameters, or ``{}`` when there are none to load."""
        path = self.path_for(name)
        if not path.exists():
            return {}
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            log.warning("ignoring unreadable artifact %s: %s", path, err)
            return {}
        return body if isinstance(body, dict) else {}

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        """Write parameters, replacing whatever was there.

        Written to a temporary file and renamed, so a crash mid-write leaves the
        previous parameters intact rather than a truncated file that every later
        run silently ignores.
        """
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
        log.info("wrote %s", path)
        return path
