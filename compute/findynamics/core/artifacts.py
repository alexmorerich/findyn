"""Fitted-parameter storage for engines.

``AssetEngine.fit`` runs monthly and ``predict`` runs daily, in separate
processes: whatever the refit chose has to survive between them.

Two implementations of the same tiny interface:

* :class:`ArtifactStore` — a JSON file per engine under ``compute/artifacts/``.
  Fine for a laptop, and **useless in production**: the refit and the daily run
  are different GitHub Actions containers, the directory is gitignored, and the
  runner is destroyed minutes later. The equity engine's correct response to a
  missing fit is to publish nothing, so a production cron on local storage would
  have published nothing forever.
* :class:`RemoteArtifactStore` — the same interface over R2, through the
  serving plane's HMAC-authenticated admin door (FINDYN_V1_SPEC.md §6).

:func:`build_artifact_store` picks between them from the environment, so no
engine and no job has to know which one it is talking to.

**Artifacts are immutable and addressed by model_version.** A version is written
once; re-writing the same bytes is a no-op and re-writing *different* bytes is an
error. That is the property that makes "which model produced this state" an
answerable question — a mutable artifact under a published version means every
backtest of that version silently becomes a backtest of something else.

A missing or unreadable artifact is never fatal — ``load`` returns ``{}`` and the
engine falls back to its configured defaults or declines to publish. A daily run
must not stop because a refit has not happened yet.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from findynamics.core.signing import headers as sign_headers

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


class ArtifactStorage(Protocol):
    """What an engine needs from artifact storage, and nothing more."""

    def load(self, name: str) -> dict[str, Any]: ...

    def save(self, name: str, payload: dict[str, Any]) -> Any: ...


#: Environment variable holding the serving plane's admin base URL. Accepts the
#: write-back endpoint too, so a deployment that already sets FINDYN_ADMIN_URL
#: needs no second variable.
ADMIN_URL_ENV = "FINDYN_ADMIN_URL"
SECRET_ENV = "ADMIN_HMAC_SECRET"

#: Set to pin a run to one exact model version instead of following `latest`.
#: A backtest or a replay of a published state should use this; a daily run
#: should not, or it would never pick up a refit.
VERSION_ENV = "FINDYN_MODEL_VERSION"


def _admin_base(url: str) -> str:
    """Normalise an admin URL to its base, accepting the write-back endpoint."""
    trimmed = url.rstrip("/")
    return trimmed[: -len("/results")] if trimmed.endswith("/results") else trimmed


class RemoteArtifactStore:
    """Fitted models in R2, via the serving plane's admin API.

    Reads and writes are HMAC-signed with the same secret and the same
    ``{timestamp}.{body}`` construction as the write-back, so there is one
    credential and one signature format for the whole compute → serving door.
    """

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        version: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base = _admin_base(base_url)
        self._secret = secret
        self._version = version
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"RemoteArtifactStore(base={self._base!r}, version={self._version or 'latest'})"

    @property
    def directory(self) -> str:
        return f"{self._base}/artifacts"

    def _url(self, name: str, version: str) -> str:
        return f"{self._base}/artifacts/{quote(name, safe='')}/{quote(version, safe='')}"

    def load(self, name: str) -> dict[str, Any]:
        """The pinned version if one is configured, else whatever is current.

        Every failure degrades to ``{}``: a network blip during a daily run
        should leave the engine publishing what it can rather than aborting, and
        every engine already handles "no fit yet".
        """
        import httpx

        version = self._version or "latest"
        url = self._url(name, version)
        try:
            response = httpx.get(
                url,
                headers=sign_headers(self._secret, ""),
                timeout=self._timeout,
            )
        except httpx.HTTPError as err:
            log.warning("artifact %s unreachable (%s); continuing without it", url, err)
            return {}

        if response.status_code == 404:
            log.info("no stored artifact for %s@%s yet", name, version)
            return {}
        if response.status_code >= 400:
            log.warning(
                "artifact %s returned %d: %s", url, response.status_code, response.text[:200]
            )
            return {}

        try:
            body = response.json()
        except ValueError as err:
            log.warning("artifact %s is not JSON (%s)", url, err)
            return {}
        if not isinstance(body, dict):
            return {}

        resolved = response.headers.get("x-findyn-model-version", version)
        log.info("loaded artifact %s@%s from R2", name, resolved)
        return body

    def save(self, name: str, payload: dict[str, Any]) -> str:
        """Store under the payload's own ``model_version``. Write-once.

        The version is taken from the payload rather than passed in, so an
        artifact cannot be filed under a name its own contents disagree with —
        the serving side rejects that mismatch as well, on the same reasoning.
        """
        import httpx

        version = str(payload.get("model_version") or "").strip()
        if not version:
            raise ValueError(
                f"artifact {name!r} has no model_version; a fitted model that "
                "cannot be addressed by version cannot be published"
            )

        body = json.dumps(payload, indent=2, sort_keys=True)
        url = self._url(name, version)
        response = httpx.put(
            url,
            content=body,
            headers={
                "content-type": "application/json",
                **sign_headers(self._secret, body),
            },
            timeout=self._timeout,
        )
        if response.status_code == 409:
            raise ValueError(
                f"artifact {name}@{version} already exists with different content; "
                "fitted models are immutable — bump the model version instead"
            )
        response.raise_for_status()
        log.info("stored artifact %s@%s in R2 (%s)", name, version, response.status_code)
        return version


def build_artifact_store(
    directory: Path | None = None,
    env: dict[str, str] | None = None,
) -> ArtifactStorage:
    """Remote storage when the serving plane is configured, local otherwise.

    The environment decides, so a job never names a backend. A developer running
    the same command with no secrets gets the local directory and a production
    run gets R2 — and there is no flag anyone can forget to pass that would let
    an ephemeral local artifact reach production.
    """
    settings = os.environ if env is None else env
    url = settings.get(ADMIN_URL_ENV)
    secret = settings.get(SECRET_ENV)

    if url and secret:
        return RemoteArtifactStore(url, secret, version=settings.get(VERSION_ENV) or None)

    log.info(
        "%s/%s are not both set; using local artifact storage. This is correct for "
        "development and wrong for production — a scheduled run on local storage "
        "loses every refit when the container is destroyed.",
        ADMIN_URL_ENV,
        SECRET_ENV,
    )
    return ArtifactStore(directory)


__all__ = [
    "ADMIN_URL_ENV",
    "DEFAULT_DIR",
    "ENV_VAR",
    "SECRET_ENV",
    "VERSION_ENV",
    "ArtifactStorage",
    "ArtifactStore",
    "RemoteArtifactStore",
    "artifact_dir",
    "build_artifact_store",
]
