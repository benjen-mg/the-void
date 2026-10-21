"""HTTP and the download cache for the reference-database hooks
(round 56). Standard library only: urllib honours the system's proxy
settings (HTTPS_PROXY and friends) without extra configuration.

Every network call goes through `get_bytes`, which takes an `opener`
argument so tests - and anyone behind a peculiar network - can supply
their own transport.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from thevoid_core.version import __version__

USER_AGENT = f"the-Void/{__version__} (+reference database hooks)"
TIMEOUT_S = 60
CHUNK = 1 << 16

CACHE_ENVIRONMENT_VARIABLE = "THE_VOID_CACHE_DIR"


class DownloadError(RuntimeError):
    """A download failed; the message says what to try."""


class Cancelled(RuntimeError):
    """The progress callback asked to stop."""


def cache_dir() -> Path:
    """Where downloaded archives are kept between runs:
    $THE_VOID_CACHE_DIR, else the platform's usual user cache folder."""
    override = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        path = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "the-void" / "cache"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Caches" / "the-void"
    else:
        path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "the-void"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _urlopen(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def get_bytes(url: str, *, opener=None, progress=None, timeout=TIMEOUT_S) -> bytes:
    """The body at `url`. `progress(done_bytes, total_bytes_or_None)`
    is called as data arrives; returning False cancels (Cancelled).
    `opener(url)` replaces the network entirely when given: it returns
    the body as bytes."""
    if opener is not None:
        data = opener(url)
        if progress is not None and progress(len(data), len(data)) is False:
            raise Cancelled(url)
        return data
    try:
        with _urlopen(url, timeout) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            chunks, done = [], 0
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                chunks.append(chunk)
                done += len(chunk)
                if progress is not None and progress(done, total) is False:
                    raise Cancelled(url)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"{url}: HTTP {exc.code} {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise DownloadError(
            f"{url}: {exc.reason}. Check the network connection (and any proxy "
            "settings); the rest of the Void works offline."
        ) from None
    except TimeoutError:
        raise DownloadError(f"{url}: no response within {timeout} s") from None


def cached_download(url: str, name: str, *, expected_sha256=None, opener=None,
                    progress=None, refresh=False) -> tuple[Path, str, bool]:
    """(path, sha256, matches_expected) for `url`, downloading it into the
    cache as `name` unless a copy is there already (refresh=True
    downloads again). matches_expected is None when there is nothing to
    compare with. A copy is only ever stored whole (written to a .part
    file, then renamed), so a cached file with an unexpected checksum is
    a different upstream version, not a broken download: it is used,
    and the caller says so."""
    path = cache_dir() / name
    if path.is_file() and not refresh:
        digest = sha256_of(path)
        return path, digest, (None if expected_sha256 is None else digest == expected_sha256)
    data = get_bytes(url, opener=opener, progress=progress)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    digest = hashlib.sha256(data).hexdigest()
    return path, digest, (None if expected_sha256 is None else digest == expected_sha256)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
