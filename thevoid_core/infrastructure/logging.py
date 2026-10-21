import logging
import logging.handlers
from pathlib import Path

from thevoid_core.infrastructure.settings import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: One megabyte per file, three old files kept - enough to cover a
#: session that went wrong, small enough never to matter on disk.
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3


def log_directory() -> Path:
    return Settings.data_directory() / "logs"


def log_file_path() -> Path:
    return log_directory() / "the_void.log"


def configure_logging(*, to_file: bool = True) -> Path | None:
    """Console logging, plus (since round 51) a rotating log file under
    ~/.the_void/logs, so that a user launched from a desktop icon, with
    no console, still has something to attach to a bug report
    (Help > Open Log Folder). Returns the log file's path, or None if
    it could not be created - logging to a file is never allowed to
    stop the application from starting.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console)

    if not to_file:
        return None

    path = log_file_path()

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(
            handler.baseFilename
        ) == path:
            return path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
    except OSError:
        return None

    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)

    return path
