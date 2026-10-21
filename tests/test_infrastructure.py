"""Settings and logging configuration.

Small modules, but Settings is not cosmetic: ORGANIZATION and
APPLICATION are what QSettings keys its on-disk location by, so a
change to either silently orphans every stored preference. That is
worth a test that fails loudly when it happens, rather than a rename
sailing through unnoticed.
"""

import logging
from pathlib import Path

from thevoid_core.infrastructure.logging import configure_logging
from thevoid_core.infrastructure.settings import Settings


def test_the_qsettings_identity_is_the_rebranded_one():
    assert Settings.ORGANIZATION == "the Void"
    assert Settings.APPLICATION == "the Void"


def test_organization_and_application_match():
    """QSettings composes its path from both. They are the same string
    on purpose - a single-product organisation - and a test that pins
    the relationship catches one being renamed without the other.
    """
    assert Settings.ORGANIZATION == Settings.APPLICATION


def test_data_directory_is_a_hidden_folder_under_the_home_directory():
    directory = Settings.data_directory()

    assert isinstance(directory, Path)
    assert directory.parent == Path.home()
    assert directory.name.startswith(".")


def test_data_directory_is_stable_across_calls():
    assert Settings.data_directory() == Settings.data_directory()


def test_configure_logging_sets_the_root_level_to_info():
    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = list(root.handlers)

    try:
        root.handlers.clear()

        configure_logging()

        assert root.level == logging.INFO
        assert root.handlers
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


def test_configure_logging_formats_with_time_and_level():
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level

    try:
        root.handlers.clear()

        configure_logging()

        formatter = root.handlers[0].formatter
        formatted = formatter.format(
            logging.LogRecord(
                name="thevoid_core",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="hello",
                args=(),
                exc_info=None,
            )
        )

        assert "INFO" in formatted
        assert formatted.endswith("hello")
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


def test_configure_logging_is_safe_to_call_twice():
    """basicConfig is a no-op once handlers exist; calling it again
    from a test run, or from an application restarted in-process, must
    not raise or stack duplicate handlers.
    """
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level

    try:
        root.handlers.clear()

        configure_logging()
        count_after_first = len(root.handlers)
        configure_logging()

        assert len(root.handlers) == count_after_first
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
