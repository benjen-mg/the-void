"""Exit cleanly once the tests are done.

When the interpreter shuts down after a run that opened Qt windows and
VTK views, Qt and VTK objects can be destroyed in an order Qt does not
allow. On GitHub's test machines (Python 3.12) every test passed and
the process then crashed ("shared QObject was deleted directly",
segmentation fault), so the run was reported as failed. Once pytest has
printed its summary, nothing else needs to happen, so a run that used
Qt exits at that point with pytest's own result.
"""

import os
import sys

_EXIT_STATUS = None


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


def pytest_unconfigure(config):
    if _EXIT_STATUS is None or "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
