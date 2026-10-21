import logging
import sys
import traceback
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from thevoid_core.infrastructure.logging import configure_logging
from thevoid_core.infrastructure.settings import Settings
from thevoid_core.ui.theme import apply_theme
from thevoid_core.ui.windows.main_window import MainWindow
from thevoid_core.version import __version__

#: Packaged next to this module rather than looked up by an absolute
#: path, so the icon resolves correctly regardless of the current
#: working directory the application happens to be launched from.
ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "icon.png"


def install_exception_hook() -> None:
    """Round 51: an exception escaping a Qt slot used to be printed to a
    console a desktop user never sees, and the action silently did
    nothing. Now it is written to the log file and shown, with a pointer
    to Help > Copy Diagnostics.
    """
    def hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        text = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        logging.getLogger("thevoid_core").critical("Unhandled exception:\n%s", text)

        try:
            from PySide6.QtWidgets import QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "Unexpected Error",
                    f"{exc_type.__name__}: {exc_value}\n\nThe details are in "
                    "the log file. Help > Copy Diagnostics puts them on the "
                    "clipboard for a bug report.",
                )
        except Exception:
            pass

    sys.excepthook = hook


class TheVoidApplication:
    def run(self) -> int:
        configure_logging()
        install_exception_hook()
        logging.getLogger("thevoid_core").info("the Void %s starting", __version__)

        app = QApplication(sys.argv)

        # Needed for QSettings to find a consistent location.
        app.setOrganizationName(Settings.ORGANIZATION)
        app.setApplicationName(Settings.APPLICATION)
        app.setApplicationVersion(__version__)

        if ICON_PATH.exists():
            app.setWindowIcon(QIcon(str(ICON_PATH)))

        apply_theme(app)

        window = MainWindow()
        window.show()

        return app.exec()


def main() -> int:
    return TheVoidApplication().run()


if __name__ == "__main__":
    sys.exit(main())
