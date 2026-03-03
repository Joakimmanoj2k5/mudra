import sys

from PyQt5.QtWidgets import QApplication

from utils.first_run import run_first_time_setup
from ui.screens.main_window import MudraMainWindow


def main() -> int:
    run_first_time_setup()
    app = QApplication(sys.argv)
    window = MudraMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
