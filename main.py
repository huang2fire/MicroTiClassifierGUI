import sys

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
)

from util.config import ConfigManager
from util.gui import MicroTiGUI


def main():
    ConfigManager.load("./config/config.toml")
    config = ConfigManager.get()

    app = QApplication(sys.argv)
    app.setStyle(config["app"]["style"])
    app.setFont(QFont(config["app"]["font"]))
    app.setWindowIcon(QIcon(config["app"]["icon"]))

    window = MicroTiGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
