import sys
import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QApplication,
                               QWidget, QMainWindow, QFrame)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import VALIDATOR_ICON
from hydr.validator.signals import Receiver
from hydr.validator.state import State
from hydr.validator.selection import SetSelection
from hydr.validator.spectrogram import SpectrogramFrame
from hydr.validator.audio import AudioControls
from hydr.validator.classification import ClassificationControls
from hydr.validator.filters import Filters
from hydr.validator.samples import SampleDisplay
from hydr.validator.navigation import Navigation


class Application(QApplication):
    def __init__(self, depfile):
        QApplication.__init__(self, [])
        self.depfile = depfile
        self.state = State(self.depfile, self)
        self.setDoubleClickInterval(200)


class MainWindow(QMainWindow, Receiver):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        Receiver.__init__(self)

        self.setWindowTitle('VALIDATOR')
        self.setWindowIcon(QIcon(VALIDATOR_ICON))

        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self.showMaximized()


class MainWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(3, 0, 3, 3)
        self.layout.setSpacing(3)

        self.set_selection = SetSelection(self)
        self.spectrogram = SpectrogramFrame(self)
        self.audio_and_navigation = AudioAndNavigation(self)
        self.controls_and_filters = ClassificationAndFilters(self)
        self.sample_display = SampleDisplay(self)

        self.layout.addWidget(self.set_selection)
        self.layout.addWidget(self.spectrogram)
        self.layout.addWidget(self.audio_and_navigation)
        self.layout.addWidget(self.controls_and_filters)
        self.layout.addWidget(self.sample_display)


class AudioAndNavigation(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.audio_controls = AudioControls(self)
        self.navigation = Navigation(self)

        self.layout.addWidget(self.audio_controls)
        self.layout.addWidget(self.navigation)


class ClassificationAndFilters(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.classification_controls = ClassificationControls(self)
        self.filters = Filters(self)
        self.layout.addWidget(self.classification_controls)
        self.layout.addWidget(self.filters)
        self.layout.addStretch(1)


def run(depfile: str):
    app = Application(depfile)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
