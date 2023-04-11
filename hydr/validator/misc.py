import sys
import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox, QFrame

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
# from this package
from hydr.definitions import WARNING_ICON


def change_set_warning_popup(what_changed: str):
    """
    Displays a pop-up from which the user must select one of the three following
    options:

        1. submit details on current sample and proceed
        2. discard details on current sample and proceed
        3. cancel operation

    :param what_changed: str - Used in display message
    :return:
    """
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon(WARNING_ICON))
    msg_box.setWindowTitle('Unsaved Changes')
    msg_box.setText(f'Changing `{what_changed}` will reload sample set with '
                    f'unsaved changes!')
    msg_box.setInformativeText("Do you want to save your changes?")
    msg_box.setStandardButtons(
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    msg_box.setDefaultButton(QMessageBox.Save)
    return msg_box.exec()


def missing_code_popup():
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon(WARNING_ICON))
    msg_box.setWindowTitle('Missing Code')
    msg_box.setText('Unable to submit sample with missing code!  ')
    msg_box.setInformativeText("You must correct this before you can submit.  ")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setDefaultButton(QMessageBox.Ok)
    return msg_box.exec()


def invalid_index_popup(valid_values):
    msg_box = QMessageBox()
    msg_box.setWindowIcon(QIcon(WARNING_ICON))
    msg_box.setWindowTitle('Invalid Index')
    msg_box.setText(f'A valid index will be an integer in the range: {valid_values}.')
    msg_box.setInformativeText("Change index operation cancelled. ")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setDefaultButton(QMessageBox.Ok)
    return msg_box.exec()


class QHLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
