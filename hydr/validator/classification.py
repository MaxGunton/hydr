import sys
import os

from PySide6.QtCore import Slot, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QFocusEvent, QFont
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QFrame, QPushButton,
                               QLineEdit, QRadioButton, QButtonGroup, QCheckBox)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import BLAST_CLASSES, STATUS_COLORS
from hydr.validator.signals import Receiver
from hydr.types import Status, LoadMethod
from hydr.validator.misc import missing_code_popup, QHLine, QVLine


# TODO: Add a button to remove a bound (holding shift and clicking on or something) a
#       bound will select it and then the button will become active and it
#       can be removed.
# TODO: Change occurrences of classification to label
class ClassificationControls(QFrame, Receiver):

    def __init__(self, parent):
        QFrame.__init__(self, parent)
        Receiver.__init__(self)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        section_label_font = QFont()
        section_label_font.setBold(True)
        section_label_font.setPointSize(11)

        self.class_heading = QLabel('Classification:', self)
        self.class_heading.setFont(section_label_font)
        self.class_heading.setFixedHeight(25)

        self.label_details = QFrame(self)
        self.label_details.layout = QHBoxLayout(self.label_details)
        self.label_details.layout.setContentsMargins(0, 0, 0, 0)
        self.label_details.layout.setSpacing(0)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)

        # Left
        self.left = QFrame(self.label_details)
        self.left.setLineWidth(1)
        self.left.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.left.layout = QVBoxLayout(self.left)
        self.left.layout.setContentsMargins(0, 0, 3, 0)
        self.left.layout.setSpacing(0)

        c = STATUS_COLORS
        btn_font = QFont()
        btn_font.setBold(True)
        btn_font.setPointSize(10)
        # btn_font.set
        self.submit = QPushButton('Submit', self.left)
        self.submit.clicked.connect(self.submit_pressed)
        self.submit.setMaximumHeight(25)
        self.submit.setMaximumWidth(150)
        self.submit.setFont(btn_font)
        self.submit.setObjectName('submit')
        self.submit.setStyleSheet(
            "#submit {background-color: " + c['Status.Submitted'] + ";}"
        )
        self.revisit = QPushButton('Revisit', self.left)
        self.revisit.clicked.connect(self.revisit_pressed)
        self.revisit.setMaximumHeight(25)
        self.revisit.setMaximumWidth(150)
        self.revisit.setFont(btn_font)
        self.revisit.setObjectName('revisit')
        self.revisit.setStyleSheet(
            "#revisit {background-color: " + c['Status.Revisit'] + ";}"
        )
        self.skip = QPushButton('Skip', self.left)
        self.skip.clicked.connect(self.skip_pressed)
        self.skip.setMaximumHeight(25)
        self.skip.setMaximumWidth(150)
        self.skip.setFont(btn_font)
        self.skip.setObjectName('skip')
        self.skip.setStyleSheet(
            "#skip {background-color: " + c['Status.Skipped'] + ";}"
        )
        self.autoset = QPushButton('Auto Set', self.left)
        self.autoset.clicked.connect(self.autoset_pressed)
        self.autoset.setMaximumHeight(25)
        self.autoset.setMaximumWidth(150)
        self.autoset.setFont(btn_font)
        self.autoset.setObjectName('autoset')
        self.autoset.setStyleSheet(
            "#autoset {background-color: " + c['Status.Modified'] + ";}"
        )
        self.clear = QPushButton('Clear', self.left)
        self.clear.clicked.connect(self.clear_pressed)
        self.clear.setMaximumHeight(25)
        self.clear.setMaximumWidth(150)
        self.clear.setFont(btn_font)
        self.clear.setObjectName('clear')
        self.clear.setStyleSheet(
            "#clear {background-color: " + c['Status.MachineLabelled'] + ";}"
        )

        self.left.layout.addStretch(1)
        self.left.layout.addWidget(self.submit)
        self.left.layout.addWidget(self.revisit)
        self.left.layout.addWidget(self.skip)
        self.left.layout.addStretch(1)
        self.left.layout.addWidget(QHLine(self.left))
        self.left.layout.addStretch(1)
        self.left.layout.addWidget(self.autoset)
        self.left.layout.addWidget(self.clear)
        self.left.layout.addStretch(1)



        # Middle
        self.middle = QFrame(self.label_details)
        # self.middle.setLineWidth(1)
        # self.middle.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.middle.layout = QVBoxLayout(self.middle)
        self.middle.layout.setContentsMargins(0, 0, 0, 0)
        self.middle.layout.setSpacing(0)
        self.class_label = {c: QLabel(c, self.middle) for c in BLAST_CLASSES}
        self.middle.layout.addWidget(QHLine(self.middle))
        for i, label in enumerate(self.class_label.values()):
            if i % 2 == 0:
                label.setObjectName(f'class_{i}')
                label.setStyleSheet(f"#class_{i}" +
                                    " {background-color: rgb(220, 220, 220);}")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedHeight(15)
            self.middle.layout.addWidget(label)
            self.middle.layout.addWidget(QHLine(self.middle))
        self.middle.layout.addStretch(1)

        self.right = QFrame(self.label_details)
        # self.right.setLineWidth(1)
        # self.right.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.right.layout = QHBoxLayout(self.right)
        self.right.layout.setContentsMargins(0, 0, 0, 0)
        self.right.layout.setSpacing(0)
        self.samples = []
        self.separators = []

        self.label_details.layout.addWidget(self.left)
        self.label_details.layout.addWidget(self.middle)
        self.label_details.layout.addWidget(self.right)

        self.layout.addWidget(self.class_heading)
        self.layout.addWidget(self.label_details)

        self.on_load_sample(self, LoadMethod.Fresh)

    # This function always loads the same
    def on_load_sample(self, _, how):
        ns = self.state.new_samples
        self.disable(True if self.state.index is None else False)
        num_original_widgets = len(self.samples)  # 0 initially
        self.samples += [
            Classification(i, self.right)
            for i in range(num_original_widgets, len(ns))
        ]
        self.separators += [
            QVLine(self.right)
            for _ in range(num_original_widgets, len(ns))
        ]
        num_new_widgets = len(self.samples)
        for i in range(num_original_widgets, num_new_widgets):
            self.right.layout.addWidget(self.separators[i])
            self.right.layout.addWidget(self.samples[i])

        # set new values for the widgets
        for i, w in enumerate(self.samples):
            if i < len(ns):
                visible = False if ns[i].start is None or ns[i].end is None else True
                w.setVisible(visible)
                self.separators[i].setVisible(visible)
                include_peak = (
                    True if self.state.new_samples[i].peak is not None else False
                )
                comment = self.state.new_samples[i].comment
                code = self.state.new_samples[i].code

                w.include_peak.setChecked(include_peak)
                w.comment.setText(
                    '[optional]'
                    if comment is None or comment == ''
                    else comment
                )
                if code is not None:
                    if code.startswith('blast'):
                        w.class_btn['blast'].setChecked(True)
                    elif code == 'grumble':
                        w.class_btn['grumble'].setChecked(True)
                    elif code == 'animal_noise':
                        w.class_btn['animal noise'].setChecked(True)
                    elif code == 'bump/scrap':
                        w.class_btn['bump/scrap'].setChecked(True)
                    elif code == 'vessel':
                        w.class_btn['vessel'].setChecked(True)
                    elif code == 'background':
                        w.class_btn['background'].setChecked(True)
                    elif code == 'undetermined':
                        w.class_btn['undetermined'].setChecked(True)
                    elif code == 'other':
                        w.class_btn['other'].setChecked(True)
                w.sample_index = self.state.index
            else:
                w.setVisible(False)
                self.separators[i].setVisible(False)


    @Slot()
    def submit_pressed(self):
        if not any([True for i in self.state.new_samples if i.code is None]):
            self.state.current_sample['val_status'] = Status.Submitted
            self.state.refresh(save=True)
            # QTimer.singleShot(50, self.state.next_index)
            QTimer.singleShot(50, self.state.update_index)
        else:
            missing_code_popup()

    @Slot()
    def revisit_pressed(self):
        self.state.current_sample['val_status'] = Status.Revisit
        self.state.refresh(save=True)
        # QTimer.singleShot(50, self.state.next_index)
        QTimer.singleShot(50, self.state.update_index)


    @Slot()
    def skip_pressed(self):
        self.state.new_samples = None
        self.state.current_sample['val_status'] = Status.Skipped
        self.state.refresh(save=True)
        # QTimer.singleShot(50, self.state.next_index)
        QTimer.singleShot(50, self.state.update_index)

    @Slot()
    def autoset_pressed(self):
        self.state.new_samples = None
        self.state.new_samples[0].code = 'blast'
        self.state.current_sample['val_status'] = Status.Modified
        self.state.autoset(save=True)

    @Slot()
    def clear_pressed(self):
        self.state.new_samples = None
        self.state.current_sample['val_status'] = Status.MachineLabelled
        self.state.refresh(save=True)

    def disable(self, disable: bool):
        for w in self.samples:
            w.disable(disable)
        self.submit.setDisabled(disable)
        self.revisit.setDisabled(disable)
        self.skip.setDisabled(disable)
        self.clear.setDisabled(disable)
        self.autoset.setDisabled(disable)


# TODO: See if Classification can detect it's index on it's own
class Classification(QFrame, Receiver):

    def __init__(self, index: int, parent=None):
        QFrame.__init__(self, parent=parent)
        Receiver.__init__(self)
        self.index = index
        self.sample_index = None
        self.original_index = self.state.index
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        # self.setLineWidth(1)
        # self.setFrameStyle(QFrame.Box | QFrame.Plain)
        # print(f'index: {self.index}')
        self.class_btn = {c: QRadioButton(self) for c in BLAST_CLASSES}
        self.class_btn_group = QButtonGroup(self)
        self.layout.addWidget(QHLine(self))
        for i, b in enumerate(self.class_btn.values()):
            b.setFixedHeight(15)
            if i % 2 == 0:
                b.setObjectName(f'selection_{i}')
                b.setStyleSheet(f"#selection_{i}" +
                                " {background-color: rgb(220, 220, 220);}")
            self.class_btn_group.addButton(b)
            self.layout.addWidget(b)
            self.layout.addWidget(QHLine(self))

        self.ip_frame = QFrame(self)
        self.ip_frame.layout = QHBoxLayout(self.ip_frame)
        self.ip_frame.layout.setContentsMargins(0, 0, 0, 0)
        self.include_peak_label = QLabel('Include Peak:', self.ip_frame)
        self.include_peak = QCheckBox('', self.ip_frame)
        self.ip_frame.layout.addWidget(self.include_peak_label)
        self.ip_frame.layout.addWidget(self.include_peak)
        self.ip_frame.layout.addStretch(1)

        self.btm = QFrame(self)
        self.btm.layout = QHBoxLayout(self.btm)
        self.btm.layout.setContentsMargins(0, 0, 0, 0)
        self.comment_label = QLabel('Comment:', self.btm)
        self.comment = CommentLine(self.btm)
        self.btm.layout.addWidget(self.comment_label)
        self.btm.layout.addSpacing(13)
        self.btm.layout.addWidget(self.comment)

        self.layout.addWidget(self.ip_frame)
        self.layout.addWidget(self.btm)
        self.layout.addStretch(1)

        self.class_btn_group.buttonToggled.connect(self.sync_code_with_state)
        self.comment.textChanged.connect(self.sync_comment_with_state)
        self.include_peak.toggled.connect(self.include_peak_toggled)

    @Slot()
    def sync_comment_with_state(self, text):
        if (
                self.index < len(self.state.new_samples) and
                self.sample_index == self.state.index
        ):
            comment = '' if text == '[optional]' else text
            comment_orig = self.state.new_samples[self.index].comment
            status_orig = self.state.current_sample['val_status']
            self.state.new_samples[self.index].comment = comment
            if comment_orig != comment and status_orig != Status.Modified:
                self.state.current_sample['val_status'] = Status.Modified
                self.state.refresh()

    @Slot()
    def sync_code_with_state(self, button, checked):
        # if (
        #         self.index < len(self.state.new_samples) and
        #         self.sample_index == self.state.index
        # ):
        if self.index < len(self.state.new_samples):
            if checked:
                for k, v in self.class_btn.items():
                    if button == v:
                        self.state.new_samples[self.index].code = k.replace(' ', '_')
                        break
                if self.sample_index == self.state.index:
                    status_orig = self.state.current_sample['val_status']
                    self.state.current_sample['val_status'] = Status.Modified
                    if status_orig != Status.Modified:
                        print('Code update successfully')
                        print('Saving ...')
                        self.state.refresh()

    @Slot()
    def include_peak_toggled(self, checked):
        if (
                self.index < len(self.state.new_samples) and
                self.sample_index == self.state.index
        ):
            if checked:
                self.state.add_peak(self.index)
            else:
                self.state.new_samples[self.index].peak = None
                self.state.refresh()

    def disable(self, disable):
        self.include_peak.setDisabled(disable)
        for v in self.class_btn.values():
            v.setDisabled(disable)
        self.comment.setDisabled(disable)


class CommentLine(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if self.text() == '[optional]':
            self.setText('')

    def focusOutEvent(self, event: QFocusEvent):
        super().focusOutEvent(event)
        if self.text() == '':
            self.setText('[optional]')
