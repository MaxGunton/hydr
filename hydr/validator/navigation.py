import sys
import os
import colorsys

from PySide6.QtCore import Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QFrame, QPushButton)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.validator.signals import Receiver
from hydr.types import LoadMethod, Status
from hydr.definitions import STATUS_COLORS


class Navigation(QFrame, Receiver):

    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        Receiver.__init__(self)
        self.layout = QHBoxLayout(self)

        button_font = QFont()
        button_font.setBold(True)

        submit_color = self.darken_color(STATUS_COLORS[str(Status.Submitted)])
        skip_color = self.darken_color(STATUS_COLORS[str(Status.Skipped)])
        revisit_color = self.darken_color(STATUS_COLORS[str(Status.Revisit)])
        ml_color = self.darken_color(STATUS_COLORS[str(Status.MachineLabelled)])

        # self.first_sample = QPushButton('First Sample', self)
        # self.first_sample = QPushButton('FIRST', self)
        self.first_sample = QPushButton('\u21e4 First', self)
        self.first_sample.setFont(button_font)
        # self.first_sample.setObjectName('first_sample')
        # self.first_sample.setStyleSheet('#first_sample {color:}')
        self.first_sample.setMaximumHeight(25)
        self.first_sample.setMaximumWidth(150)
        self.first_sample.clicked.connect(self.first_sample_pressed)

        # self.previous_submitted = QPushButton('Previous Submitted', self)
        self.previous_submitted = QPushButton('\u2190 Submitted', self)
        self.previous_submitted.setFont(button_font)
        self.previous_submitted.setObjectName('previous_submitted')
        self.previous_submitted.setStyleSheet(
            "#previous_submitted {color: " + submit_color + ";}"
        )
        self.previous_submitted.setMaximumHeight(25)
        self.previous_submitted.setMaximumWidth(150)
        self.previous_submitted.clicked.connect(self.previous_submitted_pressed)

        # self.previous_skipped = QPushButton('Previous Skipped', self)
        self.previous_skipped = QPushButton('\u2190 Skipped', self)
        self.previous_skipped.setFont(button_font)
        self.previous_skipped.setObjectName('previous_skipped')
        self.previous_skipped.setStyleSheet(
            "#previous_skipped {color: " + skip_color + ";}"
        )
        self.previous_skipped.setMaximumHeight(25)
        self.previous_skipped.setMaximumWidth(150)
        self.previous_skipped.clicked.connect(self.previous_skipped_pressed)

        # self.previous_revisit = QPushButton('Previous Revisit', self)
        self.previous_revisit = QPushButton('\u2190 Revisit', self)
        self.previous_revisit.setFont(button_font)
        self.previous_revisit.setObjectName('previous_revisit')
        self.previous_revisit.setStyleSheet(
            "#previous_revisit {color: " + revisit_color + ";}"
        )
        self.previous_revisit.setMaximumHeight(25)
        self.previous_revisit.setMaximumWidth(150)
        self.previous_revisit.clicked.connect(self.previous_revisit_pressed)

        # self.previous_unprocessed = QPushButton('Previous Unprocessed', self)
        self.previous_unprocessed = QPushButton('\u2190 Unprocessed', self)
        self.previous_unprocessed.setFont(button_font)
        self.previous_unprocessed.setObjectName('previous_unprocessed')
        self.previous_unprocessed.setStyleSheet(
            "#previous_unprocessed {color: " + ml_color + ";}"
        )
        self.previous_unprocessed.setMaximumHeight(25)
        self.previous_unprocessed.setMaximumWidth(150)
        self.previous_unprocessed.clicked.connect(self.previous_unprocessed_pressed)

        # self.previous = QPushButton('Previous', self)
        self.previous = QPushButton('\u2190 -1', self)
        self.previous.setFont(button_font)
        self.previous.setObjectName('previous')
        self.previous.setMaximumHeight(25)
        self.previous.setMaximumWidth(150)
        self.previous.clicked.connect(self.previous_pressed)

        # self.next = QPushButton('Next', self)
        self.next = QPushButton('+1 \u2192', self)
        self.next.setFont(button_font)
        self.next.setObjectName('next')
        self.next.setMaximumHeight(25)
        self.next.setMaximumWidth(150)
        self.next.clicked.connect(self.next_pressed)

        # self.next_unprocessed = QPushButton('Next Unprocessed', self)
        self.next_unprocessed = QPushButton('Unprocessed \u2192', self)
        self.next_unprocessed.setFont(button_font)
        self.next_unprocessed.setObjectName('next_unprocessed')
        self.next_unprocessed.setStyleSheet(
            "#next_unprocessed {color: " + ml_color + ";}"
        )
        self.next_unprocessed.setMaximumHeight(25)
        self.next_unprocessed.setMaximumWidth(150)
        self.next_unprocessed.clicked.connect(self.next_unprocessed_pressed)

        # self.next_revisit = QPushButton('Next Revisit', self)
        self.next_revisit = QPushButton('Revisit \u2192', self)
        self.next_revisit.setFont(button_font)
        self.next_revisit.setObjectName('next_revisit')
        self.next_revisit.setStyleSheet(
            "#next_revisit {color: " + revisit_color + ";}"
        )
        self.next_revisit.setMaximumHeight(25)
        self.next_revisit.setMaximumWidth(150)
        self.next_revisit.clicked.connect(self.next_revisit_pressed)

        # self.next_skipped = QPushButton('Next Skipped', self)
        self.next_skipped = QPushButton('Skipped \u2192', self)
        self.next_skipped.setFont(button_font)
        self.next_skipped.setObjectName('next_skipped')
        self.next_skipped.setStyleSheet(
            "#next_skipped {color: " + skip_color + ";}"
        )
        self.next_skipped.setMaximumHeight(25)
        self.next_skipped.setMaximumWidth(150)
        self.next_skipped.clicked.connect(self.next_skipped_pressed)

        # self.next_submitted = QPushButton('Next Submitted', self)
        self.next_submitted = QPushButton('Submitted \u2192 ', self)
        self.next_submitted.setFont(button_font)
        self.next_submitted.setObjectName('next_submitted')
        self.next_submitted.setStyleSheet(
            "#next_submitted {color: " + submit_color + ";}"
        )
        self.next_submitted.setMaximumHeight(25)
        self.next_submitted.setMaximumWidth(150)
        self.next_submitted.clicked.connect(self.next_submitted_pressed)

        # self.last_sample = QPushButton('Last Sample')
        self.last_sample = QPushButton('Last \u21e5')
        self.last_sample.setFont(button_font)
        self.last_sample.setObjectName('last_sample')
        self.last_sample.setMaximumHeight(25)
        self.last_sample.setMaximumWidth(150)
        self.last_sample.clicked.connect(self.last_sample_pressed)

        self.layout.addWidget(self.first_sample)
        self.layout.addWidget(self.previous_submitted)
        self.layout.addWidget(self.previous_skipped)
        self.layout.addWidget(self.previous_revisit)
        self.layout.addWidget(self.previous_unprocessed)
        self.layout.addWidget(self.previous)
        self.layout.addWidget(self.next)
        self.layout.addWidget(self.next_unprocessed)
        self.layout.addWidget(self.next_revisit)
        self.layout.addWidget(self.next_skipped)
        self.layout.addWidget(self.next_submitted)
        self.layout.addWidget(self.last_sample)

        self.on_load_sample(self, LoadMethod.Fresh)

    def on_load_sample(self, sender, how):
        # if self.state.index is None:
        #     self.disable(True)
        # else:
        self.disable(False)
        if self.state.index is not None and self.state.shistory_idx is not None:
            disable_previous = (
                True
                if self.state.shistory_idx <= 0
                else False
            )
            disable_next = (
                True
                if self.state.shistory_idx >= len(self.state.shistory) - 1
                else False
            )
            self.previous_submitted.setDisabled(disable_previous)
            self.previous_skipped.setDisabled(disable_previous)
            self.previous_revisit.setDisabled(disable_previous)
            self.previous_unprocessed.setDisabled(disable_previous)
            self.previous.setDisabled(disable_previous)
            self.next.setDisabled(disable_next)
            self.next_unprocessed.setDisabled(disable_next)
            self.next_skipped.setDisabled(disable_next)
            self.next_revisit.setDisabled(disable_next)
            self.next_submitted.setDisabled(disable_next)
            self.last_sample.setDisabled(disable_next)

    @Slot()
    def first_sample_pressed(self):
        self.state.save_changes()
        self.state.first_index()

    @Slot()
    def previous_submitted_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Submitted, forward=False)

    @Slot()
    def previous_skipped_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Skipped, forward=False)

    @Slot()
    def previous_revisit_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Revisit, forward=False)

    @Slot()
    def previous_unprocessed_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.MachineLabelled | Status.Modified, forward=False)

    @Slot()
    def previous_pressed(self):
        self.state.save_changes()
        self.state.prev_index()

    @Slot()
    def next_pressed(self):
        self.state.save_changes()
        self.state.next_index()

    @Slot()
    def next_unprocessed_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.MachineLabelled | Status.Modified)

    @Slot()
    def next_revisit_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Revisit)

    @Slot()
    def next_skipped_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Skipped)

    @Slot()
    def next_submitted_pressed(self):
        self.state.save_changes()
        self.state.update_index(Status.Submitted)

    @Slot()
    def last_sample_pressed(self):
        self.state.save_changes()
        self.state.last_index()

    def disable(self, disable):
        self.first_sample.setDisabled(disable)
        self.previous_submitted.setDisabled(disable)
        self.previous_skipped.setDisabled(disable)
        self.previous_revisit.setDisabled(disable)
        self.previous_unprocessed.setDisabled(disable)
        self.previous.setDisabled(disable)
        self.next.setDisabled(disable)
        self.next_unprocessed.setDisabled(disable)
        self.next_revisit.setDisabled(disable)
        self.next_skipped.setDisabled(disable)
        self.next_submitted.setDisabled(disable)

    @staticmethod
    def darken_color(color):
        r, g, b = [
            int(i) / 255
            for i in color.replace(' ', '')[4:-1].split(',')
        ]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        new_color = colorsys.hsv_to_rgb(h, min(max(v*1.1, 0.6), 1.0), v*0.8)
        new_color = (
            'rgb(' +
            ', '.join([str(min(max(int(round(i*255)), 0), 255)) for i in new_color]) +
            ')'
        )
        return new_color
