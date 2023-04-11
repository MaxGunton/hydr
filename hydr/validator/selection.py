import sys
import os

import numpy as np


from PySide6.QtCore import Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QComboBox, QFrame, QLineEdit

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.validator.signals import Receiver
from hydr.validator.misc import invalid_index_popup
from hydr.types import Status


class SetSelection(QFrame, Receiver):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent)
        Receiver.__init__(self)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # left, top, right, bottom

        self.sample_num_label = QLabel('Sample Number:', self)
        self.sample_num = QLabel('', self)
        self.sample_num.setObjectName('sample_num')
        self.sample_num.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.sample_num.setStyleSheet("#sample_num {background-color: "
                                      "rgb(250, 250, 250);}")

        self.sample_id_label = QLabel('Sample ID:', self)
        self.sample_id = QLineEdit(self)
        self.sample_id.returnPressed.connect(self.sample_id_set)
        self.sample_id.setMaximumWidth(50)

        self.model_label = QLabel('Model:', self)
        self.model = QComboBox(self)
        self.model.addItems(self.state.all_models)
        self.model.currentIndexChanged.connect(self.update_model)

        self.sn_label = QLabel('SN:', self)
        self.sn = QComboBox(self)
        self.sn.addItems(self.state.all_sns)
        self.sn.currentIndexChanged.connect(self.update_sn)
        self.progress = QLabel(self)

        self.layout.addWidget(self.sample_num_label)
        self.layout.addWidget(self.sample_num)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.sample_id_label)
        self.layout.addWidget(self.sample_id)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.sn_label)
        self.layout.addWidget(self.sn)
        self.layout.addStretch(1)
        self.layout.addWidget(self.progress)

        self.state.sn = self.sn.currentText()  # triggers an update

    @Slot()
    def update_sn(self) -> None:
        if (
                self.state.sn != self.sn.currentText() and
                self.sn.currentText() != ''
        ):
            self.state.sn = self.sn.currentText()

    @Slot()
    def update_model(self) -> None:
        if (
                self.state.model != self.model.currentText() and
                self.model.currentText() != ''
        ):
            self.state.model = self.model.currentText()

    @Slot()
    def sample_id_set(self) -> None:
        self.state.save_changes()
        try:
            index = int(self.sample_id.text())
            self.state.set_index(index)
        except (ValueError, IndexError):
            invalid_index_popup(f'[0 - {self.state.df_all.shape[0]-1}]')
            self.sample_id.setText(str(self.state.index))  # back to original value

    def update_progress(self):
        index = (
            None
            if self.state.index is None or self.state.index not in self.state.shistory
            else self.state.shistory.index(self.state.index) + 1
        )
        self.sample_num.setText(f'{index}')

        df_set_idxs = self.state.df_set.index.tolist()
        df = self.state.df_all

        a = len(df_set_idxs)
        v = len(
            [
                i
                for i in df.index[df['val_status'] == Status.Submitted].tolist()
                if i in df_set_idxs
            ]
        )
        r = len(
            [
                i
                for i in df.index[df['val_status'] == Status.Revisit].tolist()
                if i in df_set_idxs
            ]
        )
        s = len(
            [
                i
                for i in df.index[df['val_status'] == Status.Skipped].tolist()
                if i in df_set_idxs
            ]
        )
        m = len(
            [
                i
                for i in df.index[df['val_status'] == Status.Modified].tolist()
                if i in df_set_idxs
            ]
        )
        progress = np.round((v+s+r)/a * 100, 2) if a != 0 else 100.00
        msg = (f'{v + s + m + r} of {a} Processed -- [{progress}%] '
               f'({v} submitted | {r} revisit | {s} skipped | {m} modified')
        self.progress.setText(msg)

    def on_load_sample(self, _, how):
        self.sample_id.setText(str(self.state.index))
        self.update_progress()

    def on_set_updated(self, _):
        self.model.setCurrentIndex(self.state.all_models.index(self.state.model))
        self.sn.setCurrentIndex(self.state.all_sns.index(self.state.sn))
        self.update_progress()
