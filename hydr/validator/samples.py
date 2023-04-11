import sys
import os
import numpy as np
import pandas as pd

from PySide6.QtGui import QFont, QColor, QPalette, Qt, QPen
from PySide6.QtCore import QRect, Slot
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QFrame,
                               QTableWidgetItem, QAbstractItemView, QStyledItemDelegate,
                               QHeaderView, QStyleOptionViewItem, QStyle)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.validator.signals import Receiver
from hydr.types import LoadMethod, Status
from hydr.utils import unique_items


class SampleDisplay(QFrame, Receiver):
    _table_values = None
    _columns = None
    _cheadings = None
    _colors = None

    def __init__(self, parent):
        QFrame.__init__(self, parent)
        Receiver.__init__(self)
        self.layout = QVBoxLayout(self)
        self.setMinimumHeight(200)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)

        self._columns = [
            'file', 'global_start', 'global_end', 'code', 'score', 'blast-1_confidence',
            'blast-2_confidence', 'blast-3_confidence', 'blast-4_confidence',
            'undetermined_confidence', 'background_confidence'
        ]

        section_label_font = QFont()
        section_label_font.setBold(True)
        section_label_font.setPointSize(11)

        self.label = QLabel('Samples:', self)
        self.label.setFont(section_label_font)

        # create table and add headings
        self._cheadings = (
                ['#', 'Sample ID'] +
                [c.replace('_', ' ').title() for c in self._columns]
        )
        self.table_frame = QFrame(self)
        self.table_frame.layout = QHBoxLayout(self.table_frame)
        self.table_frame.layout.setContentsMargins(0, 0, 0, 0)
        self.table_frame.layout.setSpacing(0)

        headings_font = QFont()
        headings_font.setBold(True)
        headings_font.setPointSize(9)

        df = self.state.df_set.copy()
        self.table = QTableWidget(df.shape[0], len(self._cheadings), self.table_frame)
        self.table.clicked.connect(self.table_clicked)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for i in range(self.table.columnCount()):
            if i in range(5):
                resize_mode = QHeaderView.ResizeMode.Fixed
            elif i in [5, 6]:
                resize_mode = QHeaderView.ResizeMode.Interactive
            elif i == (self.table.columnCount()-1):
                resize_mode = QHeaderView.ResizeMode.Stretch
            else:
                resize_mode = QHeaderView.ResizeMode.ResizeToContents
            self.table.horizontalHeader().setSectionResizeMode(i, resize_mode)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 160)
        self.table.setColumnWidth(3, 135)
        self.table.setColumnWidth(4, 135)
        self.table.setColumnWidth(5, 100)
        self.table.setColumnWidth(6, 70)

        self.table.horizontalHeader().sectionClicked.connect(self.header_clicked)
        self.table.setItemDelegate(ColorDelegate())
        self.table.setHorizontalHeaderLabels(self._cheadings)
        for i in range(self.table.columnCount()):
            self.table.horizontalHeaderItem(i).setFont(headings_font)
        self.table_frame.layout.addWidget(self.table)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.table_frame)

        self.on_set_updated(self)
        self.on_load_sample(self, LoadMethod.Fresh)

    @property
    def colors(self):
        if self._colors is None:
            self._colors = {
                Status.Submitted: QColor(121, 210, 121),
                Status.Modified: QColor(255, 194, 102),
                Status.Revisit: QColor(230, 204, 255),
                Status.Skipped: QColor(255, 255, 204),
                Status.MachineLabelled: QColor(255, 179, 179),
                Status.NoStatus: QColor(217, 217, 217)
            }
        return self._colors

    def enter_data_row(self, row):
        status = row['val_status']
        color = self.colors[status]
        index = row['index']
        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, index + 1)
        item.setBackground(color)
        self.table.setItem(index, 0, item)

        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, self.state.shistory[index])
        item.setBackground(color)
        self.table.setItem(index, 1, item)
        for j, c in enumerate(self._columns):
            value = row[c]
            if type(value) is pd.Timestamp:
                value = str(value)
            elif type(value) is np.float64:
                value = float(value)
            else:
                if c == 'file':
                    value = os.path.basename(value)
                elif c == 'code' and status in Status.Submitted | Status.Skipped:
                    value = ','.join(
                        unique_items(
                            [str(i.code) for i in row['val_samples'] if
                             i.code is not None]
                        )
                    )

            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, value)
            item.setBackground(color)
            self.table.setItem(index, j+2, item)

    def on_set_updated(self, _):
        df = self.state.df_set.copy()
        df['index'] = np.arange(df.shape[0])
        self.table.setRowCount(df.shape[0])
        self.table.setSortingEnabled(False)

        # Populate table
        df.apply(self.enter_data_row, axis=1)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, order=Qt.AscendingOrder)

    def on_load_sample(self, sender, how):
        if (
                self.state.current_sample['val_status'] is not None and
                self.state.shistory_idx is not None
        ):
            status = self.state.current_sample['val_status']
            c = self.colors[status]
            codes = ','.join(
                unique_items(
                    [str(i.code) for i in self.state.new_samples if i.code is not None]
                )
            )
            rownum = self.state.shistidx_to_rownum[self.state.shistory_idx]
            for i in range(self.table.columnCount()):
                if (
                        self._columns[i-2] == 'code' and
                        status in Status.Submitted | Status.Revisit
                ):
                    self.table.setItem(rownum, i, QTableWidgetItem(codes))
                item = self.table.item(rownum, i)
                item.setBackground(c)
            self.table.selectRow(rownum)
        else:
            self.table.clearSelection()

    @Slot()
    def table_clicked(self):
        shistidx = self.state.rownum_to_shistidx[self.table.currentRow()]
        if shistidx != self.state.shistory_idx:
            self.state.save_changes()
            self.state.set_index(self.state.shistory[shistidx])

    @Slot()
    def header_clicked(self):
        mapping = [
            (i, self.table.item(i, 0).data(Qt.DisplayRole) - 1)
            for i in range(self.table.rowCount())
        ]
        self.state.rownum_to_shistidx = [shist for rownum, shist in mapping]
        mapping.sort(key=lambda x: x[1])
        self.state.shistidx_to_rownum = [rownum for rownum, shist in mapping]


class ColorDelegate(QStyledItemDelegate, Receiver):
    _colors = None

    def __init__(self, *args):
        QStyledItemDelegate.__init__(self, *args)
        Receiver.__init__(self)

    @property
    def colors(self):
        if self._colors is None:
            self._colors = {
                Status.Submitted: QColor(121, 210, 121),
                Status.Modified: QColor(255, 194, 102),
                Status.Revisit: QColor(230, 204, 255),
                Status.Skipped: QColor(255, 255, 204),
                Status.MachineLabelled: QColor(255, 179, 179),
                Status.NoStatus: QColor(217, 217, 217)
            }
        return self._colors

    def paint(self, painter, option: QStyleOptionViewItem, index):
        if QStyle.State_Selected in option.state:
            option.state = QStyle.State_Enabled
            vid = self.state.df_set.columns.get_loc('val_status')
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            option.font = font
            option.palette.setColor(
                QPalette.Highlight,
                self.colors[self.state.df_set.iloc[index.row(), vid]]
            )

            QStyledItemDelegate.paint(self, painter, option, index)

            rect = QRect(option.rect)
            painter.setPen(QPen(Qt.black, 4))
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

            # Draw left edge on left-most cell
            if index.column() == 0:
                painter.drawLine(rect.topLeft(), rect.bottomLeft())

            # Draw right edge of right-most cell
            if index.column() == index.model().columnCount() - 1:
                painter.drawLine(rect.topRight(), rect.bottomRight())
        else:
            QStyledItemDelegate.paint(self, painter, option, index)