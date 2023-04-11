import sys
import os
import datetime as dt

from PySide6.QtCore import QTime, QDateTime, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QFrame, QCheckBox,
                               QTimeEdit, QDateTimeEdit, QSpinBox, QComboBox)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.validator.signals import Receiver
from hydr.validator.misc import QHLine
from hydr.definitions import BLASTS_224x224_6CAT_CLASSES


class Filters(QFrame, Receiver):

    def __init__(self, parent):
        QFrame.__init__(self, parent)
        Receiver.__init__(self)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)

        section_label_font = QFont()
        section_label_font.setBold(True)
        section_label_font.setPointSize(11)

        self.filter_label = QLabel('Filters:', self)
        self.filter_label.setFont(section_label_font)
        self.filter_label.setLineWidth(1)
        self.filter_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.filter_label.setContentsMargins(0, 0, 0, 0)
        self._update_set = True
        self._calendar_popup = False
        self._ignore_second = not self._calendar_popup

        # Time of Day Filter
        self.time_of_day = QFrame(self)
        self.time_of_day.setLineWidth(1)
        self.time_of_day.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.time_of_day.layout = QHBoxLayout(self.time_of_day)
        self.time_of_day.layout.setContentsMargins(0, 0, 0, 0)
        self.tod_active = QCheckBox('Keep samples occurring between:', self.time_of_day)
        self.start_time = QTimeEdit(self.time_of_day)
        self.start_time.timeChanged.connect(self.start_time_changed)
        self.end_time = QTimeEdit(self.time_of_day)
        self.end_time.timeChanged.connect(self.end_time_changed)
        self.tod_active.toggled.connect(self.time_of_day_toggled)
        self.time_of_day.layout.addWidget(self.tod_active)
        self.time_of_day.layout.addWidget(self.start_time)
        self.time_of_day.layout.addWidget(QLabel('to'))
        self.time_of_day.layout.addWidget(self.end_time)

        # Date Range Filter
        self.datetime_range = QFrame(self)
        self.datetime_range.setLineWidth(1)
        self.datetime_range.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.datetime_range.layout = QHBoxLayout(self.datetime_range)
        self.datetime_range.layout.setContentsMargins(0, 0, 0, 0)
        self.dtr_active = QCheckBox(
            'Keep samples between date and times:', self.datetime_range
        )
        self.start_datetime = QDateTimeEdit(self.datetime_range)
        self.start_datetime.setDisplayFormat('yyyy-MM-dd hh:mm:ss')
        self.start_datetime.dateTimeChanged.connect(self.start_datetime_changed)
        self.start_datetime.setCalendarPopup(self._calendar_popup)
        self.end_datetime = QDateTimeEdit(self.datetime_range)
        self.end_datetime.setDisplayFormat('yyyy-MM-dd hh:mm:ss')
        self.end_datetime.dateTimeChanged.connect(self.end_datetime_changed)
        self.end_datetime.setCalendarPopup(self._calendar_popup)
        self.dtr_active.toggled.connect(self.datetime_range_toggled)
        self.datetime_range.layout.addWidget(self.dtr_active)
        self.datetime_range.layout.addWidget(self.start_datetime)
        self.datetime_range.layout.addWidget(QLabel('and'))
        self.datetime_range.layout.addWidget(self.end_datetime)

        # Sample Length Filter
        self.sample_length = QFrame(self)
        self.sample_length.setLineWidth(1)
        self.sample_length.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.sample_length.layout = QHBoxLayout(self.sample_length)
        self.sample_length.layout.setContentsMargins(0, 0, 0, 0)
        self.sl_active = QCheckBox('Keep only samples under:', self.sample_length)
        self.max_seconds = QSpinBox(self.sample_length)
        self.max_seconds.setMaximum(43200)
        self.max_seconds.setMinimum(1)
        self.max_seconds.valueChanged.connect(self.max_seconds_changed)
        self.sl_active.toggled.connect(self.sample_length_toggled)
        self.sample_length.layout.addWidget(self.sl_active)
        self.sample_length.layout.addWidget(self.max_seconds)
        self.sample_length.layout.addWidget(QLabel('seconds'))

        # Codes Filter
        self.all_codes = ['Any'] + [i for i in BLASTS_224x224_6CAT_CLASSES]
        self.codes_filter = QFrame(self)
        self.codes_filter.setLineWidth(1)
        self.codes_filter.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.codes_filter.layout = QHBoxLayout(self.codes_filter)
        self.codes_filter.layout.setContentsMargins(0, 0, 0, 0)
        self.codes_label = QLabel('Include Samples with ML code:', self.codes_filter)
        self.codes = QComboBox(self.codes_filter)
        self.codes.addItems(self.all_codes)
        self.codes.currentIndexChanged.connect(self.codes_changed)
        self.codes_filter.layout.addSpacing(18)
        self.codes_filter.layout.addWidget(self.codes_label)
        self.codes_filter.layout.addWidget(self.codes)
        self.codes_filter.layout.addStretch(1)

        self.layout.addWidget(self.filter_label)
        self.layout.addWidget(self.time_of_day)
        self.layout.addWidget(QHLine(self))
        self.layout.addWidget(self.datetime_range)
        self.layout.addWidget(QHLine(self))
        self.layout.addWidget(self.sample_length)
        self.layout.addWidget(QHLine(self))
        self.layout.addWidget(self.codes_filter)
        self.layout.addStretch(1)

        self.on_set_updated(self)

    @Slot()
    def time_of_day_toggled(self, checked):
        f = self.state.filters
        if f['time_of_day'][0] != checked and self._update_set:
            _, st, et = f['time_of_day']
            f['time_of_day'] = (checked, st, et)
            self.state.filters = f

    @Slot()
    def start_time_changed(self, e: QTime):
        new_st = dt.time(hour=e.hour(), minute=e.minute(), second=e.second())
        f = self.state.filters
        tod_active, st, et = f['time_of_day']
        if new_st != st and self._update_set:
            if self.tod_active.isChecked():
                f['time_of_day'] = (tod_active, new_st, et)
                self.state.filters = f
            else:
                self.state.filters['time_of_day'] = (tod_active, new_st, et)

    @Slot()
    def end_time_changed(self, e):
        new_et = dt.time(hour=e.hour(), minute=e.minute(), second=e.second())
        f = self.state.filters
        tod_active, st, et = f['time_of_day']
        if new_et != et and self._update_set:
            if self.tod_active.isChecked():
                f['time_of_day'] = (tod_active, st, new_et)
                self.state.filters = f  # setting filter like this will update set
            else:
                # setting filter like this does not update set
                self.state.filters['time_of_day'] = (tod_active, st, new_et)

    @Slot()
    def datetime_range_toggled(self, checked):
        f = self.state.filters
        if f['datetime_range'][0] != checked and self._update_set:
            _, sdt, edt = f['datetime_range']
            f['datetime_range'] = (checked, sdt, edt)
            self.state.filters = f

    @Slot()
    def start_datetime_changed(self, e: QDateTime):
        date, time = e.date(), e.time()
        new_sdt = dt.datetime(
            year=date.year(), month=date.month(), day=date.day(), hour=time.hour(),
            minute=time.minute(), second=time.second()
        )
        f = self.state.filters
        dtr_active, sdt, edt = f['datetime_range']
        if new_sdt != sdt and self._update_set:
            if self._calendar_popup and self._ignore_second:
                self._ignore_second = False
                self.start_datetime.setDateTime(
                    QDateTime(sdt.year, sdt.month, sdt.day, sdt.hour, sdt.minute,
                              sdt.second)
                )
                return
            self._ignore_second = True
            if self.dtr_active.isChecked():
                f['datetime_range'] = (dtr_active, new_sdt, edt)
                self.state.filters = f
            else:
                self.state.filters['datetime_range'] = (dtr_active, new_sdt, edt)

    @Slot()
    def end_datetime_changed(self, e: QDateTime):
        date, time = e.date(), e.time()
        new_edt = dt.datetime(
            year=date.year(), month=date.month(), day=date.day(), hour=time.hour(),
            minute=time.minute(), second=time.second()
        )
        f = self.state.filters
        dtr_active, sdt, edt = f['datetime_range']
        if new_edt != edt and self._update_set:
            if self._calendar_popup and self._ignore_second:
                self._ignore_second = False
                self.end_datetime.setDateTime(
                    QDateTime(edt.year, edt.month, edt.day, edt.hour, edt.minute,
                              edt.second)
                )
                return
            self._ignore_second = True
            if self.dtr_active.isChecked():
                f['datetime_range'] = (dtr_active, sdt, new_edt)
                self.state.filters = f
            else:
                self.state.filters['datetime_range'] = (dtr_active, sdt, new_edt)

    @Slot()
    def sample_length_toggled(self, checked):
        f = self.state.filters
        if f['sample_length'][0] != checked and self._update_set:
            self.state.save_changes()
            _, sl = f['sample_length']
            f['sample_length'] = (checked, sl)
            self.state.filters = f

    @Slot()
    def max_seconds_changed(self, new_sl):
        f = self.state.filters
        sl_active, sl = f['sample_length']
        if new_sl != sl and self._update_set:
            if self.sl_active.isChecked():
                self.state.save_changes()
                f['sample_length'] = (sl_active, new_sl)
                self.state.filters = f
            else:
                self.state.filters['sample_length'] = (sl_active, new_sl)

    @Slot()
    def codes_changed(self):
        if self.state.filters['codes'] != self.codes.currentText() and self._update_set:
            f = self.state.filters
            f['codes'] = self.codes.currentText()
            self.state.filters = f

    def on_set_updated(self, _):
        self._update_set = False

        # Time of Day Filter
        tod_active, st, et = self.state.filters['time_of_day']
        self.start_time.setTime(QTime(st.hour, st.minute, st.second))
        self.end_time.setTime(QTime(et.hour, et.minute, et.second))

        # Date Range Filter
        dtr_active, sdt, edt = self.state.filters['datetime_range']
        self.start_datetime.setDateTime(
            QDateTime(sdt.year, sdt.month, sdt.day, sdt.hour, sdt.minute, sdt.second)
        )
        self.end_datetime.setDateTime(
            QDateTime(edt.year, edt.month, edt.day, edt.hour, edt.minute, edt.second)
        )

        # Sample Length Filter
        sl_active, slen = self.state.filters['sample_length']
        self.max_seconds.setValue(slen)

        # Codes Filter
        code = self.state.filters['codes']
        self.codes.setCurrentIndex(self.all_codes.index(code))

        # Set Check
        self.tod_active.setChecked(tod_active)
        self.dtr_active.setChecked(dtr_active)
        self.sl_active.setChecked(sl_active)

        self._update_set = True
