import datetime
import sys
import os
import colorsys
import datetime as dt

import numpy as np
from nnAudio.features import STFT
import soundfile as sf
import torch
from matplotlib.colors import LogNorm
from matplotlib import cm as colormap
from PIL import Image, ImageQt

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QMouseEvent, QKeyEvent, QFont
from PySide6.QtWidgets import QLabel, QApplication, QFrame, QHBoxLayout, QVBoxLayout

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.utils import secs_to_frames, unique_items
from hydr.definitions import (DISPLAY_SPEC_PARAMS, PEAK_SPEC_PARAMS, NUM_COLORS,
                                    LINE_THICKNESS, DASH_LENGTH, DASH_GAP, NO_SAMPLE,
                                    VALIDATOR_COLORS, STATUS_COLORS)
from hydr.types import Status, LoadMethod, Sample
from hydr.validator.signals import Receiver
from hydr.validator.misc import QHLine


# TODO: Convert pixels to seconds on export
# TODO: Add a little bound label beside the start of each bound with a number
# TODO: Make bounds selectable for removal
# TODO: Have additional peaks added when multiple bound (how to do this?)
# TODO: Draw previous bounds and peaks if any
# TODO: Implement logic to deal with multiple peaks
# TODO: Make sure spectrogram is always a standard size (i.e. resize to some defined
#       dimensions)
# TODO: Add the ability to drag bounds and peaks
class SpectrogramFrame(QFrame, Receiver):

    _colors = None

    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        Receiver.__init__(self)
        self.setObjectName('SpectrogramFrame')
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)  # width of color around spectrogram
        self.layout.setSpacing(0)

        label_font = QFont()
        label_font.setBold(True)
        label_font.setPointSize(9)

        date_font = QFont()
        date_font.setBold(True)
        date_font.setPointSize(11)

        self.details = QFrame(self)
        self.details.layout = QHBoxLayout(self.details)
        self.details.layout.setContentsMargins(0, 0, 0, 4)

        self.sample_details = QFrame(self.details)
        self.sample_details.setObjectName('sample_details')
        self.sample_details.layout = QHBoxLayout(self.sample_details)
        self.sample_details.layout.setContentsMargins(0, 0, 0, 0)

        self.codes_label = QLabel('Code(s):', self.sample_details)
        self.codes_label.setFont(label_font)
        self.codes = QLabel('', self.sample_details)
        self.status_label = QLabel('Status:', self.sample_details)
        self.status_label.setFont(label_font)
        self.status = QLabel('', self.sample_details)
        self.peaks_label = QLabel('Peak(s):', self.sample_details)
        self.peaks_label.setFont(label_font)
        self.peaks = QLabel('', self.sample_details)
        self.sample_details.layout.addWidget(self.codes_label)
        self.sample_details.layout.addWidget(self.codes)
        self.sample_details.layout.addSpacing(20)
        self.sample_details.layout.addWidget(self.status_label)
        self.sample_details.layout.addWidget(self.status)
        self.sample_details.layout.addSpacing(20)
        self.sample_details.layout.addWidget(self.peaks_label)
        self.sample_details.layout.addWidget(self.peaks)

        self.duration_details = QFrame(self.details)
        self.duration_details.setObjectName('duration_details')
        self.duration_details.layout = QHBoxLayout(self.duration_details)
        self.duration_details.layout.setContentsMargins(0, 0, 0, 0)
        self.duration_label = QLabel('Duration:', self.duration_details)
        self.duration_label.setFont(label_font)
        self.duration = QLabel('', self.duration_details)
        self.duration_details.layout.addWidget(self.duration_label)
        self.duration_details.layout.addWidget(self.duration)

        self.details.layout.addWidget(self.sample_details)
        self.details.layout.addStretch(1)
        self.details.layout.addWidget(self.duration_details)

        self.time_details = QFrame(self)
        # self.time_details.setLineWidth(1)
        # self.time_details.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.time_details.layout = QHBoxLayout(self.time_details)
        self.time_details.layout.setContentsMargins(0, 0, 0, 0)
        self.start_time = QLabel('', self.time_details)
        self.start_time.setFont(label_font)
        self.start_time.setObjectName('start_time')
        self.start_time.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.date = QLabel('', self.time_details)
        self.date.setObjectName('date')
        self.date.setFont(date_font)
        self.date.setAlignment(Qt.AlignCenter | Qt.AlignBottom)
        self.end_time = QLabel('', self.time_details)
        self.end_time.setFont(label_font)
        self.end_time.setObjectName('end_time')
        self.end_time.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.time_details.layout.addWidget(self.start_time)
        self.time_details.layout.addStretch(1)
        self.time_details.layout.addWidget(self.date)
        self.time_details.layout.addStretch(1)
        self.time_details.layout.addWidget(self.end_time)

        self.spectrogram = Spectrogram(self)

        self.layout.addWidget(self.details)
        self.layout.addWidget(self.time_details)
        self.layout.addWidget(self.spectrogram)
        self.on_load_sample(self, LoadMethod.Fresh)

    @property
    def colors(self):
        return STATUS_COLORS

    def on_load_sample(self, _, how):
        s_sec = self.state.current_sample['start']
        s, e = (
            self.state.current_sample['global_start'],
            self.state.current_sample['global_end']
        )
        if how == LoadMethod.Fresh:
            start = 'None' if s is None else s.strftime("%H:%M:%S")
            end = 'None' if e is None else e.strftime("%H:%M:%S")
            date = 'None' if s is None else s.strftime("%Y-%m-%d")
            d = (e - s).total_seconds() if s is not None and e is not None else None
            d = '{:.1f} seconds'.format(d) if d is not None else 'None'
            self.duration.setText(d)
            self.start_time.setText(start)
            self.date.setText(date)
            self.end_time.setText(end)

        status = (
            self.state.current_sample['val_status']
            if self.state.current_sample['val_status'] is not None
            else Status.NoStatus
        )
        codes = unique_items(
            [i.code for i in self.state.new_samples if i.code is not None]
        )
        peaks = [i.peak-s_sec for i in self.state.new_samples if i.peak is not None]
        peaks = [
            (s + dt.timedelta(seconds=p)).strftime('%H:%M:%S.%f')
            if s is not None
            else '{:.6f}'.format(p) for p in peaks
        ]
        self.peaks.setText(', '.join(peaks))
        self.codes.setText(', '.join([f'`{i}`' for i in codes]))
        self.status.setText(str(status).split('.')[-1])
        c = self.brighten_color(self.colors[str(status)])
        self.sample_details.setStyleSheet(
            "#sample_details {background-color: " + c + ";}"
        )
        self.duration_details.setStyleSheet(
            "#duration_details {background-color: " + c + ";}"
        )
        self.start_time.setStyleSheet(
            "#start_time {background-color: " + c + ";}"
        )
        self.date.setStyleSheet(
            "#date {background-color: " + c + ";}"
        )
        self.end_time.setStyleSheet(
            "#end_time {background-color: " + c + ";}"
        )
        self.setStyleSheet(
            "#SpectrogramFrame {background-color: " + self.colors[str(status)] + ";}"
        )

    @staticmethod
    def brighten_color(color):
        r, g, b = [
            int(i) / 255
            for i in color.replace(' ', '')[4:-1].split(',')
        ]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        new_color = colorsys.hsv_to_rgb(h, s*0.3, min(max(v*1.1, 0.6), 1.0))
        new_color = (
            'rgb(' +
            ', '.join([str(min(max(int(round(i*255)), 0), 255)) for i in new_color]) +
            ')'
        )
        return new_color


# TODO: Show the current code in the top left corner if it has been set
# TODO: Show the sample duration and date in the top right corner
# TODO: Have the option to hide these using a checkbox
class Spectrogram(QLabel, Receiver):
    _kernels = None
    _log_norm = None
    _no_spec = None
    _audio = None
    _sr = None
    _base_spec = None
    _peak_spec = None
    _initial_draw = None
    _single_click_last = None
    _last_mouse_event = None

    _cs = None  # dictionary containing all info about the current sample

    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        Receiver.__init__(self)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setScaledContents(True)

        # TODO: Figure out a good minimums to use
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        self._kernels = {
            sr: {
                'display': STFT(**{**DISPLAY_SPEC_PARAMS, **dict(sr=sr)}),
                'peak': STFT(**{**PEAK_SPEC_PARAMS, **dict(sr=sr)}),
            } for sr in self.state.deployment.srs
        }
        self._log_norm = LogNorm(clip=True)
        self._no_spec = np.append(
            np.array(Image.open(NO_SAMPLE).getdata()).reshape(300, 1000, 3),
            np.ones((300, 1000, 1)) * 255,
            axis=2
        ).astype(np.uint8)
        self._initial_draw = True
        self.on_load_sample(self, LoadMethod.Fresh)
        self.setFocus()

    @property
    def playhead(self):
        return self.secs_from_sof_to_xpixel(self.state.playhead)
    @property
    def cs(self):
        return self.state.current_sample
    @property
    def colors(self):
        return VALIDATOR_COLORS

    @playhead.setter
    def playhead(self, playhead):
        self.state.playhead = self.xpixel_to_secs_from_sof(playhead)

    def on_load_sample(self, _, how):
        if how == LoadMethod.Fresh:
            # Below will assign None if they can't do it
            self.get_audio()  # assign values to `self._audio` and `self._sr`
            self.get_base_spec()  # assign value to `self._base_spec`
            self.get_peak_spec()  # assign value to `self._peak_spec`
            self.playhead = self.secs_from_sof_to_xpixel(
                self.state.new_samples[0].start
            )
            self.setDisabled(True if self._base_spec is self._no_spec else False)
            # self.setFocus()
        elif how == LoadMethod.AutoSet:
            self.autoset_sample()
            # self.setFocus()
        elif how == LoadMethod.Refresh:
            self.playhead = self.playhead

    def get_audio(self):
        file, start, end = self.cs['file'], self.cs['start'], self.cs['end']
        self._audio, self._sr = None, None
        if file is not None and start is not None and end is not None:
            try:
                with sf.SoundFile(file) as wavobj:
                    self._sr = wavobj.samplerate
                    start_f = secs_to_frames(start, self._sr)
                    end_f = secs_to_frames(end, self._sr)
                    total_f = end_f - start_f
                    wavobj.seek(start_f)
                    audio = wavobj.read(total_f)
                self._audio = torch.tensor(audio).float()
            except sf.LibsndfileError:
                pass

    def get_base_spec(self):
        self._base_spec = self._no_spec
        if self._audio is not None and self._sr is not None:
            s = torch.flip(
                torch.squeeze(self._kernels[self._sr]['display'](self._audio)),
                (0,)
            ).numpy() ** 25
            s = self._log_norm(np.where(s == np.inf, np.finfo('float32').max, s))
            s = colormap.turbo(s)  # [:, :, :3]
            self._base_spec = (
                    (s - s.min()) / (s.max() - s.min()) * 255
            ).astype(np.uint8)

    def get_peak_spec(self):
        self._peak_spec = None
        if self._audio is not None and self._sr is not None:
            s = torch.flip(
                torch.squeeze(self._kernels[self._sr]['peak'](self._audio)),
                (0,)
            ).numpy() ** 25
            s = self._log_norm(np.where(s == np.inf, np.finfo('float32').max, s))
            s = s.sum(axis=0)
            self._peak_spec = s

    def compute_peak(self, start, end):  # seconds from start
        peak = None
        if (
                self._peak_spec is not None
                and start is not None
                and end is not None
                and start < end
        ):
            s = self._peak_spec.copy()
            seconds_per_pixel = (self.cs['end'] - self.cs['start']) / s.shape[0]
            start_p = int(round(start / seconds_per_pixel))
            end_p = int(round(end / seconds_per_pixel))
            s = s[start_p:end_p]
            s = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(11, dtype=int), 'valid'),
                0, s
            )
            window_width = 1001
            if s.shape[0] < window_width:
                peak = self.time_to_xpixel(
                    (start_p + (end_p - start_p)/2) * seconds_per_pixel
                )
            else:
                s = np.median(
                    np.lib.stride_tricks.sliding_window_view(s, window_width, axis=0),
                    axis=1
                )
                peak = np.gradient(s, 2, edge_order=1, axis=0).argmax(axis=0)
                # pixels relative to start of sample
                peak = self.time_to_xpixel(
                    (start_p + peak + (window_width / 2)) * seconds_per_pixel
                )
        return peak

    def autoset_sample(self):
        # This function should load a sample like it is new (i.e. never validated
        # before)
        start, end = 0, self.time_to_xpixel(self.cs['end'] - self.cs['start'])
        peak = self.compute_peak(0, self.cs['end']-self.cs['start'])
        if peak is not None:
            start = self.time_to_xpixel(
                max(
                    self.xpixel_to_time(peak) - 2,
                    self.cs['start'] - self.cs['start']
                )
            )
            end = self.time_to_xpixel(
                min(
                    self.xpixel_to_time(peak) + 5,
                    self.cs['end'] - self.cs['start']
                )
            )
        # self._start_history = [start]
        # self._end_history = [end]
        self.state.new_samples[0].start = self.xpixel_to_secs_from_sof(start)
        self.state.new_samples[0].end = self.xpixel_to_secs_from_sof(end)
        self.state.new_samples[0].peak = self.xpixel_to_secs_from_sof(peak)
        self.playhead = start
        self.state.current_sample['val_status'] = Status.Modified
        self.state.refresh()

    def on_playhead_updated(self, _):
        self.draw_vlines()

    def on_add_peak(self, _, index):
        start = self.state.new_samples[index].start
        end = self.state.new_samples[index].end
        peak = self.xpixel_to_secs_from_sof(
            self.compute_peak(
                start - self.cs['start'],
                end - self.cs['start']
            )
        )
        self.state.new_samples[index].peak = peak
        self.state.refresh()

    # TODO: Only update the necessary portions of the pixmap and see if we can avoid
    #       doing a full conversion from numpy array everytime we redraw
    def draw_vlines(self):
        s = np.copy(self._base_spec)
        lt = LINE_THICKNESS
        hlt = int(round(LINE_THICKNESS/2))
        active_sample = self.state.new_samples[-1]
        samples = self.state.new_samples[:-1]
        for i, sample in enumerate(samples):
            start = self.secs_from_sof_to_xpixel(sample.start)
            end = self.secs_from_sof_to_xpixel(sample.end)
            color = self.colors[i % NUM_COLORS]
            # SOLID LINES
            s[:, (start - lt):(start + lt), :] = color
            s[:, (end - lt):(end + lt), :] = color
            if sample.peak is not None:
                l1 = self.secs_from_sof_to_xpixel(sample.peak) - lt
                l2 = self.secs_from_sof_to_xpixel(sample.peak) + lt
                # DASHED LINES
                dash_step = (DASH_GAP + DASH_LENGTH)
                for k in range(2):
                    for j in range(dash_step*k, dash_step*k + DASH_LENGTH):
                        s[j::dash_step*2, l1:l2, :] = self.colors['peak'][k]
        if active_sample.start is not None:
            l1 = self.secs_from_sof_to_xpixel(active_sample.start) - lt
            l2 = self.secs_from_sof_to_xpixel(active_sample.start) + lt
            s[:, l1:l2, :] = self.colors['active']['start']
        if active_sample.end is not None:
            l1 = self.secs_from_sof_to_xpixel(active_sample.end) - lt
            l2 = self.secs_from_sof_to_xpixel(active_sample.end) + lt
            s[:, l1:l2, :] = self.colors['active']['end']
        if active_sample.peak is not None:
            l1 = self.secs_from_sof_to_xpixel(active_sample.peak) - lt
            l2 = self.secs_from_sof_to_xpixel(active_sample.peak) + lt
            # DASHED LINES
            dash_step = (DASH_GAP + DASH_LENGTH)
            for k in range(2):
                for j in range(dash_step * k, dash_step * k + DASH_LENGTH):
                    s[j::dash_step * 2, l1:l2, :] = self.colors['peak'][k]
        if (
                self.playhead is not None
                and self.playhead != self.secs_from_sof_to_xpixel(active_sample.start)
        ):
            l1, l2 = self.playhead - hlt, self.playhead + hlt
            s[:, l1:l2, :] = self.colors['playline']
        # NOTE: After initial draw we always want the spectrogram to match the size of
        #       previously one.  This prevents variations in spectrogram images due to
        #       varying sample lengths
        if self._initial_draw:
            self._initial_draw = False
            self.setPixmap(self.numpy_to_qpixmap(s))
        self.setPixmap(self.numpy_to_qpixmap(s).scaled(self.pixmap().size()))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        self._single_click_last = True
        button = event.buttons()
        xpos = int(
            round(
                event.position().x() * self._base_spec.shape[1] / self.width()
            )
        )
        mods = QApplication.keyboardModifiers()
        QTimer.singleShot(
            QApplication.instance().doubleClickInterval(),
            lambda: self.single_click(xpos, button, mods)
        )

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self._single_click_last = False
        self.playhead = int(
            round(
                event.position().x() * self._base_spec.shape[1] / self.width()
            )
        )

    def single_click(self, xpos, button, mods):
        if self._single_click_last:
            modified = False
            if button is Qt.MouseButton.LeftButton:
                if mods == Qt.ShiftModifier:
                    self.state.new_samples[-1].peak = self.xpixel_to_secs_from_sof(xpos)
                else:
                    self.state.new_samples[-1].start = self.xpixel_to_secs_from_sof(xpos)
                    if not self.state.playing:
                        self.playhead = xpos
                modified = True
            elif button is Qt.MouseButton.RightButton:
                self.state.new_samples[-1].end = self.xpixel_to_secs_from_sof(xpos)
                modified = True
            if modified:
                self.state.current_sample['val_status'] = Status.Modified
            start = self.state.new_samples[-1].start
            end = self.state.new_samples[-1].end
            peak = self.state.new_samples[-1].peak
            if peak is not None and not (start <= peak <= end):
                self.on_add_peak(self, -1)
            self.state.refresh()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        start, end = self.state.new_samples[-1].start, self.state.new_samples[-1].end
        if event.key() == 32:  # spacebar was pressed while focus on spectrogram
            if start is not None and end is not None:
                start, end = (end, start) if start > end else (start, end)
                self.state.new_samples[-1].start = start
                self.state.new_samples[-1].end = end
                self.state.current_sample['val_status'] = Status.Modified
                self.state.new_samples += [
                    Sample(
                        start=None,
                        end=None,
                        peak=None,
                        code=self.state.new_samples[-1].code,
                        comment=self.state.new_samples[-1].comment
                    )
                ]
                self.playhead = self.cs['start']
                self.state.refresh()

    def xpixel_to_secs_from_sof(self, xpixel):
        return (
            None
            if self.cs['start'] is None or xpixel is None
            else self.xpixel_to_time(xpixel) + self.cs['start']
        )

    def secs_from_sof_to_xpixel(self, seconds):
        return (
            None
            if self.cs['start'] is None or seconds is None
            else self.time_to_xpixel(seconds - self.cs['start'])
        )

    def time_to_xpixel(self, t):
        w = self._base_spec.shape[1] - 1
        d = (self.cs['end'] - self.cs['start'])
        pixels_per_second = w / d
        return int(round(t * pixels_per_second))

    def xpixel_to_time(self, x):
        w = self._base_spec.shape[1] - 1
        d = (self.cs['end'] - self.cs['start'])
        seconds_per_pixel = d / w
        return x * seconds_per_pixel

    @staticmethod
    def numpy_to_qpixmap(array):
        image = ImageQt.ImageQt(Image.fromarray(array))
        return QPixmap.fromImage(image)
