import sys
import os
import wave
import pyaudio
from threading import Thread

from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QFrame, QPushButton

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import (STEP_BACK_ICON, PLAY_ICON, PAUSE_ICON, STOP_ICON,
                              STEP_AHEAD_ICON, BLOCKSIZE, AUDIO_STEP)
from hydr.validator.signals import Receiver


class AudioControls(QFrame, Receiver):
    _play_position = None
    _playhead_start = None

    def __init__(self, parent):
        QFrame.__init__(self, parent)
        Receiver.__init__(self)
        self.layout = QHBoxLayout(self)
        self.setLineWidth(1)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.state.playing = False

        self.play_icon = QIcon(PLAY_ICON)
        self.pause_icon = QIcon(PAUSE_ICON)

        self.step_back = QPushButton(QIcon(STEP_BACK_ICON), '', self)
        self.step_back.setMaximumHeight(25)
        self.step_back.setMinimumWidth(50)
        self.play_pause = QPushButton(self.play_icon, '', self)
        self.play_pause.setMaximumHeight(25)
        self.play_pause.setMinimumWidth(50)
        self.stop = QPushButton(QIcon(STOP_ICON), '', self)
        self.stop.setMaximumHeight(25)
        self.stop.setMinimumWidth(50)
        self.step_ahead = QPushButton(QIcon(STEP_AHEAD_ICON), '', self)
        self.step_ahead.setMaximumHeight(25)
        self.step_ahead.setMinimumWidth(50)

        self.step_back.clicked.connect(self.step_back_pressed)
        self.play_pause.clicked.connect(self.play_pause_pressed)
        self.stop.clicked.connect(self.stop_pressed)
        self.step_ahead.clicked.connect(self.step_ahead_pressed)

        self.layout.addWidget(self.step_back)
        self.layout.addWidget(self.play_pause)
        self.layout.addWidget(self.stop)
        self.layout.addWidget(self.step_ahead)
        self.layout.addStretch(1)

    @property
    def playhead_start(self):
        # return (
        #     self._playhead_start
        #     if self._playhead_start is not None
        #     else self.state.new_samples[0].start
        # )
        return self.state.new_samples[0].start

    # @playhead_start.setter
    # def playhead_start(self, playhead_start):
    #     self._playhead_start = playhead_start

    @Slot()
    def step_back_pressed(self):
        new_position = self.state.playhead - AUDIO_STEP
        self.state.playhead = max(new_position, self.state.current_sample['start'])

    @Slot()
    def play_pause_pressed(self):
        self.state.playing = not self.state.playing
        if self.state.playing:
            self.play_pause.setIcon(self.pause_icon)
            audio_thread = Thread(target=self.play_audio)
            audio_thread.start()
        else:
            self.play_pause.setIcon(self.play_icon)

    @Slot()
    def stop_pressed(self):
        self.state.playing = False
        self.state.playhead = self.playhead_start
        self.play_pause.setIcon(self.play_icon)

    @Slot()
    def step_ahead_pressed(self):
        new_position = self.state.playhead + AUDIO_STEP
        self.state.playhead = min(new_position, self.state.current_sample['end'])

    # def on_playhead_updated(self, _):
    #     if not self.state.playing:
    #         self.playhead_start = self.state.playhead

    # TODO: Try to remove the slight lag in playline to audio (probably done by
    #       optimizing Spectrogram.draw_vlines method)
    def play_audio(self):
        file = self.state.current_sample['file']
        end = self.state.current_sample['end']

        wf = wave.open(file, 'rb')
        sr = wf.getframerate()
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=sr,
                        output=True)

        while self.state.playing:
            self._play_position = self.state.playhead
            start_f = int(round(self.state.playhead * sr))
            end_f = int(round(end * sr))
            wf.setpos(start_f)
            total = 0
            while total < (end_f - start_f) and self.state.playing:
                if self._play_position != self.state.playhead:
                    break
                if start_f + total + BLOCKSIZE < end_f:
                    data = wf.readframes(BLOCKSIZE)
                    total += BLOCKSIZE
                    self._play_position += BLOCKSIZE / sr
                    self.state.playhead += BLOCKSIZE / sr
                    stream.write(data)

                else:
                    final_block_size = end_f - start_f - total
                    data = wf.readframes(final_block_size)
                    total += final_block_size
                    self._play_position += final_block_size / sr
                    self.state.playhead += final_block_size / sr
                    stream.write(data)
            else:
                if total >= end_f - start_f:
                    self.state.playing = False
                    self.state.playhead = self.playhead_start
        self.play_pause.setIcon(self.play_icon)

        # self._play_position = None
        stream.close()
        p.terminate()

    def on_load_sample(self, _, how):
        self.disable(True if self.state.index is None else False)

    def disable(self, disable):
        self.step_back.setDisabled(disable)
        self.play_pause.setDisabled(disable)
        self.stop.setDisabled(disable)
        self.step_ahead.setDisabled(disable)


