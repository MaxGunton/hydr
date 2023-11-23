# python standard library
import os
import datetime as dt
from typing import List, Dict, Optional
from collections import Counter
import soundfile as sf
import sys
from tqdm import tqdm
from enum import Enum, Flag, auto

# installed packages
import pandas as pd
import numpy as np

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import ANALYSIS_DELAY, TQDM_WIDTH
from hydr.utils import unique_items


class Status(Flag):
    MachineLabelled = 1
    Submitted = 2
    Skipped = 4
    Modified = 8
    Revisit = 16
    NoStatus = 32


class LoadMethod(Enum):
    Fresh = auto()
    Refresh = auto()
    AutoSet = auto()


class Sample:
    _start = None
    _end = None
    _peak = None
    _code = None
    _comment = None

    def __init__(self, start=None, end=None, code=None, comment=None, peak=None):
        self._start = start
        self._end = end
        self._code = code
        self._comment = comment
        self._peak = peak

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def peak(self):
        return self._peak

    @property
    def code(self):
        return self._code

    @property
    def comment(self):
        return self._comment

    @start.setter
    def start(self, start):
        self._start = start

    @end.setter
    def end(self, end):
        self._end = end

    @code.setter
    def code(self, code):
        self._code = code

    @comment.setter
    def comment(self, comment):
        self._comment = comment

    @peak.setter
    def peak(self, peak):
        self._peak = peak


class Classifications:
    _validations = None

    def __init__(self):
        self._all_df = pd.DataFrame()

    @property
    def all_df(self):
        return self._all_df

    @property
    def labels(self):
        return unique_items(self._all_df['code']) if not self.all_df.empty else []

    @property
    def sort_order(self):
        so = [(f'{i}_confidence', False) for i in self.labels if i != 'background']
        if 'background' in self.labels:
            so += [('background_confidence', True)]
        return so

    @all_df.setter
    def all_df(self, all_df):
        self._all_df = all_df

    @property
    def validations(self):
        return self.get_validations()

    @validations.setter
    def validations(self, validations):
        self._validations = validations

    def classifications_to_validations(self, start, end):
        c = self.clean_df(self.all_df, start, end, self.sort_order)
        return c

    def get_validations(self, start=None, end=None, check_for_missing=False):
        c = self.classifications_to_validations(start, end)
        # print(c.columns)

        if self._validations is None:
            # print('Setting validations from classifications')
            self._validations = c
            # self._validations = self._validations.reset_index(drop=True)
            # print(self._validations.columns)
            # self._validations.to_csv('test.csv')
            return self._validations

        if check_for_missing:
            idx = {i: c.columns.get_loc(i) for i in ['file', 'start', 'end', 'model']}
            to_add = []

            # Check if validations contains each row of classifications by looking for
            # row with matching: `file`, `start`, `end`, and `model` values
            for i in tqdm(range(c.shape[0])):
                row = c.iloc[i, [idx['file'], idx['start'], idx['end'], idx['model']]]
                if self._validations[
                    np.logical_and(
                        np.logical_and(self._validations['file'] == row['file'],
                                       self._validations['start'] == row['start']),
                        np.logical_and(self._validations['end'] == row['end'],
                                       self._validations['model'] == row['model'])
                    )
                ].empty:
                    to_add.append(row)
            df = pd.concat(to_add)
            self._validations = self._validations.append(df)
            # self._validations = self.sort_df(self._validations, self.sort_order)
            self._validations = self.clean_df(
                self._validations,
                start,
                end,
                self.sort_order
            )
        # self._validations = self.sort_df(self._validations, self.sort_order)
        return self._validations


    @staticmethod
    def clean_df(df, start, end, sort_order):
        # create copies of all variable we will work with
        df = df.copy()
        # convert global start and global end to datetime objects if strings
        if start is not None:
            # add buffer to start (allows deployment noises to settle)
            start += dt.timedelta(seconds=ANALYSIS_DELAY)
            # if type(df.iloc[0, df.columns.get_loc('global_start')]) is str:
            #     df['global_start'] = df['global_start'].apply(str_to_dt)
            df = df[df['global_start'] >= start]  # samples that start after start
        if end is not None:
            # if type(df.iloc[0, df.columns.get_loc('global_end')]) is str:
            #     df['global_end'] = df['global_end'].apply(str_to_dt)
            df = df[df['global_end'] <= end]  # samples that end before end
        # combine adjacent
        df = Classifications.combine_adjacent_df(df)
        # order for validation
        df = Classifications.sort_df(df, sort_order)
        return df

    @staticmethod
    def sort_df(df, so):
        # 1) copy input `pd.DataFrame`
        df = df.copy()

        # 2) so = [(by=<column_name>, ascending=<bool>)]
        dfs = [df[df["code"] == c[:-11]].sort_values(by=c, ascending=a) for c, a in so]
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def combine_adjacent_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        This makes the assumption that detection dataframes contain file, start,
        end, and code columns (probably safe to say these are standard)
        Creates a copy of the input `pd.DataFrame` and combines entries that either overlap
        or occur without time gap between them AND share the same code.

        Returns copy of the input `pd.DataFrame` combining the adjacent & shared code
        entries.
        """

        # 1) create a copy of df_in
        df_orig = df.copy()

        # 2) separate df_orig into list of DataFrames with each unique code/file combo
        dfs = [
            df_orig.loc[np.logical_and(df_orig["code"] == c, df_orig["file"] == f), :]
            for c in set(df_orig['code'])
            for f in set(df_orig.loc[df_orig['code'] == c, "file"])
        ]
        assert (df_orig.shape[0] == sum(
            [df.shape[0] for df in dfs]))  # assert no lost rows
        dfs_new = []
        for df in dfs:  # for each unqiue code file combo
            # sort chronologically
            df = df.sort_values(by='start', axis=0, ascending=True, kind='quicksort')
            # column `m` indicates adjacent entries overlap
            df['m'] = (df['start'].shift(-1) - df['end'] <= 0)
            # drop middle entries from overlapping sections
            df = df[df['m'].shift(+1) + df['m'] != 2]
            # column `n` contains end times for overlapping sections
            df['n'] = df['end'].shift(-1)
            # assign `n` value to `end` if overlaps
            df.loc[df['m'], 'end'] = df.loc[df['m'], 'n']
            # `o` indicates non overlapping entries
            df['o'] = df['m'].shift(+1) + df['m'] != 1
            # keep the non overlapping and starts of overlaps
            df = df[df['o'] | df['m']]
            # reset the indexes
            df.reset_index(drop=True, inplace=True)
            # drop the computational rows
            df = df.drop(columns=['m', 'n', 'o'])
            # append the updated dataframe
            dfs_new.append(df)
        df_new = pd.concat(dfs_new, ignore_index=True)  # combine and return
        df_new['global_end'] = df_new.apply(
            lambda r: r['global_start'] + dt.timedelta(seconds=r['end'] - r['start']),
            axis=1
        )  # adjust the values for global_end
        return df_new


class WavFile:
    _file = None
    _sr = None
    _frames = None
    _duration = None
    _corrupted = False

    _start = None
    _end = None
    _utc_offset = None
    _gain = None

    def __init__(self, file):
        self._file = os.path.abspath(file)
        try:
            with sf.SoundFile(self._file) as wavobj:
                self._sr = wavobj.samplerate
                self._frames = wavobj.frames
            self._duration = dt.timedelta(seconds=self.frames/self.sr)
        except sf.LibsndfileError:
            self._corrupted = True

    @property
    def file(self):
        return self._file

    @property
    def frames(self):
        return self._frames

    @property
    def sr(self):
        return self._sr

    @property
    def duration(self):
        return self._duration

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._start + self._duration

    @property
    def utc_offset(self):
        return self._utc_offset

    @property
    def corrupted(self):
        return self._corrupted

    @property
    def gain(self):
        return self._gain

    @start.setter
    def start(self, start):
        self._start = start

    @utc_offset.setter
    def utc_offset(self, utc_offset):
        self._utc_offset = utc_offset

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    def set_extra_details(self, **kwargs):
        if 'start' in kwargs:
            self.start = kwargs['start']
        if 'utc_offset' in kwargs:
            self.utc_offset = kwargs['utc_offset']
        if 'gain' in kwargs:
            self.gain = kwargs['gain']


class Hydrophone:
    _sn = None
    _files = None
    _ext_counts = None
    _deployment_start = None
    _deployment_end = None

    _all_wavs = None

    _lat = None
    _lon = None
    _syncs = None

    def __init__(self, sn: str, files: List[str]) -> None:
        self._sn = sn
        self._files = [os.path.abspath(f) for f in files]
        wavs = [f for f in self.files if f[-3:] in ['wav', 'WAV']]
        self._all_wavs = [
            WavFile(f)
            for f in tqdm(wavs, desc='processing wav files'.ljust(TQDM_WIDTH))
        ]

    @property
    def sn(self) -> str:
        return self._sn

    @property
    def lat(self) -> Optional[float]:
        return self._lat

    @property
    def lon(self) -> Optional[float]:
        return self._lon

    @property
    def lat_s(self) -> Optional[str]:
        if self._lat is None:
            return None
        else:
            lat = self._lat
            lat_deg = int(abs(lat) // 1)
            lat_min = int(((abs(lat) % 1) * 60) // 1)
            lat_sec = (((abs(lat) % 1) * 60) % 1) * 60
            lat_deg = lat_deg * -1 if lat < 0 else lat_deg
            return f'{lat_deg}\u00b0 {lat_min}\u02b9 {lat_sec:.2f}"'

    @property
    def lon_s(self) -> Optional[str]:
        if self._lon is None:
            return None
        else:
            lon = self._lon
            lon_deg = int(abs(lon) // 1)
            lon_min = int(((abs(lon) % 1) * 60) // 1)
            lon_sec = (((abs(lon) % 1) * 60) % 1) * 60
            lon_deg = lon_deg * -1 if lon < 0 else lon_deg
            return f'{lon_deg}\u00b0 {lon_min}\u02b9 {lon_sec:.2f}"'


    @property
    def files(self) -> List[str]:
        return self._files

    @property
    def all_wavs(self) -> List[WavFile]:
        return self._all_wavs

    @property
    def relevant_wavs(self) -> List[WavFile]:
        start = (
            self.audio_start
            if self.deployment_start is None
            else self.deployment_start
        )
        end = self.audio_end if self.deployment_end is None else self.deployment_end
        ret = [
            w for w in self.all_wavs
            if not (
                    (w.start < start and w.end < start)
                    or
                    (w.start > end and w.end > end)
            )
        ]
        return ret

    @property
    def unused_wavs(self):
        return [w.file for w in self.all_wavs if w not in self.relevant_wavs]

    @property
    def wav_durations(self):
        return [w.duration
                for w in self.relevant_wavs
                if w.duration is not None]

    @property
    def total_audio(self):
        ds = self.deployment_start
        de = self.deployment_end
        rw = self.relevant_wavs
        firsts = [
            (w.duration - (ds - w.start)).total_seconds()
            for w in rw
            if (w.start < ds < w.end)
        ]
        betweens = [
            (w.duration).total_seconds()
            for w in rw
            if not (w.start < ds < w.end) and not (w.start < de < w.end)
        ]
        lasts = [
            (w.duration - (w.end - de)).total_seconds()
            for w in rw
            if (w.start < de < w.end)
        ]
        return dt.timedelta(seconds=sum(firsts + betweens + lasts))

    @property
    def audio_start(self):
        return min([w.start for w in self.all_wavs])

    @property
    def audio_end(self):
        return max([w.end for w in self.all_wavs])

    @property
    def deployment_start(self):
        return (
            self.audio_start
            if self._deployment_start is None
            else self._deployment_start
        )

    @property
    def deployment_end(self):
        return self.audio_end if self._deployment_end is None else self._deployment_end

    @property
    def relevant_gaps(self):
        times = sorted([
            (w.file, w.start, w.end)
            for w in self.relevant_wavs
            if w.start is not None
        ],
            key=lambda x: x[0]
        )
        return self.compute_gaps(times)

    @property
    def all_gaps(self):
        times = sorted([
            (w.file, w.start, w.end)
            for w in self.all_wavs
            if w.start is not None
        ],
            key=lambda x: x[0]
        )
        return self.compute_gaps(times)

    @property
    def bad_wavfiles(self):
        return [w.file for w in self.relevant_wavs if w.corrupted]

    @property
    def bad_logfiles(self):
        return [
            f'{w.file[:-4]}.log.xml'
            for w in self.relevant_wavs
            if w.utc_offset is None
        ]

    @property
    def ext_counts(self) -> Dict[str, int]:
        return self._ext_counts

    @property
    def sr_counts(self):
        return dict(Counter([w.sr for w in self.relevant_wavs]))

    @property
    def utc_offset_counts(self) -> Dict[str, int]:
        return dict(
            Counter(
                [
                    w.utc_offset
                    for w in self.relevant_wavs
                    if w.utc_offset is not None
                ]
            )
        )

    @property
    def gain_counts(self) -> Dict[str, int]:
        return dict(Counter([w.gain for w in self.relevant_wavs if w.gain is not None]))

    @lat.setter
    def lat(self, lat):
        if -90 <= lat <= 90:
            self._lat = lat

    @lon.setter
    def lon(self, lon):
        if -180 <= lon <= 180:
            self._lon = lon

    @ext_counts.setter
    def ext_counts(self, ext_counts):
        self._ext_counts = ext_counts

    @deployment_start.setter
    def deployment_start(self, deployment_start):
        self._deployment_start = (
            self.audio_start
            if deployment_start == "*"
            else deployment_start
        )

    @deployment_end.setter
    def deployment_end(self, deployment_end):
        self._deployment_end = (
            self.audio_end
            if deployment_end == "*"
            else deployment_end
        )

    @staticmethod
    def compute_gaps(times):
        files = [f for f, _, _ in times]
        starts = [s for _, s, _ in times]
        starts = starts[1:] + [starts[0]]
        ends = [e for _, _, e in times]

        times = np.stack([starts, ends], axis=0)
        gaps = np.diff(times, axis=0).astype("timedelta64[ns]").astype(float).flatten().tolist()
        gaps[-1] = None
        gaps = [
            (files[i], g * 1e-9)
            if g is not None
            else (files[i], None)
            for i, g in enumerate(gaps)
        ]
        return gaps


class Deployment:
    _convention = None
    _hydrophones = None
    _classifications = None

    def __init__(self, convention=None):
        self._convention = convention
        self._hydrophones = {}
        self._classifications = Classifications()

    def add_hydrophone(self, hydrophone: Hydrophone) -> None:
        self._hydrophones[hydrophone.sn] = hydrophone

    def remove_hydrophone(self, sn: str) -> None:
        self._hydrophones.pop(sn)

    @property
    def convention(self) -> str:
        return self._convention

    @property
    def hydrophones(self) -> Dict[str, Hydrophone]:
        return self._hydrophones

    @property
    def sns(self) -> List[str]:
        return [sn for sn in self.hydrophones]

    @property
    def srs(self) -> List[int]:
        return unique_items([sr for h in self.hydrophones.values() for sr in h.sr_counts])

    @property
    def deployment_start(self):
        return min([h.deployment_start for sn, h in self.hydrophones.items()])

    @property
    def deployment_end(self):
        return max([h.deployment_end for sn, h in self.hydrophones.items()])

    @property
    def classifications(self):
        return self._classifications.all_df

    @property
    def validations(self):
        return self._classifications.get_validations(self.deployment_start,
                                                     self.deployment_end)

    @validations.setter
    def validations(self, validations):
        self._classifications.validations = validations

    def append_classifications(self, df: pd.DataFrame) -> None:
        self._classifications.all_df = self._classifications.all_df.append(
            df, ignore_index=True)

