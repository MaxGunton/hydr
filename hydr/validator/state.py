import sys
import os

import pandas as pd
from blinker import signal
import numpy as np
import datetime as dt
# import soundfile as sf

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.utils import singleton, unique_items, load_depfile, save_depfile
from hydr.definitions import ANALYSIS_DELAY  # , VALIDATOR_SAMPLE_BUFFER
from hydr.types import Status, Sample, LoadMethod
from hydr.validator.signals import Connector, Signals


# TODO: load_current_sample, clear_sample, and assigning to self.new_samples trigger
#       the on_index_updated signal

# TODO: Load any samples that overlap with the loaded sample (i.e. if they were
#       classified differently)
@singleton
class State(Connector):
    """
    Since this is a singleton everyone will have the same copy; therefore, data defining
    gui state will always be consistent for all widgets at any given time.  We just need
    to ensure that the widgets all grab the data at the "Same" times.  This can be
    accomplished by the use of two signals:

      1. state_updated (triggers widgets to look at state and see if the change affects
                        them)

    """
    _depfile = None
    _deployment = None
    _df_all = None
    _df_set = None
    _model = None
    _sn = None
    _filters = None

    _playhead = None  # int
    _playing = None

    _index = None  # int
    _current_sample = None  # FIXME: Make it impossible to change this value
    _new_samples = None  # List[Samples]

    _shistory = None  # List of indexes
    _shistory_idx = None  # int
    _shistidx_to_rownum = None
    _rownum_to_shistidx = None

    def __init__(self, depfile, app):
        super().__init__()
        self._depfile = depfile
        self._deployment = load_depfile(depfile)
        self._app = app
        self._df_all = self.deployment.validations
        self._model = self.all_models[0]
        self._sn = self.all_sns[0]
        # filters take the form: (<active>, <value>, ..., <value>)
        self._filters = {
            'time_of_day': (True, dt.time(hour=4), dt.time(hour=20)),
            'datetime_range': (
                True,
                self.deployment.deployment_start,
                self.deployment.deployment_end,
            ),
            'sample_length': (True, 180),
            'codes': 'Any'
        }
        self.update_df_set()

    @property
    def depfile(self):
        return self._depfile

    @property
    def deployment(self):
        return self._deployment

    @property
    def df_all(self):
        return self._df_all

    @property
    def df_set(self):
        return self._df_set

    @property
    def model(self):
        return self._model

    @property
    def all_models(self):
        return unique_items(self._df_all['model'])

    @property
    def sn(self):
        return self._sn

    @property
    def all_sns(self):
        return unique_items(self._df_all['sn'])

    @property
    def filters(self):
        return self._filters

    @property
    def index(self):
        return self._index

    @property
    def playhead(self):
        return self._playhead

    @property
    def current_sample(self):
        return self._current_sample

    @property
    def new_samples(self):
        return self._new_samples

    @property
    def shistory(self):
        return self._shistory

    @property
    def shistory_idx(self):
        return self._shistory_idx

    @property
    def playing(self):
        return self._playing

    @property
    def shistidx_to_rownum(self):
        return self._shistidx_to_rownum

    @property
    def rownum_to_shistidx(self):
        return self._rownum_to_shistidx


    @model.setter
    def model(self, model):
        self._model = model
        self.update_df_set()

    @sn.setter
    def sn(self, sn):
        self._sn = sn
        # update the date range to the one for the specific hydrophone
        # TODO: Put an option to set to deployment bounds in filters widget as well as
        #       an option to add a delay
        sdt = (self.deployment.hydrophones[self._sn].deployment_start +
               dt.timedelta(seconds=ANALYSIS_DELAY))
        edt = self.deployment.hydrophones[self._sn].deployment_end
        dtr_active, _, _ = self.filters['datetime_range']
        self.filters['datetime_range'] = (dtr_active, sdt, edt)
        self.update_df_set()

    @filters.setter
    def filters(self, filters):
        self._filters = filters
        self.update_df_set()

    @playhead.setter
    def playhead(self, playhead):
        self._playhead = playhead
        signal(Signals.playhead_updated).send(self)

    @index.setter
    def index(self, index):
        self._index = index
        self.playing = False
        self.load_current_sample()

    @new_samples.setter
    def new_samples(self, new_samples):
        if new_samples is None or new_samples == []:
            new_samples = [
                Sample(
                    start=self.current_sample['start'],
                    end=self.current_sample['end'],
                    code=self.current_sample['code'],
                    comment='',
                    peak=None
                )
            ]
        self._new_samples = new_samples

    @playing.setter
    def playing(self, playing):
        self._playing = playing

    @shistidx_to_rownum.setter
    def shistidx_to_rownum(self, shistidx_to_rownum):
        self._shistidx_to_rownum = shistidx_to_rownum

    @rownum_to_shistidx.setter
    def rownum_to_shistidx(self, rownum_to_shistidx):
        self._rownum_to_shistidx = rownum_to_shistidx

    def on_get_state(self, _):
        signal(Signals.state).send(self, value=self)

    def update_df_set(self, reset_index=True):
        """
        This method should be called whenever sn, model, or filters are changed.
        """
        df = self.df_all[
            np.logical_and(
                self.df_all['model'] == self.model,
                self.df_all['sn'] == self.sn
            )
        ].copy()

        td_active, st, et = self.filters['time_of_day']
        dtr_active, sdt, edt = self.filters['datetime_range']
        sl_active, slen = self.filters['sample_length']
        codes = self.filters['codes']

        # TODO: Add logic to deal with samples where they start in one day and end in
        #       the next
        if td_active:
            start_times = df['global_start'].apply(
                lambda x: dt.time(hour=x.hour, minute=x.minute, second=x.second)
            )
            end_times = df['global_end'].apply(
                lambda x: dt.time(hour=x.hour, minute=x.minute, second=x.second)
            )
            df = df[np.logical_and(start_times >= st, end_times <= et)]
        if dtr_active:
            df = df[np.logical_and(df['global_start'] >= sdt, df['global_end'] <= edt)]
        if sl_active:
            df = df[(df['end']-df['start']) <= slen]
        if codes != 'Any':
            df = df[df['code'] == codes]

        self._df_set = df
        self._shistory = self.df_set.index.tolist()
        self._shistidx_to_rownum = [i for i in range(self.df_set.shape[0])]
        self._rownum_to_shistidx = self._shistidx_to_rownum.copy()
        # self._index_mapping = [(i, i) for i in self._shistory]
        signal(Signals.set_updated).send(self)
        if reset_index:
            self._index = None
            self._shistory_idx = None
            self.update_index(Status.Modified | Status.MachineLabelled)
        else:
            self.set_index(self.index)

    def load_current_sample(self):
        cols = ['file', 'global_start', 'global_end', 'start', 'end', 'code',
                'val_status', 'val_samples']
        col_idxs = [self.df_all.columns.get_loc(c) for c in cols]
        if self.index is not None:
            r = self.df_all.iloc[self.index, col_idxs].copy()
            # with sf.SoundFile(r['file']) as wo:
            #     buffer = VALIDATOR_SAMPLE_BUFFER
            #     r['start'] = max(r['start'] - buffer, 0)
            #     r['end'] = min(r['end'] + buffer, (wo.frames / wo.samplerate))
        else:
            r = pd.Series({c: None for c in cols})
        self._current_sample = {c: r[c] for c in cols[:-1]}
        self.new_samples = r['val_samples']
        # drop samples where start is None or end is None
        self.new_samples = [
            s
            for s in self.new_samples
            if not (s.start is None or s.end is None)
        ]
        signal(Signals.load_sample).send(self, how=LoadMethod.Fresh)

    # TODO: May want to set save=False if application seems slow
    def refresh(self, save=True):
        if save:
            self.save_changes()
        signal(Signals.load_sample).send(self, how=LoadMethod.Refresh)

    # TODO: May want to set save=False if application seems slow
    def autoset(self, save=True):
        if save:
            self.save_changes()
        signal(Signals.load_sample).send(self, how=LoadMethod.AutoSet)

    def update_index(self, include=Status.Modified | Status.MachineLabelled |
                                   Status.Revisit, forward=True):
        if self.shistory_idx is None:
            options = self.rownum_to_shistidx
        else:
            rn = self.shistidx_to_rownum[self.shistory_idx]
            options = (
                self.rownum_to_shistidx[rn+1:]
                if forward
                else self.rownum_to_shistidx[:rn]
            )
        pool = []
        for s in Status:
            if s in include:
                pool += [
                    i
                    for i, v in enumerate((self.df_set['val_status'] == s).tolist())
                    if v
                ]
        next_up = [i for i in options if i in pool]
        next_up = next_up if forward else [i for i in reversed(next_up)]
        if len(next_up) > 0:
            self._shistory_idx = next_up[0]
            self.index = self._shistory[self._shistory_idx]
        else:
            self._shistory_idx = None
            self.index = None

    def prev_index(self):
        if self.shistory:
            rn = (
                -1
                if self.shistory_idx is None
                else max(
                    self.shistidx_to_rownum[self.shistory_idx] - 1,
                    0
                )
            )
            self._shistory_idx = self.rownum_to_shistidx[rn]
            self.index = self.shistory[self._shistory_idx]
        else:
            self._shistory_idx = None
            self.index = None

    def next_index(self):
        if self.shistory:
            rn = (
                0
                if self.shistory_idx is None
                else min(
                    self.shistidx_to_rownum[self.shistory_idx] + 1,
                    len(self.shistidx_to_rownum) - 1
                )
            )
            self._shistory_idx = self.rownum_to_shistidx[rn]
            self.index = self.shistory[self._shistory_idx]
        else:
            self._shistory_idx = None
            self.index = None

    def first_index(self):
        if self.shistory:
            self._shistory_idx = self.rownum_to_shistidx[0]
            self.index = self.shistory[self._shistory_idx]
        else:
            self._shistory_idx = None
            self.index = None

    def last_index(self):
        if self.shistory:
            self._shistory_idx = self.rownum_to_shistidx[-1]
            self.index = self.shistory[self.shistory_idx]
        else:
            self._shistory_idx = None
            self.index = None

    def set_index(self, index):
        # print(self.df_all.index.tolist())
        if index not in self.df_all.index.tolist():
            raise IndexError
        self._shistory_idx = (
            None
            if index not in self.shistory
            else self.shistory.index(index)
        )
        if self._shistory_idx is None:
            self._index = index
            self._model = self.df_all.iloc[index, :]['model']
            self._sn = self.df_all.iloc[index, :]['sn']
            self.toggle_filters(active=False)
            sdt = (self.deployment.hydrophones[self._sn].deployment_start +
                   dt.timedelta(seconds=ANALYSIS_DELAY))
            edt = self.deployment.hydrophones[self._sn].deployment_end
            dtr_active, _, _ = self.filters['datetime_range']
            self.filters['datetime_range'] = (dtr_active, sdt, edt)
            self.update_df_set(reset_index=False)
        else:
            self.index = index

    def toggle_filters(self, active):
        for f, v in self._filters.items():
            if f in ['time_of_day', 'datetime_range']:
                _, s, e = v
                self._filters[f] = (active, s, e)
            elif f == 'sample_length':
                _, length = v
                self._filters[f] = (active, length)
            elif f =='codes':
                self._filters[f] = 'Any'

    def add_peak(self, index):
        signal(Signals.add_peak).send(self, index=index)

    def save_changes(self, status=None):
        if self.index is not None:
            status = self.current_sample['val_status'] if status is None else status
            # print(self.index, status)
            new_vals = {'val_samples': self.new_samples, 'val_status': status}
            cidx = [self._df_all.columns.get_loc(k) for k in new_vals]
            # print(f'index: {self.index}')
            self._df_all.iloc[self.index, cidx] = pd.Series(new_vals)
            # print('df_all updated sucessfully')
            # print(f'shistory_idx: {self._shistory_idx}')
            # print(f'{self._df_set.columns}')
            val_status_cid = self._df_set.columns.get_loc('val_status')
            # print(f'shistory_idx: {self._shistory_idx}')
            self._df_set.iloc[self._shistory_idx, val_status_cid] = status
            # print('df_set updated sucessfully')
            self.deployment.validations = self.df_all
            save_depfile(self.deployment, self.depfile, False)
            # print('Saved Sucessfully')

