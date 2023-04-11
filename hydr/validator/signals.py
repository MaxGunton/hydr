from enum import Enum, auto
from blinker import signal
import time
import datetime as dt


class Signals(Enum):
    # application state data
    get_state = auto()  # request for state object
    state = auto()  # response to request for state object

    playhead_updated = auto()  # using in AudioControls and Spectrogram

    set_updated = auto()  # State sends out, used by SetSelection
    load_sample = auto()
    add_peak = auto()  # using in ClassificationControls and Spectrogram


class Connector:
    def __init__(self, *args, **kwargs):
        """
        Should be subclassed by classes that we want to receive signals.  It uses
        blinker and the observer pattern to register/connect current `Connector` class
        to any signals from `Signals` class if it implements an `on_<signal_name>`
        method
        """
        # go through all signals and if this class has an "on_<signal_name>" method
        # connect it to the signal
        sig_handlers = [
            (s, h)
            for s, h in [(s, getattr(self, f"on_{s.name}", None)) for s in Signals]
            if h and callable(h)
        ]
        for s, h in sig_handlers:
            signal(s).connect(h)


class Receiver(Connector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = None
        self._lock = False

    @property
    def state(self):
        if self._state is None:
            self._lock = True  # set lock and wait for a response
            signal(Signals.get_state).send(self)  # send out the request
            start = dt.datetime.now()
            while self._lock:  # wait for the response handler response to disable lock
                time.sleep(0.01)
                if (dt.datetime.now() - start).total_seconds() > 5:
                    raise TimeoutError(
                        f"No response to request to `{Signals.get_state}`.  "
                    )
        return self._state

    def on_state(self, sender, value):
        self._state = value
        self._lock = False  # turn the lock off if on


class SignalLogger(Receiver):
    """
    A method to log the signals to ensure they are being sent
    """
    def __init__(self):
        super().__init__()

    def on_get_state(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.get_state, value)

    def on_state(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.state, value)

    def on_playhead_updated(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.playhead_updated, value)

    def on_submit(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.submit, value)

    def on_skip(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.skip, value)

    def on_reset_sample(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.reset_sample, value)

    def on_index_updated(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.index_updated, value)

    def on_set_updated(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.set_updated, value)

    def on_add_peak(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.add_peak, value)

    def on_remove_peak(self, sender, **kwargs):
        value = kwargs['value'] if kwargs else 'N/A'
        self.print_signal_details(sender, Signals.remove_peak, value)

    @staticmethod
    def print_signal_details(sender, s, v):
        print(f'SENDER: {sender} | SIGNAL: {s} | VALUE: {v}')

