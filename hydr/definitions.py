# python standard library
import os
import numpy as np
from matplotlib import cm as colormap

SRC_PATH = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
TQDM_WIDTH = 25
CONVENTIONS = ['SoundTrap']

ANALYSIS_DELAY = 600  # seconds (i.e. 10 minutes)
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

# VALIDATOR >>>
VALIDATOR_ICON = f'{SRC_PATH}/icons/validator.png'
WARNING_ICON = f'{SRC_PATH}/icons/warning.png'
STATUS_COLORS = {
            "Status.Submitted": "rgb(0, 127, 0)",  # Green
            "Status.Revisit": "rgb(174, 94, 255)",  # "rgb(115, 3, 252)",  # Purple
            "Status.Skipped": "rgb(255, 255, 0)",  # "rgb(255, 234, 0)",  # Yellow
            "Status.Modified": "rgb(255, 127, 0)",  # "rgb(230, 138, 0)",  # Orange
            "Status.MachineLabelled": "rgb(200, 0, 0)",  # Red
            "Status.NoStatus": "rgb(0, 0, 0)",  # Black
}
STEP_BACK_ICON = f'{SRC_PATH}/icons/step_back.png'
PLAY_PAUSE_ICON = f'{SRC_PATH}/icons/play_pause.png'
PLAY_ICON = f'{SRC_PATH}/icons/play.png'
PAUSE_ICON = f'{SRC_PATH}/icons/pause.png'
STOP_ICON = f'{SRC_PATH}/icons/stop.png'
STEP_AHEAD_ICON = f'{SRC_PATH}/icons/step_ahead.png'
NO_SAMPLE = f'{SRC_PATH}/icons/no_sample.png'
NUM_COLORS = 8
LINE_THICKNESS = 2
DASH_LENGTH = 20
DASH_GAP = int(round(DASH_LENGTH/GOLDEN_RATIO))
BLOCKSIZE = 1024  # 2048  # this is for the wav player through pyaudio
AUDIO_STEP = 0.9  # seconds that jump when step back or ahead is pressed

BLAST_CLASSES = [
    'grunt',
    'background',
    'blast',
    'bump/scrap',
    'rumble',
    'vessel',
    'other'
]
DISPLAY_SPEC_PARAMS = dict(
    n_fft=4096,
    win_length=4096,
    freq_bins=375,
    hop_length=295,
    window="hann",
    freq_scale='linear',
    center=True,
    pad_mode='reflect',
    iSTFT=False,
    fmin=13,
    fmax=1000,
    trainable=False,
    output_format="Magnitude",
    verbose=False
)
PEAK_SPEC_PARAMS = dict(
        n_fft=2048,  # 2048, 4096
        win_length=2048,  # 2048, 4096
        freq_bins=150,  # 128, 224
        hop_length=16,  # 16
        window='hann',
        freq_scale='linear',
        center=False,
        pad_mode='reflect',
        iSTFT=False,
        fmin=150,  # 13, 75
        fmax=300,  # 300, 500
        trainable=False,
        output_format="Magnitude",
        verbose=False,
)
VALIDATOR_SAMPLE_BUFFER = 2  # seconds
VALIDATOR_COLORS = {
    i: (colormap.get_cmap('Dark2', NUM_COLORS).colors * 255).astype(np.uint8)[i, :]
    for i in range(NUM_COLORS)
}
VALIDATOR_COLORS['active'] = {
    'start': np.array([255, 255, 255, 255]).astype(np.uint8),
    'end': np.array([0, 0, 0, 255]).astype(np.uint8),
}
VALIDATOR_COLORS['playline'] = np.array([255, 0, 0, 255]).astype(np.uint8)
VALIDATOR_COLORS['peak'] = [
    np.array([255, 255, 255, 255]).astype(np.uint8),
    np.array([0, 0, 0, 255]).astype(np.uint8)
]
# <<< VALIDATOR


# BLAST_224X224_6CAT >>>
DEVICES = tuple([f'cuda:{i}' for i in range(100)] + ['cpu'])
BLASTS_224x224_6CAT_WEIGHTS = f'{SRC_PATH}/hydr/models/blasts_224x224_6cat.pth'
BLASTS_224x224_6CAT_OUTPUT_COLUMNS = [
    "file",
    "start",
    "end",
    "code",
    "score",
    "background_confidence",
    "blast-1_confidence",
    "blast-2_confidence",
    "blast-3_confidence",
    "blast-4_confidence",
    "undetermined_confidence",
    "model",
]

BLASTS_224x224_6CAT_SPEC_PARAMS = dict(
    n_fft=4096,
    win_length=4096,
    freq_bins=None,
    hop_length=1071,
    window="hann",
    freq_scale="no",
    center=True,
    pad_mode="reflect",
    iSTFT=False,
    fmin=0,
    trainable=False,
    output_format="Magnitude",
    verbose=False
)
BLASTS_224x224_6CAT_CLASSES = (
    "blast-1",
    "blast-2",
    "blast-3",
    "blast-4",
    "undetermined",
    "background"
)
# <<< BLAST_224X224_6CAT
WAV_DETAILS_COLUMNS = [
    'file',
    'sr',
    'frames',
    'start',
    'end',
    'duration',
    'utc_offset',
    'gain',
    'seconds_to_next_wavfile',
]
# PROJECT DIRECTORY >>>
TEMPLATE_DIRECTORY = f'{SRC_PATH}/templates'
TEMPLATES = {i: f'{TEMPLATE_DIRECTORY}/{i}' for i in os.listdir(TEMPLATE_DIRECTORY)}
NEW_DEPLOYMENT_STRUCTURE = [
    {
        "00_hydrophone_data": [],
        "01_machine_learning": [
            {"blast_224x224_6cat": []},
        ],
        "02_analysis": [
            {"multilateration": []}
        ],
        "03_report": [
            {
                "embedded": [
                    {"samples": []}
                ],
                "source_data": []
            },
        ],
        "10_forms_photos_etc": [],
        "11_ais_data": [],
        "12_vms_data": [],
    },
    ("README.md", TEMPLATES["project_readme.md"]),
    ("NOTES.txt", TEMPLATES["project_notes.txt"]),
]
# <<< PROJECT DIRECTORY

# DISPLAY FORMATTING >>>
LABEL_WIDTH = 20
LINE_LENGTH = 88
# <<< DISPLAY FORMATTING

# FILENAME RESTRICTIONS >>>
# ALLOWED_FILENAME_CHARACTERS is [a-zA-Z0-9-_]
ALLOWED_FILENAME_CHARACTERS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "_",
}


class ContainsRestrictedCharacter(Exception):
    """
    Exception raised when str contains characters other than those in
    ALLOWED_FILENAME_CHARACTERS.
    """

    def __init__(self, name: str = ""):
        """
        Defines the error message and initializes its superclass
        :param name: str - the offending name that threw the error (optional)
        """
        self.message = (
            f"{name} contains characters not included in set [a-zA-Z0-9_-]".strip(" ")
        )
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
# <<< FILENAME RESTRICTIONS
