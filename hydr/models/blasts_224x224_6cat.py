import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import List, Tuple
from PIL import Image
from soundfile import SoundFile, SEEK_END
from torchvision import transforms
from tqdm import tqdm
from nnAudio.features.stft import STFT


# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import (BLASTS_224x224_6CAT_WEIGHTS,
                              BLASTS_224x224_6CAT_SPEC_PARAMS,
                              BLASTS_224x224_6CAT_CLASSES,
                              BLASTS_224x224_6CAT_OUTPUT_COLUMNS,
                              TQDM_WIDTH)


def amplitude_to_db(spectrogram: np.ndarray) -> np.ndarray:
    """
    Based on the `amplitude_to_db` and `power_to_db` librosa functions, but with
    hard-coded parameter values.  Having this removes the dependancy on librosa

    > **Warning**
    >
    > Applying this per-batch as opposed to per-sample leads to a significant variation
    > in results.

    :param spectrogram: np.ndarray - spectrogram with values representing amplitude of
                                     frequencies
    :return: np.ndarray - spectrogram with values representing dB values of frequencies
    """

    ref_value, amin, top_db = 1.0, 1e-10, 80.0
    s = np.square(np.abs(spectrogram))  # convert to power spectra
    s = 10 * np.log10(np.maximum(amin, s))
    s -= 10.0 * np.log10(ref_value)
    s = np.maximum(s, s.max() - top_db)
    return s


class Model(nn.Module):
    """
    This class defines the convolutional neural network that is blasts_224x224_6cat.
    The `features` attribute contains the feature extraction layers (i.e.
    convolutional layers), and the `classifier` attribute contains the densely connected
    reduction layers (i.e. the classification layers).

    The `data_transforms` attribute contains the necessary transformations to preform on
    a PIL image when they are passed into the `classify` function.

    This class has two methods: `forward` and `classify`.  Forward is used for both
    training and classification, and simply feeds a 224x224 torch.Tensor through the
    feature extraction and classification layer resulting in a 1 dimensional
    torch.Tensor with 6 values corresponding to each class
    """
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=(1, 1), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=(1, 1), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=(1, 1), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 6),
        )
        self.data_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5],)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # feature extraction layers
        x = x.view(x.size(0), -1)  # flatten the tensor before classification layers
        x = self.classifier(x)  # classification layers
        return x

    def classify(self, batch: List[Image.Image], dev: str) -> pd.DataFrame:
        classes = BLASTS_224x224_6CAT_CLASSES
        batch = torch.stack([self.data_transforms(s) for s in batch]).to(device=dev)
        dataset = torch.utils.data.TensorDataset(batch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch.shape[0])

        self.eval()  # put model into evaluation mode
        probs = []

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for i, images in enumerate(dataloader):
                images = images[0]
                outputs = self(images)
                probabilities = softmax(outputs.data)
                probs.append(probabilities.cpu().numpy())

        probs = np.squeeze(np.vstack(probs))

        # if there is only one in the batch need to put it in a list
        probs = [probs] if type(probs[0]) != np.ndarray else probs

        prob0 = [i[0] for i in probs]
        prob1 = [i[1] for i in probs]
        prob2 = [i[2] for i in probs]
        prob3 = [i[3] for i in probs]
        prob4 = [i[4] for i in probs]
        prob5 = [i[5] for i in probs]

        # create a dataframe containing all the predictions
        df = pd.DataFrame({0: prob0, 1: prob1, 2: prob2, 3: prob3, 4: prob4, 5: prob5})
        df["index"] = range(len(df))
        df["code"] = df.drop("index", axis="columns").idxmax(axis=1)
        df["code"] = df["code"].apply(lambda x: classes[x])
        df = df.rename(columns={i: f'{classes[i]}_confidence' for i in range(6)})
        df['score'] = df.apply(self.compute_score, axis=1)
        return df

    @staticmethod
    def compute_score(row: pd.Series) -> float:
        bg = row['background_confidence']
        b1 = row['blast-1_confidence']
        b2 = row['blast-2_confidence']
        b3 = row['blast-3_confidence']
        b4 = row['blast-4_confidence']
        return (
            np.power(
                np.max([b1, b2, b3, b4]),
                np.sum([b1, b2, b3, b4])
            ) + b1) / bg * np.power(b1, b2)


class ModelRunner:
    # fixed parameters
    _model = Model()
    _step_secs = 6.0
    _sample_secs = 10.0
    _offset_secs = np.arange(0, np.lcm(int(_step_secs), int(_sample_secs)),
                             int(_step_secs))
    _spec_kernel = None
    _spec_kernel_sr = None
    # tunable parameters
    _device = "cpu"
    _batch_size = 150

    def __init__(self, device: str = 'cpu', batch_size: int = 150) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model.load_state_dict(torch.load(BLASTS_224x224_6CAT_WEIGHTS,
                                               map_location=torch.device(self.device)))

    @property
    def model(self) -> Model:
        return self._model

    @property
    def step_secs(self) -> float:
        return self._step_secs

    @property
    def sample_secs(self) -> float:
        return self._sample_secs

    @property
    def offset_secs(self) -> np.ndarray:
        return self._offset_secs

    def get_spec_kernel(self, sr) -> STFT:
        if self._spec_kernel_sr != sr:
            params = {**BLASTS_224x224_6CAT_SPEC_PARAMS, **dict(fmax=sr / 2, sr=sr)}
            self._spec_kernel = STFT(**params).to(device=torch.device(self.device))
            self._spec_kernel_sr = sr
        return self._spec_kernel

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        if not torch.cuda.is_available() and device.startswith("cuda:"):
            print(f"Scan using device: `{device}` not available; defaulting to `cpu`.")
            device = "cpu"
        elif not device.startswith('cuda:') and device != 'cpu':
            raise RuntimeError(f"Unknown device: `{device}`.  Try using one of the "
                               f"following instead: 'cpu', cuda:0, cuda:1, ...")
        self._device = device
        self._model.to(device=torch.device(device))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            error = TypeError if type(batch_size) is not int else ValueError
            raise error(f"Invalid `batch_size` parameter: `{batch_size}`.  Value "
                        f"must be integer >= 1.  ")
        self._batch_size = batch_size

    def grab_audio_batch(self, wavobj: SoundFile,
                         batch_number: int) -> Tuple[np.ndarray, bool]:
        sr = wavobj.samplerate
        stp_f = int(round(self.step_secs * sr))
        smp_f = int(round(self.sample_secs * sr))
        bs = self.batch_size
        ost_f = self.offset_secs * sr

        final = False
        # compute and seek to start frame, then read in audio
        start_frame = batch_number * bs * stp_f
        wavobj.seek(start_frame)
        a = wavobj.read((bs - 1) * stp_f + smp_f)
        # copy audio and shift to create 5 audio segments with proper offsets
        abp = [
            np.roll(a.copy(), -ost, 0)[:-ost] if ost != 0 else a.copy() for ost in ost_f
        ]
        # reshape to (batch_num, sample_frames) and select every 3 samples
        abp = [
            i[: (len(i) // smp_f) * smp_f].reshape(len(i) // smp_f, -1)[::3, :]
            for i in abp
            if (len(i) // smp_f) != 0  # For batches that don't have len(ost_f) samples
        ]
        bsc = sum([i.shape[0] for i in abp])  # compute the batch size
        if bsc < bs:
            # if actual batch size < expected add `final_sample` and set `final` flag
            wavobj.seek(-smp_f, SEEK_END)
            final_sample = wavobj.read(smp_f).reshape(1, -1)
            last_idx = 0
            for idx, i in enumerate(abp):
                if i.shape[0] < abp[0].shape[0]:
                    last_idx = idx
                    break
            if len(abp) == 0:
                abp.append(final_sample)
            else:
                abp[last_idx] = np.concatenate((abp[last_idx], final_sample), axis=0)
            bsc += 1
            final = True
        # zip together 5 arrays into one with shape (batch_size, sample_frames)
        audio_batch = np.empty((bsc, smp_f))
        for i in range(len(abp)):
            audio_batch[i::len(abp), :] = abp[i]  # was len(ost_f)
        return audio_batch, final

    def audio_to_spec_nnaudio(self, audio_batch: np.ndarray,
                              spec_kernel: STFT) -> List[Image.Image]:
        # put the audio batch onto the device specified and cast as torch.Tensor
        audio_batch = torch.tensor(audio_batch, device=self.device).float()

        # convert to spectrogram
        spec_torch = spec_kernel(audio_batch)
        spec_batch = []
        for i in range(spec_torch.shape[0]):
            # convert values to db
            s = amplitude_to_db(spec_torch[i, :, :].to(device="cpu").numpy())
            # flip vertically
            s = s[::-1, :]
            # normalize
            s = ((s - s.min()) / (s.max() - s.min()) * 255).astype(np.uint8)
            # crop
            s = s[-224:, :224]
            # cast as PIL Image
            s = Image.fromarray(s)
            # add to list
            spec_batch.append(s)
        return spec_batch

    def classify_spec_batch(self, wavobj: SoundFile, spec_batch: List[Image.Image],
                            batch_number: int, final: bool) -> pd.DataFrame:
        frms = wavobj.frames
        filename = wavobj.name
        sr = wavobj.samplerate
        stp_f = int(round(self.step_secs * sr))
        smp_f = int(round(self.sample_secs * sr))
        bs = self.batch_size

        df = self.model.classify(spec_batch, self.device)

        df["file"] = filename

        df["start"] = (df["index"] + batch_number * bs) * (stp_f / sr)
        df["end"] = df["start"] + (smp_f / sr)
        if final:
            # if it was the final batch of the file then the last sample was the last
            # sample_length (10 seconds) of the file the values must be changed
            start_iloc = df.columns.get_loc("start")
            end_iloc = df.columns.get_loc("end")
            df.iloc[-1, start_iloc] = (frms - smp_f) / sr
            df.iloc[-1, end_iloc] = frms / sr

        df["model"] = "blasts_224x224_6cat"

        df = df.drop("index", axis="columns")
        return df

    def scan(self, f) -> pd.DataFrame:
        with SoundFile(f) as wavobj:
            frms = wavobj.frames
            sr = wavobj.samplerate
            stp_f = int(round(self.step_secs * sr))
            smp_f = int(round(self.sample_secs * sr))
            bs = self.batch_size
            spec_kernel = self.get_spec_kernel(sr)
            all_batch_dfs = []
            for bn in tqdm(
                    range(int(np.ceil(frms / stp_f / bs))),
                    desc='classifying batches'.ljust(TQDM_WIDTH)
            ):
                # check if wavfile is long enough
                if frms < smp_f:
                    print(f"File too short, skipping.")
                    break

                # grab audio batch
                audio_b, final = self.grab_audio_batch(wavobj, bn)

                # convert to audio batch to spectrogram batch
                spec_b = self.audio_to_spec_nnaudio(audio_b, spec_kernel)

                # classify the spectrogram batch
                batch_df = self.classify_spec_batch(wavobj, spec_b, bn, final)

                # add batch results to list
                all_batch_dfs.append(batch_df)
        # combine list of batch results into a single DataFrame and return
        return self.combine_batch_dfs(all_batch_dfs)

    @staticmethod
    def combine_batch_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The list of DataFrames sent into this method are combined and the columns are
        ordered by `hydrophone.definitions.DETECTION_COLUMNS`.  If the list is empty
        then an empty DataFrame is created with the columns
        `hydrophone.definitions.DETECTION_COLUMNS`.  The resulting DataFrame
        is returned.

        :param df_list: List[pd.DataFrame] - list of dataframes to concatenate
        :return: pd.DataFrame - result of concatenating df_list into single DataFrame
                                & columns sorted into the order defined by
                                `hydrophone.definitions.DETECTION_COLUMNS`
        """
        # combine the list of DataFrames into one and order the columns
        df = (
            pd.concat(df_list, ignore_index=True)[BLASTS_224x224_6CAT_OUTPUT_COLUMNS]
            if df_list
            else pd.DataFrame(columns=BLASTS_224x224_6CAT_OUTPUT_COLUMNS)
        )
        return df
