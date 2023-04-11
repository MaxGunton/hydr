import os

import torch
# from scipy.signal import butter
# from scipy.stats import mode  # number that occurs most frequently

import datetime as dt
import pandas as pd
import soundfile as sf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nnAudio.features.stft import STFT
from tqdm import tqdm
from PIL import Image


# TODO: Look into what I used for the training set stuff I was doing; it may be the same
#       code as this, but it may be different too
tqdm.pandas()

# CONFIGURATION VARIABLES ----------------------------------------------------------------------------------------------
SAMPLE_LENGTH = 10  # seconds
SAMPLE_START_PADDING = 1  # seconds
BATCH_SIZE = 150
SPECTROGRAM_ARGUMENT_PRESETS = {
    'no': {},
    'rough_blast_peaks': dict(
        n_fft=2048,  # 2048, 4096
        win_length=2048,  # 2048, 4096
        freq_bins=150,  # 128, 224
        hop_length=16,  # 16
        window='hann',
        freq_scale='linear',
        center=False,
        pad_mode='constant',
        iSTFT=False,
        fmin=150,  # 13, 75
        fmax=300,  # 300, 500
        trainable=False,
        output_format="Magnitude",
        verbose=False,
    ),
    'standard': dict(
        n_fft=4096,
        win_length=4096,
        freq_bins=224,
        hop_length=1055,
        window='hann',
        freq_scale='linear',
        center=False,
        pad_mode='constant',
        iSTFT=False,
        fmin=13,
        fmax=500,
        trainable=False,
        output_format="Magnitude",
        verbose=False,
    ),
}


def get_stft_kernel(sr, preset='no', kwargs={}):
    """
    This function returns a STFT kernel based on the samplerate and spec_type/kwargs provided.

    **NOTE:** If kwargs are provided these will overwrite any of the values used in the preset

    **Params**

    sr: `int`: samplerate of the audio sample needed to generate accurate spectrogram
    spec_type: `str`: one of `{'default', 'rough_blast_peaks'}` (i.e key in SPECTROGRAM_ARGUMENT_PRESETS dictionary)
    kwargs: `dict`: keyword arguments for initializing the nnAudio.features.stft.STFT object

    **Returns**

    nnAudio.features.stft.STFT: A kernel for converting audio array to a spectrogram

    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert preset in SPECTROGRAM_ARGUMENT_PRESETS, f'Unknown spectrogram preset: {preset}'
    args = SPECTROGRAM_ARGUMENT_PRESETS.get(preset)

    # overwrite preset args with any supplied kwargs
    for k, v in kwargs.items():
        args[k] = v

    kernel = STFT(**args, sr=sr).to(device=device)
    return kernel


def save_spectrograms(df_batch, kernel, batch_num, output_directory):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_norm = mpl.colors.LogNorm(clip=True)

    batch_size = df_batch.shape[0]
    audio_samples_batch = np.stack(df_batch['audio'].to_numpy(), axis=0)
    audio_samples_batch = torch.tensor(audio_samples_batch, device=device).float()

    spectrogram_batch = kernel(audio_samples_batch) ** 25  # increases the discrepancy between the large & small values
    spectrogram_batch = spectrogram_batch.to(device='cpu')

    spectrogram_batch = torch.flip(spectrogram_batch, (1,)).numpy()  # flip vertical axis and convert to numpy array
    spectrogram_batch = np.where(spectrogram_batch == np.inf, np.finfo('float32').max, spectrogram_batch)  # clip values
    shape = spectrogram_batch.shape

    spectrogram_batch = spectrogram_batch.reshape((batch_size, -1))
    spectrogram_batch = np.apply_along_axis(log_norm, 1, spectrogram_batch)
    spectrogram_batch = np.apply_along_axis(mpl.cm.turbo, 1, spectrogram_batch)
    spectrogram_batch = (spectrogram_batch * 255).astype(np.uint8)
    spectrogram_batch = spectrogram_batch.reshape(tuple([i for i in shape]+[-1]))

    os.makedirs(output_directory, exist_ok=True)
    peak_idx = df_batch.columns.get_loc('peak')
    half_thickness = max(int(round(spectrogram_batch.shape[2] / 1000)), 1)
    for i in range(spectrogram_batch.shape[0]):
        spectrogram = spectrogram_batch[i, :, :, :]
        height, width = spectrogram.shape[0:2]
        peak = int(round((width/SAMPLE_LENGTH) * df_batch.iloc[i, peak_idx]))
        peak = half_thickness if peak < half_thickness else peak  # move peak to edge of plot
        peak = width-half_thickness if peak > (width-half_thickness) else peak  # move peak to edge of plot
        peak_val = np.ones((height, half_thickness*2, 4), dtype=np.uint8) * 255
        spectrogram[:, peak-half_thickness:peak+half_thickness, :] = peak_val

        spectrogram = Image.fromarray(spectrogram_batch[i, :, :, :])
        spectrogram.save(f"{output_directory}/{batch_num}-{i}.png")


def get_blast_peaks(df_batch, kernel, batch_num, output_directory):
    """
    This function takes a batch of audio samples and returns a list containing the seconds from the start of the sample
    to where the peak frame is.

    **Params**

    df_batch: `pd.DataFrame`: contains a batch of sample data including the wav audio data
    kernel: `torch.?`

    **Returns**

    `list`: Containing the number of seconds from the start of the sample where the peak frame is.
    """
    # 1) convert audio batch into torch.tensor (in preparation for conversion)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # set device as GPU if available otherwise use CPU
    audio_samples_batch = np.stack(df_batch['audio'].to_numpy(), axis=0)  # convert pd.Series to np.ndarray
    audio_samples_batch = torch.tensor(audio_samples_batch, device=device).float()  # cast np.ndarray to torch.tensor

    # 2) send audio batch to spectrogram batch
    spectrogram_batch = kernel(audio_samples_batch) ** 25  # increase discrepancy between big & small values (was 30)
    spectrogram_batch = spectrogram_batch.to(device='cpu')  # push tensor onto CPU
    spectrogram_batch = torch.flip(spectrogram_batch, (1,)).numpy()  # flip vertical axis and convert to numpy array
    spectrogram_batch = np.where(spectrogram_batch == np.inf, np.finfo('float32').max, spectrogram_batch)  # clip values
    shape = spectrogram_batch.shape  # get the shape of the (<batch_size>, <height>, <width>)

    # 3) Compute the conversion factor (frame index to seconds from start of sample)
    conversion_factor = SAMPLE_LENGTH / shape[2]  # Sample length / spectrogram width

    # 4) Normalize each spectrogram (from 0 to 1) <OPTIONAL STEP>  (CAN PREVENT OVERFLOW IN FOLLOWING STEPS)
    # OPTION 1 - Linear ------------------------------------------------------------------------------------------------
    # spectrogram_batch = spectrogram_batch / (shape[1]*10)  # number of rows (prevent overflow on summations in step 5)
    # spectrogram_batch = spectrogram_batch / spectrogram_batch.max()  # divide by the maximum value
    # ------------------------------------------------------------------------------------------------------------------
    # OPTION 2 - Logarithmic -------------------------------------------------------------------------------------------
    log_norm = mpl.colors.LogNorm(clip=True)  # initialize the normalization function
    spectrogram_batch = spectrogram_batch.reshape((shape[0], -1))  # reshape the batch for normalization
    spectrogram_batch = np.apply_along_axis(log_norm, 1, spectrogram_batch)  # apply normalization (per sample)
    spectrogram_batch = spectrogram_batch.reshape(shape)  # reshape back to orignal shape
    # ------------------------------------------------------------------------------------------------------------------

    # 5a) Aggregate columns to reduce effect of outliers
    # This is done by taking sum of all frequencies for some time increment (i.e. number of columns)
    spectrogram_batch = spectrogram_batch.sum(axis=1)  # sum the freq columns output shape => (<batch_size>, <width>)
    spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(11, dtype=int), 'valid'),
                                            1, spectrogram_batch)  # sliding window sum every 101 values (to smooth)
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(1001, dtype=int) / 1001, 'valid'),
    #                                         1, spectrogram_batch)  # sliding window sum every 10 values (to smooth)
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(1001, dtype=int) / 1001, 'valid'),
    #                                         1, spectrogram_batch)  # sliding window sum every 10 values (to smooth)
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(1001, dtype=int) / 1001, 'valid'),
    #                                         1, spectrogram_batch)  # sliding window sum every 10 values (to smooth)
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(1001, dtype=int) / 1001, 'valid'),
    #                                         1, spectrogram_batch)  # sliding window sum every 10 values (to smooth)

    # 5b) To remove spikes make the values of each equal to the mode of the 10 values on each side.
    median_window_width = 1001
    spectrogram_batch = np.median(np.lib.stride_tricks.sliding_window_view(spectrogram_batch, median_window_width,
                                                                           axis=1), axis=2)
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, np.ones(11, dtype=int), 'valid'),
    #                                         1, spectrogram_batch)  # sliding window sum every 101 values (to smooth)
    # spectrogram_batch = np.median(np.lib.stride_tricks.sliding_window_view(spectrogram_batch, 21, axis=1), axis=2)

    # 5b) Use smoothing filter to reduce noise
    # Normalize data before sending through the filter
    # TODO: Could preform this filter before aggregation and then we wouldn't need to normalize the data twice
    # linear_norm = mpl.colors.Normalize(clip=True)  # initialize the normalization function
    # spectrogram_batch = np.apply_along_axis(linear_norm, 1, spectrogram_batch)  # apply normalization (per sample)
    # spectrogram_batch = np.where(spectrogram_batch <= 0, (0 + np.finfo('float').resolution), spectrogram_batch)
    # spectrogram_batch = np.where(spectrogram_batch >= 1, (1 - np.finfo('float').resolution), spectrogram_batch)
    # print(spectrogram_batch.min())
    # print(spectrogram_batch.max())
    # spectrogram_batch = np.apply_along_axis(lambda x: butter(10, x), 1, spectrogram_batch)

    # 6) Find the peaks
    # OPTION 1 - Peak is maximum value of frequency sums ---------------------------------------------------------------
    # peaks = (spectrogram_batch.argmax(axis=1))  # peak is the greatest value
    # ------------------------------------------------------------------------------------------------------------------
    # OPTION 2 - Peak is greatest difference between adjacent frequency sums -------------------------------------------
    # k_half_width = 1  # should be int
    # k = np.ones(k_half_width*2, dtype=int)
    # k[:k_half_width] = -1
    # spectrogram_batch = np.apply_along_axis(lambda x: np.convolve(x, k, 'valid'), 1, spectrogram_batch)
    # peaks = (spectrogram_batch.argmax(axis=1) + k_half_width)  # peak is the greatest difference
    # ------------------------------------------------------------------------------------------------------------------
    # OPTION 3 - Peak is greatest gradient of frequency sums -----------------------------------------------------------
    peaks = np.gradient(spectrogram_batch, 2, edge_order=1, axis=1).argmax(axis=1)  # peak is the greatest gradient
    # ------------------------------------------------------------------------------------------------------------------

    # 6b) Save the results of the sum and peak taken
    os.makedirs(output_directory, exist_ok=True)
    for i in range(spectrogram_batch.shape[0]):
        spectrogram = spectrogram_batch[i, :]
        plt.plot(spectrogram)
        plt.axvline(x=peaks[i], color="red")
        plt.savefig(f"{output_directory}/{batch_num}-{i}.png")
        plt.clf()

    # 7) Apply conversion factor to change index to seconds from start
    peaks = (peaks+(median_window_width/2)) * conversion_factor

    return peaks


def get_audio(row):
    file, start = list(row)
    sr, audio = None, None
    try:
        with sf.SoundFile(file) as wavobj:
            sr = wavobj.samplerate
            sample_f = SAMPLE_LENGTH * sr
            start_f = int(round((start - SAMPLE_START_PADDING) * sr))
            start_f = wavobj.frames - sample_f if (sample_f + start_f) > wavobj.frames else start_f
            wavobj.seek(start_f)
            audio = wavobj.read(SAMPLE_LENGTH * sr, dtype=np.float32)  # so all samples are 10 seconds
            frames = audio.shape[0]
            assert (SAMPLE_LENGTH * sr) == frames, f'Expected value: {SAMPLE_LENGTH * sr}, but received: {frames}'
    except Exception as error:
        print(str(error) + f'  Therefore, samples from {file} will ignored.  ')
    return pd.Series((sr, audio))


def main():
    f = 'C:/Users/maxgu/Documents/testfolder/deployment.data'

    df = pd.read_csv("blast_and_undetermined.csv")

    df[['sr', 'audio']] = df[['file', 'start']].progress_apply(get_audio, axis=1)

    # drop bad rows (samples) where the file couldn't be opened or read
    df = df.dropna()
    df = df.reset_index()  # reset the index values

    srs = set(df['sr'])
    assert len(srs) == 1, f'single samplerate expected, but multiple found: {srs}'  # assert single sr
    sr = srs.pop()

    time_now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_directory_spectrograms = f'./spectrograms/{time_now}'
    output_directory_frequency_sums = f'./freq_sums/{time_now}'

    df['peak'] = None
    peak_idx = df.columns.get_loc('peak')

    kernel = get_stft_kernel(sr=sr, preset='rough_blast_peaks')
    for batch_num, i in tqdm(enumerate(range(0, df.shape[0], BATCH_SIZE))):
        j = i + BATCH_SIZE if i + BATCH_SIZE < df.shape[0] else df.shape[0]
        df.iloc[i:j, peak_idx] = get_blast_peaks(
            df.iloc[i:j, :],
            kernel,
            batch_num,
            output_directory_frequency_sums
        )

    df.to_csv("test_output.csv", index=False)

    kernel = get_stft_kernel(sr=sr, preset='standard')  # 'standard'
    for batch_num, i in tqdm(enumerate(range(0, df.shape[0], BATCH_SIZE))):
        j = i + BATCH_SIZE if i + BATCH_SIZE < df.shape[0] else df.shape[0]
        save_spectrograms(
            df.iloc[i:j, :],
            kernel,
            batch_num,
            output_directory_spectrograms
        )


if __name__ == '__main__':
    main()
