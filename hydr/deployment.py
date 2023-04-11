#!/usr/bin/env python3
# --------------------------------------------------------------------------------------
# Project:       CSIRO Hydrophone Project
# Author:        Max Gunton
# --------------------------------------------------------------------------------------
# Description:   Contains the `generate_summary` method to create a summary of an
#                individual hydrophone's data files, configuration settings and
#                calibration history.  Furthermore, the method `generate_summary_cli` is
#                a wrapper to `generate_summary` method which adds command line parsing,
#                and it is defined in the `pyproject.toml` file as a project script with
#                the name `hydrophone-summary`.  Therefore, when the hydrophone package
#                is installed `generate_summary` method can be run from the command line
#                as follows:
#
#                ```bash
#                $ hydrophone-summary [-h] [-d DEST] datadir
#                ```
# --------------------------------------------------------------------------------------

# python standard library
import shutil
import os
import sys
from collections import Counter

from typing import List
from pathlib import Path
import datetime as dt

# installed packages
from tqdm import tqdm
import pandas as pd
import numpy as np

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
import hydr.soundtrap as soundtrap  # sn_from_filename, follows_conventions
from hydr.definitions import (NEW_DEPLOYMENT_STRUCTURE, CONVENTIONS,
                              WAV_DETAILS_COLUMNS, TQDM_WIDTH)
from hydr.utils import (ok_to_write, existing_directory, valid_filename,
                        save_depfile, load_depfile, unique_items)
from hydr.types import Deployment, Hydrophone
import hydr.formatting as formatting

tqdm.pandas()  # initialize tqdm to display progress by using df.progress_apply

# TODO: Finish docstrings
# TODO: Script to export `audio coverage plot`
# TODO: Script to export `audio summary`
# TODO: Have summary reports reflect the deployment start and end times
#       - If deployment_end == audio_end (battery died while deployed)
#       - if deployment_start == audio_start (missing beginning of deployment)
# TODO: Figure out a nice flow between the summary string values and usable values for
#       the attributes used in Hydrophone class


def _recursive_writer(structure: list, parent: str) -> None:
    """
    This method created the directory structure defined by the parameter `structure` at
    the path `parent`, and raises FileExistsError if the parent directory already
    exists. It accomplished this by recursively working its way through `structure`
    which is a list containing the children of the current parent where each child is
    one of the following:

        - directory       (dict: each key is directory with its value being a
                           `substructure`) ** i.e. recursive bit **
        - empty file      (str: is filename)
        - template file   (Tuple[str, str]: tuple[0] is filename & tuple[1] is path to
                           template source)

    It first creates the directory parent and then does the following (depending on what
    it encounters):

         - when a dictionary is encountered it iterates through the keys recursively
           calling itself with `parent=<parent>/<key>` and the
           `structure=<dictionary>[<key>]`
         - when a string is encountered it creates a file with name equal to the value
           of the string in the `parent`
         - when a tuple is encountered it copies the template source from `tuple[1]` to
           a file with the name equal to `tuple[0]` in `parent`

    :param structure: list - representing directory hierarchy (see
                             `hydrophone.definitions.NEW_DEPLOYMENT_STRUCTURE`)
    :param parent: str - the current parent directory
    :return: None
    """
    os.makedirs(parent, exist_ok=False)  # create the parent directory
    for item in structure:
        # when string, create empty file with name equal to `item`
        if issubclass(type(item), str):
            Path(os.path.join(parent, item)).touch()
        # when tuple, copy template from location `item[1]` to file with name `item[0]`
        elif issubclass(type(item), tuple):
            dest, src = item
            shutil.copy(src, os.path.join(parent, dest))
        # when dict, call recursive_writer (i.e. this method) for each key in item
        elif issubclass(type(item), dict):
            for key in item:
                _recursive_writer(item[key], os.path.join(parent, key))


# FIXME: USER_METHOD
def new_project(name: str, dest: str = '.') -> None:
    """
    This method creates a new project directory defined by
    `hydrophone.definitions.NEW_DEPLOYMENT_STRUCTURE` at path `dest`/`name`.   It
    requires that `dest` be an existing directory and `name` consists of at least one
    character and only from those defined in
    `hydrophone.definitions.ALLOWED_FILENAME_CHARACTERS`.

    :param name: str - root name of project directory
    :param dest: str - location to create new project
    :return:
    """
    # check that `name` if valid filename and that `dest` is an existing directory
    # TODO: use mval for validation?
    name, dest = valid_filename(name), existing_directory(dest)

    # create new deployment directory hierarchy
    _recursive_writer(NEW_DEPLOYMENT_STRUCTURE, os.path.join(dest, name))
    print(f"\nSuccessfully created new project at:\n\t"
          f"`{os.path.abspath(os.path.join(dest,name))}`\n")


# FIXME: USER_METHOD
def new_depfile(datadir: str, convention: str = 'SoundTrap',
                dest: str = '.') -> None:
    files = [
        os.path.abspath(str(f)).replace('\\', '/')
        for f in Path(datadir).rglob('*')
    ]
    dest = f'{dest}/deployment.data' if os.path.isdir(dest) else dest

    # initialize the object to be stored in depfile
    deployment = Deployment(convention)
    if convention == 'SoundTrap':
        files = [f for f in files if soundtrap.follows_conventions(f)]  # only ST files
        sns = unique_items([soundtrap.sn_from_filename(f) for f in files])
        for i, sn in enumerate(sns):
            fs = [f for f in files if soundtrap.sn_from_filename(f) == sn]
            print(f'\nSN: {sn} -- ({i+1} of {len(sns)})')
            h = Hydrophone(sn, fs)  # create hydrophone object with associated files
            h.ext_counts = dict(Counter([soundtrap.ext_from_filename(f) for f in fs]))
            for w in tqdm(h.all_wavs, desc='extracting log details'.ljust(TQDM_WIDTH)):
                w.set_extra_details(**(soundtrap.extra_wav_details(w.file)))
            deployment.add_hydrophone(h)  # add the hydrophone to the deployment
    else:
        raise RuntimeError(f'\nUnknown convention `{convention}`.  Please choose from '
                           f'one of the following: '
                           f'{", ".join([f"`{c}`" for c in CONVENTIONS])}')
    save_depfile(deployment, dest)


# FIXME: USER_METHOD
# TODO: Allow for multiple bounds (if hydrophone moved mid deployment for example)
def set_bounds(depfile: str, sn: str, start: dt.datetime, end: dt.datetime):
    deployment = load_depfile(depfile)
    h = deployment.hydrophones[sn]
    h.deployment_start, h.deployment_end = start, end
    save_depfile(deployment, depfile, False)
    print("Bounds saved ...")


# FIXME: USER_METHOD
def export_summaries(depfile: str, sns: str = None, dest: str = '.') -> None:
    """
    Given the input parameter `depfile`, `sns`, and `dest`.  This method generates
    summaries of the data for hydrophones with serial numbers included in `sns` as well
    as any calibration data it can obtain and writes them to `<dest>/<sn>_summary.txt`.

    This includes the following:

    :param datadir: str - directory containing individual hydrophone data
    :param sns: List[str] - list of serial numbers to include (None includes all)
    :param dest: str - directory where to write output
    :return: None
    """
    # 1) load the deployment
    deployment = load_depfile(depfile)
    sns = deployment.sns if sns is None else sns

    hs = [h for sn, h in deployment.hydrophones.items() if sn in sns]
    for h in hs:
        gaps = [g for f, g in h.relevant_gaps if g is not None]
        wav_durations = [d.total_seconds() for d in h.wav_durations]
        deployment_details = dict(
            sn=h.sn,
            ext_counts=h.ext_counts,
            unused_wavs=h.unused_wavs,
            sr_counts=h.sr_counts,
            utc_offset_counts=h.utc_offset_counts,
            gain_counts=h.gain_counts,
            audio_start=h.audio_start,
            audio_end=h.audio_end,
            deployment_start=h.deployment_start,
            deployment_end=h.deployment_end,
            total_audio=h.total_audio.total_seconds(),
            bad_wavfiles=h.bad_wavfiles,
            bad_logfiles=h.bad_logfiles,
            mean_gap=np.mean(gaps),
            median_gap=np.median(gaps),
            max_gap=np.max(gaps),
            mean_duration=np.mean(wav_durations),
            median_duration=np.median(wav_durations)
        )
        device_details = (
            soundtrap.format_specs(soundtrap.device_specs_from_sn(h.sn))
            if deployment.convention == 'SoundTrap'
            else None
        )
        summary = formatting.summary_report(deployment_details, device_details)
        if ok_to_write(f"{dest}/{h.sn}_summary.txt"):
            with open(f"{dest}/{h.sn}_summary.txt", "wt") as outfile:
                outfile.write(summary)
    save_depfile(deployment, depfile, check_overwrite=False)


# FIXME: USER_METHOD
def export_wav_details(depfile: str, sns: str = None, dest: str = '.') -> None:
    """
    Given the input parameter `depfile`, `sns`, and `dest` this method creates
    pd.DataFrame objects for each hydrophone with a serial number included in `sns`.
    These dataframes contain all the `wav` files associated to the hydrophone including
    details and the result is saved to `<dest>/<sn>_wav_details.csv`.  The columns
    included in this csv are as follows:

        | wavfiles | sr | start | end | duration | utc_offset | gain | gaps |

    :param datadir: str - directory containing individual hydrophone data
    :param sns: List[str] - list of serial numbers to include (None includes all)
    :param dest: str - directory where to write output
    :return: None
    """

    # 1) load the deployment
    deployment = load_depfile(depfile)
    sns = deployment.sns if sns is None else sns

    hs = [h for sn, h in deployment.hydrophones.items() if sn in sns]
    for h in hs:
        c = {i: [] for i in WAV_DETAILS_COLUMNS}
        gaps = h.all_gaps
        for w in h.all_wavs:
            gap = [g for f, g in gaps if f == w.file].pop()
            c['file'].append(w.file)
            c['sr'].append(w.sr)
            c['frames'].append(w.frames)
            c['start'].append(w.start.strftime('%Y-%m-%d %H:%M:%S'))
            c['end'].append(w.end.strftime('%Y-%m-%d %H:%M:%S'))
            c['duration'].append(w.duration.total_seconds())
            c['utc_offset'].append(w.utc_offset)
            c['gain'].append(w.gain)
            c['seconds_to_next_wavfile'].append(gap)
        df = pd.DataFrame(c)

        # write wav details to dest
        if ok_to_write(f"{dest}/{h.sn}_wav_details.csv"):
            df.to_csv(f"{dest}/{h.sn}_wav_details.csv", index=False)
    save_depfile(deployment, depfile, check_overwrite=False)


def export_bounds(depfile: str, dest: str = '.'):
    dest = f'{dest}/bounds.csv' if os.path.isdir(dest) else dest
    deployment = load_depfile(depfile)
    sns = deployment.sns
    d_starts = [deployment.hydrophones[sn].deployment_start for sn in sns]
    d_ends = [deployment.hydrophones[sn].deployment_end for sn in sns]
    a_starts = [deployment.hydrophones[sn].audio_start for sn in sns]
    a_ends = [deployment.hydrophones[sn].audio_end for sn in sns]
    bounds = pd.DataFrame({'serial_number': sns, 'audio_start': a_starts,
                           'deployment_start': d_starts, 'audio_end': a_ends,
                           'deployment_end': d_ends})
    if ok_to_write(dest):
        bounds.to_csv(dest, index=False)


def main():
    pass


if __name__ == '__main__':
    main()
