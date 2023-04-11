#!/usr/bin/env python3
# --------------------------------------------------------------------------------------
# Project:       CSIRO Hydrophone Project
# Author:        Max Gunton
# --------------------------------------------------------------------------------------
# Description:   Contains the `combine_csvs` method intended for use to combine multiple
#                csv files with the same structure into a single file.  Furthermore, the
#                method `combine_csvs_cli` is a wrapper to `combine_csvs` method which
#                adds command line parsing, and it is defined in the `pyproject.toml`
#                file as a project script with the name `combine-csvs`.  Therefore, when
#                the hydrophone package is installed `combine_csvs` method can be run
#                from the command line as follows:
#
#                ```bash
#                $ combine_csvs [-h] [-d DEST] datadir
#                ```
# --------------------------------------------------------------------------------------

import os
import sys
from tqdm import tqdm
import datetime as dt

# if running as script need to add the current directory to the path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
import hydr.soundtrap as soundtrap
from hydr.models.blasts_224x224_6cat import ModelRunner, Model
from hydr.utils import (load_depfile, save_depfile, unique_items, ok_to_write,
                              str_to_dt)
from hydr.types import Status


# TODO: Finish Docstrings
# TODO: If we want to scan a single file or a wav directory then we should write
#       another method to do this
def blasts_224x224_6cat(depfile: str, device: str = "cpu",
                        batch_size: int = 150) -> None:
    """
    Initializes the model runner and
    :param wavpath:
    :param device:
    :param batch_size:
    :param dest:
    :return:
    """
    deployment = load_depfile(depfile)
    already_scanned = (
        []
        if deployment.classifications.empty
        else unique_items(deployment.classifications['file'])
    )
    wavs = [
        (sn, f.file) for sn, h in deployment.hydrophones.items()
        for f in h.relevant_wavs
    ]
    model_runner = ModelRunner(device, batch_size)
    print('\nScanning wav files for blasts.')
    for i, t in enumerate(wavs):
        sn, f = t
        if f in already_scanned:
            print(f"\n{f} -- ({i + 1} of {len(wavs)})\n"
                  f"Previously scanned, skipping. ")
        else:
            print(f"\n{f} -- ({i+1} of {len(wavs)})")
            df = model_runner.scan(f)
            if not df.empty and df.loc[df['background_confidence'] < 0.5, :].empty:
                # If df isn't initally empty, but is after we remove the high
                # confidence background then we should keep the minimum background
                # confidences so system knows it scanned that file.  If it is already
                # empty then it was too short to scan and will be skipped everytime
                df = df.loc[
                     df['background_confidence'] == df['background_confidence'].min(),
                     :
                     ]
            else:
                df = df.loc[df['background_confidence'] < 0.5, :]  # reduce the set
            if deployment.convention == 'SoundTrap':
                # Fetch the details dependant on SoundTrap convention ------------------
                fs = soundtrap.dt_from_filename(f)
                sn = soundtrap.sn_from_filename(f)
                df["sn"] = sn
                df["global_start"] = df["start"].apply(
                    lambda x: dt.timedelta(seconds=x) + fs
                )
                df["global_end"] = df["end"].apply(
                    lambda x: dt.timedelta(seconds=x) + fs
                )
                # ----------------------------------------------------------------------
            else:
                # shouldn't go here, but later stages depend on these values being
                # filled out
                df["sn"] = None
                df["global_start"] = None
                df["global_end"] = None
            df['val_samples'] = None
            df['val_status'] = Status.MachineLabelled
            deployment.append_classifications(df)
            print("Saving results ...")
            save_depfile(deployment, depfile, False)
            print('Save completed.')


# TODO: May want to be able to specify which sn too
# TODO: Will need to implement additional logic to parse the validation dataframe
#       dropping bits we don't need and expanding samples with multiple validations
# TODO: Add flags to:
#         - use file basename as file
#         - to only export
#               ~ validated samples
#               ~ skipped samples
#               ~ ml classifications
# TODO: Expand the validation columns when multiple and drop unwanted columns
# TODO: Put columns in a particular order
def export_classifications(depfile: str, model: str, dest='.') -> None:
    print(f"\nExporting classifications from model `{model}` ...")
    deployment = load_depfile(depfile)
    df = deployment.validations
    for sn in tqdm(unique_items(df['sn'])):
        df_w = df.loc[np.logical_and(df['model'] == model, df['sn'] == sn), :]
        outfile = f'{dest}/{sn}_{model}.csv'
        if ok_to_write(outfile):
            df_w.to_csv(outfile, index=False)


# TODO: Turn this into a real script (i.e. remove the hardcoded values)
# TODO: Add `mapping` parameter as way to map old column names to new ones no matter
#       what they were
# TODO: Add model as an argument
# TODO: Have this method append the full path by taking the 00_hydrophone directory as
#       an input
def import_classifications(depfile: str, csvdir: str):
    deployment = load_depfile(depfile)
    csvs = [os.path.join(csvdir, f) for f in os.listdir(csvdir) if f.endswith('.csv')]
    for f in csvs:
        df = pd.read_csv(f)
        df = df.rename(
            columns={
                'serial_number': 'sn',
                'background_probability': 'background_confidence',
                'blast-1_probability': 'blast-1_confidence',
                'blast-2_probability': 'blast-2_confidence',
                'blast-3_probability': 'blast-3_confidence',
                'blast-4_probability': 'blast-4_confidence',
                'undetermined_probability': 'undetermined_confidence',
            }
        )
        if not df.empty and df.loc[df['background_confidence'] < 0.5, :].empty:
            df = df.loc[
                 df['background_confidence'] == df['background_confidence'].min(),
                 :
                 ]
        else:
            df = df.loc[df['background_confidence'] < 0.5, :]  # reduce the set
        df = df.drop(columns=['flagged', 'comment'])
        df['model'] = 'blasts_224x224_6cat'
        df['score'] = df.apply(Model.compute_score, axis=1)
        df = df.astype({'sn': str})  # cast `sn` as string
        df['global_start'] = df['global_start'].apply(str_to_dt)
        df['global_end'] = df['global_end'].apply(str_to_dt)
        df['val_samples'] = None
        df['val_status'] = Status.MachineLabelled
        deployment.append_classifications(df)
    save_depfile(deployment, depfile, False)


def export_validations(depfile: str, model: str, dest: str = '.') -> None:
    deployment = load_depfile(depfile)
    validations = deployment.validations
    validations = validations[validations['val_status'] == Status.Submitted]
    cols = ['val_samples', 'file', 'model']
    cids = {c: validations.columns.get_loc(c) for c in cols}
    vals = {'file': [], 'start': [], 'end': [], 'peak': [], 'code': [], 'comment': [],
            'model': []}
    for i in range(validations.shape[0]):
        file, samples, model = validations.iloc[
            i,
            [cids['file'], cids['val_samples'], cids['model']]
        ]
        for s in samples:
            if not (s.start is None or s.end is None):
                vals['file'].append(file)
                vals['start'].append(s.start)
                vals['end'].append(s.end)
                vals['peak'].append(s.peak)
                vals['code'].append(s.code)
                vals['comment'].append(s.comment)
                vals['model'].append(model)
    df = pd.DataFrame(vals)
    df['status'] = 'Status.Validated'

    if deployment.convention == 'SoundTrap':
        df['sn'] = df['file'].apply(lambda x: soundtrap.sn_from_filename(x))
        df['global_start'] = df.apply(
            lambda x: (
                    soundtrap.dt_from_filename(x['file']) +
                    dt.timedelta(seconds=x['start'])
            ).strftime('%Y-%m-%d %H:%M:%S'),
            axis=1
        )
        df['global_end'] = df.apply(
            lambda x: (
                soundtrap.dt_from_filename(x['file']) +
                dt.timedelta(seconds=x['end'])
            ).strftime('%Y-%m-%d %H:%M:%S'),
            axis=1
        )
    else:
        df['sn'] = None
        df['global_start'] = None
        df['global_end'] = None
    df['start'] = df['start'].apply(lambda x: '' if pd.isna(x) else '{:.3f}'.format(x))
    df['end'] = df['end'].apply(lambda x: '' if pd.isna(x) else '{:.3f}'.format(x))
    df['peak'] = df['peak'].apply(lambda x: '' if pd.isna(x) else '{:.3f}'.format(x))
    df = df[['file', 'sn', 'global_start', 'global_end', 'start', 'end', 'peak',
             'code', 'comment', 'status', 'model']]

    for sn in tqdm(unique_items(df['sn'])):
        df_w = df.loc[np.logical_and(df['model'] == model, df['sn'] == sn), :]
        outfile = f'{dest}/{sn}_{model}_validations.csv'
        if ok_to_write(outfile):
            df_w.to_csv(outfile, index=False)


def file_column_fullpaths(datadir, csvdir):
    """
    WARNING
    This method assumes a sound trap naming convention and file hierarchy.
    Also this will overwrite any existing files

    :param datadir:
    :param csvdir:
    :return:
    """
    basedir = os.path.abspath(datadir)
    dfs = {
        f'{csvdir}/{i}': pd.read_csv(f'{csvdir}/{i}')
        for i in os.listdir(csvdir)
        if i.endswith('.csv')
    }
    for file, df in dfs.items():
        df['file'] = df['file'].apply(
            lambda x: os.path.join(
                basedir,
                soundtrap.sn_from_filename(os.path.basename(x)),
                os.path.basename(x)
            )
        )
        if ok_to_write(file):
            df.to_csv(file, index=False)


def file_column_basenames(csvdir):
    dfs = {
        f'{csvdir}/{i}': pd.read_csv(f'{csvdir}/{i}')
        for i in os.listdir(csvdir)
        if i.endswith('.csv')
    }
    for file, df in dfs.items():
        df['file'] = df['file'].apply(lambda x: os.path.basename(x))
        if ok_to_write(file):
            df.to_csv(file, index=False)


def combine_csvs(csvdir: str, dest: str = None) -> None:
    """
    This method takes in the parameter `dir` representing a directory containing
    csv file(s) and combines them into a single csv.  The resulting csv is written to
    dest if

    > **Note**
    >
    > This method assumes that csv files contain the same column headers

    :param datadir: str - data directory
    :param dest: str - output directory
    :return: None
    """
    files = [
        f"{csvdir}/{f}"
        for f in os.listdir(csvdir)
        if f.endswith('csv')
    ]
    dest = (
        f'{dest}/{os.path.basename(csvdir)}_combined.csv'
        if os.path.isdir(dest)
        else dest
    )
    if not files:
        print(f"Directory `{csvdir}` doesn't contain any csv files.  ")
        return
    # get all the csv files in the directory and open them as dataframes
    dfs = [pd.read_csv(f) for f in files]
    # combine them into a single dataframe
    df = pd.concat(dfs)
    # write combined dataframe to dest
    if ok_to_write(dest):
        df.to_csv(dest, index=False)


def main():
    pass
    # d = "W:/CSIRO/00_DEPLOYMENTS/PDSKP_Indonesia_20220831/00_hydrophone_data/5476"
    # f = ("W:/CSIRO/00_DEPLOYMENTS/PDSKP_Indonesia_20220831/00_hydrophone_data/5476/"
    #     "5476.220831095230.wav")
    # blasts_224x224_6cat(f, device="cuda:0", batch_size=600)
    # import_classifications('C:/Users/maxgu/Documents/testfolder/ABC_Canada_20220101/00_hydrophone_data/deployment.data')
    # import_classifications()


if __name__ == "__main__":
    main()