import argparse
import os
import sys
import datetime as dt

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
# from this package
from hydr.utils import (existing_directory, existing_file, existing_parent,
                        existing, valid_filename,  valid_device, load_depfile,
                        answered_yes, unique_items, save_depfile)
from hydr.deployment import (new_project, new_depfile, export_summaries,
                             export_wav_details, set_bounds, export_bounds)
from hydr.classifications import (blasts_224x224_6cat, export_classifications,
                                  export_validations, file_column_fullpaths,
                                  file_column_basenames, combine_csvs)
from hydr.definitions import CONVENTIONS
import hydr.validator.main as validator


def initialize_args(argslist) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if 'name' in argslist:
        parser.add_argument(
            "name", help="root name of project directory", type=valid_filename
        )
    if 'datadir' in argslist:
        parser.add_argument(
            "datadir",
            help="directory containing all hydrophone data (ex. 00_hydrophone_data)",
            type=existing_directory,
        )
    if 'depfile' in argslist:
        parser.add_argument(
            "depfile",
            help="deployment data file",
            type=existing_file,
        )
    if 'wavpath' in argslist:
        parser.add_argument(
            "wavpath",
            help="individual wav file or directory containing wav files",
            type=existing,
        )
    if 'csvdir' in argslist:
        parser.add_argument(
            "csvdir",
            help="directory containing csv files",
            default=".",
            type=existing_directory,
        )
    if 'sns' in argslist:
        parser.add_argument(
            "-s",
            "--sns",
            help="serial numbers of hydrophones comma separated",
            default=None,
        )
    # can only provide one of the following two
    if 'dest' in argslist:
        parser.add_argument(
            "-d",
            "--dest",
            help="file/directory to save results",
            default=".",
            type=existing_parent,
        )
    if 'dest_dir' in argslist:
        parser.add_argument(
            "-d",
            "--dest",
            help="directory to save results",
            default=".",
            type=existing_directory,
        )
    if 'device' in argslist:
        parser.add_argument(
            '-D',
            '--device',
            help="ex. 'cpu', 'cuda:0', ...",
            default='cuda:0',
            type=valid_device,
        )
    if 'batch_size' in argslist:
        parser.add_argument(
            "-b",
            '--batch_size',
            help="number of samples in batch",
            default=600,
            type=int,
        )
    if 'convention' in argslist:
        parser.add_argument(
            "-c",
            '--convention',
            help=f"hydrophone convention used on of: "
                 f"({', '.join([f'`{c}`' for c in CONVENTIONS])})",
            default='SoundTrap',
            choices=CONVENTIONS
        )
    args = parser.parse_args()
    return args


def new_project_cli() -> None:
    """
    Wrapper for the `hydrophone.deployment.create_new` method that adds functionality to
    pass in command-line arguments.  Those arguments are:

        - `name` - root name of project directory
        - `dest` - location to create new project

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['name', 'dest_dir'])

    # call the create_new with the command line arguments
    new_project(args.name, args.dest)


def new_depfile_cli() -> None:
    """
    Wrapper for the `hydrophone.deployment.new_depfile` method that adds functionality
    to pass in command-line arguments.  Those arguments are:

        - `datadir` - directory containing individual hydrophone data
        - `dest` - directory to save summary file in

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['datadir', 'convention', 'dest'])

    # call the new_depfile method with the command-line arguments
    new_depfile(args.datadir, args.convention, args.dest)


def export_summaries_cli() -> None:
    """
    Wrapper for the `hydrophone.deployment.export_summaries` method that adds
    functionality to pass in command-line arguments.  Those arguments are:

        - `depfile` - file containing deployment data
        - `sns` - serial numbers of hydrophones
        - `dest` - directory to save summary file

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['depfile', 'sns', 'dest_dir'])
    sns = args.sns.split(',') if args.sns is not None else None

    # call the export_summaries method with the command-line arguments
    export_summaries(args.depfile, sns, args.dest)


def export_wav_details_cli() -> None:
    """
    Wrapper for the `hydrophone.deployment.export_wav_details` method that adds
    functionality to pass in command-line arguments.  Those arguments are:

        - `depfile` - file containing deployment data
        - `sns` - serial numbers of hydrophones
        - `dest` - directory to save summary file in

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['depfile', 'sns', 'dest_dir'])
    sns = args.sns.split(',') if args.sns is not None else None

    # call the generate_summary method with the command-line arguments
    export_wav_details(args.depfile, sns, args.dest)


def set_bounds_cli() -> None:
    # get command-line arguments
    args = initialize_args(['depfile'])
    deployment = load_depfile(args.depfile)
    print(f'\nDeployment bounds should be entered as a `<start>, <end>` pair.  Use `*` '
          f'to denote start/end matches bound of available audio. \n\nExamples:\n'
          f'\t2022-12-01 01:00:00, 2022-12-31 23:59:59\n'
          f'\t{" ".ljust(9)}*{" ".ljust(9)}, 2022-12-31 23:59:59\n'
          f'\t2022-01-01 00:00:00, {" ".ljust(9)}*{" ".ljust(9)}\n\n')
    for h in deployment.hydrophones.values():
        if (h.deployment_start is not None) or (h.deployment_end is not None):
            print(f'\nHydrophone `{h.sn}` has previously added bounds:')
            print(f'\t{"start".ljust(22)}{"end".ljust(22)}')
            print(f'\t{str(h.deployment_start).ljust(22)}'
                  f'{str(h.deployment_end).ljust(22)}')
            r = input(f"\nDo you want to overwrite existing bounds? Y/[N]: ")
            if answered_yes(r):
                while True:
                    bounds = input(f'Enter bounds for `{h.sn}` (comma separated):\n\t')
                    start, end = bounds.split(',')
                    start, end = start.strip(' ').rstrip(' '), end.strip(' ').rstrip(' ')
                    try:
                        start = (start
                                 if start == "*"
                                 else dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S"))
                        end = (end
                               if end == "*"
                               else dt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S"))
                        # call the set_bounds method with the command-line arguments
                        set_bounds(args.depfile, h.sn, start, end)
                    except ValueError:
                        print(f'Unable to parse bounds:\n\t{start}, {end}\nEnsure they '
                              f'follow the form specified above.  \n')
                    else:
                        break


def export_bounds_cli() -> None:
    # get command-line arguments
    args = initialize_args(['depfile', 'dest'])

    # call the generate_summary method with the command-line arguments
    export_bounds(args.depfile, args.dest)


def blast_224x224_6cat_cli() -> None:
    """
    Wrapper for the `hydrophone.blast_224x224_6cat.scan` method that adds functionality
    to pass in command-line arguments.  Those arguments are:

        - `wavpath` - wav file or directory containing wav files
        - `device` - the device to use (i.e. 'cpu', 'cuda:0', 'cuda:1', ... )
        - `batch_size` - the number of samples to process in each batch
        - `dest` - directory to save detection csvs

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['depfile', 'device', 'batch_size'])

    # call the scan method with the command-line arguments
    blasts_224x224_6cat(args.depfile, args.device, args.batch_size)


def export_classifications_cli():
    args = initialize_args(['depfile', 'dest_dir'])
    resp = 0
    deployment = load_depfile(args.depfile)
    df = deployment.validations
    if df.empty:
        print("No classifications to export.  ")
        return
    models = unique_items(df['model'])
    if len(models) > 1:
        while resp not in [str(i) for i in range(len(models))]:
            if resp != 0:
                print(f'\n`{resp}` is an invalid selection.')
            resp = input(
                '\nPlease select classification set to export:\n\t' +
                '\n\t'.join([f'{str(i).ljust(5)}-> {m}' for i, m in enumerate(models)])
                + '\n\n\t'
            )
    model = models[int(resp)]
    export_classifications(args.depfile, model, args.dest)


def export_validations_cli():
    args = initialize_args(['depfile', 'dest_dir'])
    resp = 0
    deployment = load_depfile(args.depfile)
    df = deployment.validations
    if df.empty:
        print("No validations to export.  ")
        return
    models = unique_items(df['model'])
    if len(models) > 1:
        while resp not in [str(i) for i in range(len(models))]:
            if resp != 0:
                print(f'\n`{resp}` is an invalid selection.')
            resp = input(
                '\nPlease select validation set to export:\n\t' +
                '\n\t'.join([f'{str(i).ljust(5)}-> {m}' for i, m in enumerate(models)])
                + '\n\n\t'
            )
    model = models[int(resp)]
    export_validations(args.depfile, model, args.dest)


def file_column_fullpaths_cli():
    # get command-line arguments
    args = initialize_args(['datadir', 'csvdir'])

    # call the generate_summary method with the command-line arguments
    file_column_fullpaths(args.datadir, args.csvdir)


def file_column_basenames_cli():
    # get command-line arguments
    args = initialize_args(['csvdir'])

    # call the generate_summary method with the command-line arguments
    file_column_basenames(args.csvdir)


def combine_csvs_cli() -> None:
    """
    Wrapper for the `hydrophone.detections.combine_csvs` method that adds functionality
    to pass in command-line arguments.  Those arguments are:

        - `datadir` - directory containing csv files to combine
        - `dest` - file or directory to save combined csv file

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['csvdir', 'dest'])

    # call the generate_summary method with the command-line arguments
    combine_csvs(args.csvdir, args.dest)


def validator_cli():
    # get command-line arguments
    args = initialize_args(['depfile'])
    # call the generate_summary method with the command-line arguments
    validator.run(args.depfile)
