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
                        answered_yes, unique_items)
from hydr.deployment import (new_project, new_depfile, export_summaries,
                             export_wav_details, set_bounds, set_region,
                             set_organization)
from hydr.classifications import (blasts_224x224_6cat, export_classifications,
                                  export_validations, file_column_fullpaths,
                                  file_column_basenames, combine_csvs)
from hydr.plots import hour_plot, weekday_plot, date_plot, week_plot
from hydr.mapping import set_coords, export_map
from hydr.definitions import CONVENTIONS
import hydr.validator.main as validator


def initialize_args(argslist) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if 'all_samples' in argslist:
        parser.add_argument(
            "-a",
            '--all_samples',
            help="Use all ML classifications in plot instead of only validated samples",
            default=False,
            type=bool,
            action=argparse.BooleanOptionalAction,
        )
    if 'batch_size' in argslist:
        parser.add_argument(
            "-b",
            '--batch_size',
            help="number of samples in batch",
            default=600,
            type=int,
        )
    if 'colors' in argslist:
        parser.add_argument(
            "-c",
            '--colors',
            help="color mapping to each sample class. ex. code1,#449594/code2,#FF00FF "
                 "or single value #FF0000 to apply to all codes",
            default=None,
            type=str,
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
    if 'csvdir' in argslist:
        parser.add_argument(
            "csvdir",
            help="directory containing csv files",
            default=".",
            type=existing_directory,
        )
    if 'csvfile' in argslist:
        parser.add_argument(
            "csv",
            help="samples file",
            type=existing_file,
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
    # can only provide one of the following two
    if 'dest' in argslist:
        parser.add_argument(
            "-d",
            "--dest",
            help="file/directory to save results",
            default=".",
            type=existing_parent,
        )
    if 'destdir' in argslist:
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
    if 'labels' in argslist:
        parser.add_argument(
            "-l",
            '--labels',
            help="labels from samples to use in plot (comma separated no spaces)",
            default=None,
            type=str,
        )
    if 'multifile' in argslist:
        parser.add_argument(
            "-f",
            '--multifile',
            help="if set each code is plotted in separate file (overrides multiplot)",
            default=False,
            type=bool,
            action=argparse.BooleanOptionalAction,
        )
    if 'multiplot' in argslist:
        parser.add_argument(
            "-m",
            '--multiplot',
            help="if set each code plotted separately in single file (overridden by "
                 "multifile)",
            default=False,
            type=bool,
            action=argparse.BooleanOptionalAction,
        )
    if 'name' in argslist:
        parser.add_argument(
            "name",
            help="root name of project directory",
            type=valid_filename
        )
    if 'sns' in argslist:
        parser.add_argument(
            "-s",
            "--sns",
            help="serial numbers of hydrophones comma separated",
            default=None,
        )
    if 'wavpath' in argslist:
        parser.add_argument(
            "wavpath",
            help="individual wav file or directory containing wav files",
            type=existing,
        )
    args = parser.parse_args()
    return args


def select_model(depfile):
    resp = 0
    deployment = load_depfile(depfile)
    df = deployment.validations
    if df.empty:
        print("No samples for export.  ")
        return
    models = unique_items(df['model'])
    if len(models) > 1:
        while resp not in [str(i) for i in range(len(models))]:
            if resp != 0:
                print(f'\n`{resp}` is an invalid selection.')
            resp = input(
                '\nPlease select model for export:\n\t' +
                '\n\t'.join([f'{str(i).ljust(5)}-> {m}' for i, m in enumerate(models)])
                + '\n\n\t'
            )
    return models[int(resp)]


def parse_colors(colors):
    if colors is not None:
        colors = colors.split('/')
        if len(colors) == 1 and len(colors[0].split(',')) == 1:
            colors = colors[0]
        else:
            colors = {cc.split(',')[0]: cc.split(',')[1] for cc in colors}
    return colors


def new_project_cli() -> None:
    """
    Wrapper for the `hydrophone.deployment.create_new` method that adds functionality to
    pass in command-line arguments.  Those arguments are:

        - `name` - root name of project directory
        - `dest` - location to create new project

    :return: None
    """
    # get command-line arguments
    args = initialize_args(['name', 'destdir'])

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
    args = initialize_args(['depfile', 'sns', 'destdir'])
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
    args = initialize_args(['depfile', 'sns', 'destdir'])
    sns = args.sns.split(',') if args.sns is not None else None

    # call the generate_summary method with the command-line arguments
    export_wav_details(args.depfile, sns, args.dest)


def set_region_cli() -> None:
    args = initialize_args(['depfile'])
    set_region(args.depfile)


def set_organization_cli() -> None:
    args = initialize_args(['depfile'])
    set_organization(args.depfile)


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


def set_coords_cli():
    # get command-line arguments
    args = initialize_args(['depfile'])

    # call the create_project_directory with the arguments passed
    set_coords(args.depfile)


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


def export_classifications_cli() -> None:
    args = initialize_args(['depfile', 'destdir'])
    model = select_model(args.depfile)
    export_classifications(args.depfile, model, args.dest)


def export_validations_cli() -> None:
    args = initialize_args(['depfile', 'destdir'])
    model = select_model(args.depfile)
    export_validations(args.depfile, model, args.dest)


def export_map_cli() -> None:
    args = initialize_args(['depfile', 'destdir'])

    export_map(args.depfile, args.dest)


def file_column_fullpaths_cli() -> None:
    # get command-line arguments
    args = initialize_args(['datadir', 'csvdir'])

    # call the generate_summary method with the command-line arguments
    file_column_fullpaths(args.datadir, args.csvdir)


def file_column_basenames_cli() -> None:
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


def validator_cli() -> None:
    # get command-line arguments
    args = initialize_args(['depfile'])
    # call the generate_summary method with the command-line arguments
    validator.run(args.depfile)


#  -------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------
# TODO: Integrate these methods from mtools
# # def combine_overlapping_cli(samples_file: SamplesFile,
# #                             outfile: Outfile=None) -> None:
# #     combine_overlapping(samples_file, outfile)
# def combine_overlapping_cli() -> None:
#     """
#     Wrapper adding commandline parsing to mtools.core.combine_overlapping function.
#
#     See documentation for [mtools.core.combine_overlapping](core.html#combine_overlapping) for
#     more details.
#     """
#     # get command-line arguments
#     args = initialize_args(['csvfile', 'dest'])
#     # call the combine_overlapping_method
#     combine_overlapping(args.csv, args.dest)
#
#
#
# # def multicodes_cli(samples_file: SamplesFile,
# #                    outfile: Outfile=None) -> None:
# #     multicodes(samples_file, outfile)
# def multicodes_cli() -> None:
#     """
#     Wrapper adding commandline parsing to mtools.core.multicodes function.
#
#     See documentation for [mtools.core.multicodes](core.html#multicodes) for
#     more details.
#     """
#     args = initialize_args(['csvfile', 'dest'])
#     multicodes(args.csv, args.dest)

#  -------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------


def hour_plot_cli() -> None:
    args = initialize_args(['depfile', 'labels', 'model', 'all_samples', 'multifile',
                            'multiplot', 'colors', 'destdir'])
    model = select_model(args.depfile)
    labels = None if args.labels is None else args.labels.split(",")  # turn into a list
    colors = parse_colors(args.colors)

    print(f"depfile: {args.depfile}")
    print(f"model: {model}")
    print(f"labels: {labels}")
    print(f"all_samples: {args.all_samples}")
    print(f"multifile: {args.multifile}")
    print(f"multiplot: {args.multiplot}")
    print(f"colors: {colors}")
    print(f"dest: {args.dest}")

    hour_plot(args.depfile, model, labels, args.all_samples, args.multifile,
              args.multiplot, colors, args.dest)


def weekday_plot_cli() -> None:
    args = initialize_args(['depfile', 'labels', 'model', 'all_samples', 'multifile',
                            'multiplot', 'colors', 'destdir'])
    model = select_model(args.depfile)
    labels = None if args.labels is None else args.labels.split(",")  # turn into a list
    colors = parse_colors(args.colors)

    print(f"depfile: {args.depfile}")
    print(f"model: {model}")
    print(f"labels: {labels}")
    print(f"all_samples: {args.all_samples}")
    print(f"multifile: {args.multifile}")
    print(f"multiplot: {args.multiplot}")
    print(f"colors: {colors}")
    print(f"dest: {args.dest}")

    weekday_plot(args.depfile, model, labels, args.all_samples, args.multifile,
                 args.multiplot, colors, args.dest)


def date_plot_cli() -> None:
    args = initialize_args(['depfile', 'labels', 'model', 'all_samples', 'multifile',
                            'multiplot', 'colors', 'destdir'])
    model = select_model(args.depfile)
    labels = None if args.labels is None else args.labels.split(",")  # turn into a list
    colors = parse_colors(args.colors)

    print(f"depfile: {args.depfile}")
    print(f"model: {model}")
    print(f"labels: {labels}")
    print(f"all_samples: {args.all_samples}")
    print(f"multifile: {args.multifile}")
    print(f"multiplot: {args.multiplot}")
    print(f"colors: {colors}")
    print(f"dest: {args.dest}")

    date_plot(args.depfile, model, labels, args.all_samples, args.multifile,
              args.multiplot, colors, args.dest)


def week_plot_cli() -> None:
    args = initialize_args(['depfile', 'labels', 'model', 'all_samples', 'multifile',
                            'multiplot', 'colors', 'destdir'])
    model = select_model(args.depfile)
    labels = None if args.labels is None else args.labels.split(",")  # turn into a list
    colors = parse_colors(args.colors)

    print(f"depfile: {args.depfile}")
    print(f"model: {model}")
    print(f"labels: {labels}")
    print(f"all_samples: {args.all_samples}")
    print(f"multifile: {args.multifile}")
    print(f"multiplot: {args.multiplot}")
    print(f"colors: {colors}")
    print(f"dest: {args.dest}")

    week_plot(args.depfile, model, labels, args.all_samples, args.multifile,
              args.multiplot, colors, args.dest)
