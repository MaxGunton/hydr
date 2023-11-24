import pandas as pd
import numpy as np
import os
import sys
import datetime as dt
from tqdm import tqdm
from collections import OrderedDict

from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
import seaborn as sns

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from hydr.types import Deployment
from hydr.utils import load_depfile, ok_to_write, hex_to_rgb, balanced_lines
from hydr.classifications import extract_validations

from warnings import warn
from typing import List


def load_outfiles(dest, metadata: dict, codes: List[str], multifile: bool,
                  multiplot: bool, plot_type: str) -> List[str]:
    """
    This function is called when no outfile is specified.  It returns the list of
    outfiles that correspond to each code if multifile parameter is set, or a single
    file if it isn't.
    """
    files = metadata['wav_files']
    serial_nums_str = "&".join(
        sorted(list(set([metadata[file]['serial_number'] for file in files])))
    )
    codes = [c.replace('/', '-') for c in codes]
    if multifile:
        out = [f"{dest}/{'_'.join([serial_nums_str, c, plot_type])}.png" for c in codes]
    else:
        # This will be the case where we have a single file
        codes.sort()  # sort codes, so listed alphabetically
        codes_str = "&".join(codes)
        combined_str = None
        if len(codes) > 1:
            combined_str = 'separated' if multiplot else 'combined'
        filename_list = (
            [serial_nums_str, codes_str, plot_type] if combined_str is None else
            [serial_nums_str, codes_str, combined_str, plot_type]
        )
        out = [f"{dest}/{'_'.join(filename_list)}.png"]
    return out


def load_colors(colors, codes, multifile, multiplot):
    """
    We know that the values are good already. Load the defaults if required and warn if
    """
    # pigeon hole principle to ensure palette large enough if user provided colors all
    # match our palette
    codes.sort()  # sort the codes (for consistency)
    d_palette = sns.color_palette("colorblind", len(codes)*2)
    d_colors = OrderedDict([(code, d_palette[idx])for idx, code in enumerate(codes)])

    if colors is None or not colors:  # will also catch empty dictionary
        if not multifile and multiplot:
            # all different
            colors_f = OrderedDict([(code, d_colors[code]) for code in codes])
        else:
            # all same
            colors_f = OrderedDict([(code, d_colors[code]) for code in codes])
    elif not issubclass(type(colors), dict):
        if type(colors) is str:
            colors = hex_to_rgb(colors)  # convert to rgb
        if not multifile and multiplot and (len(codes) > 1):
            warn("Current configuration has multiplot set and more than one code of "
                 "interest.  However, only one color was provided and therefore all of "
                 "the codes will share this color.  This is most likely a mistake.  If "
                 "you don't supply a color then the default action is to assign unique "
                 "colors for each code.  This will be done automatically for you!  ")
            colors_f = d_colors
        else:
            # all colors get the same value
            colors_f = OrderedDict([(code, colors) for code in codes])
    else:  # colors is a non empty dictionary
        if type(list(colors.values())[0]) is str:  # convert to rgb
            colors = OrderedDict([(code, hex_to_rgb(colors[code])) for code in colors])
        if not multifile and multiplot:
            # check for duplicate colors
            color_list = sorted([(c, colors[c]) for c in colors], key=lambda x: x[1])
            all_duplicate_sets = []
            while len(color_list) > 0:
                duplicate_set = ''
                code, color = color_list.pop()  # off the back
                next_color = color_list[-1][1] if len(color_list) > 0 else None
                if color == next_color:
                    duplicate_set += ('color = {}\tcodes = {}'.format(color, code)
                                      if duplicate_set == '' else ', {}'.format(code))
                elif duplicate_set != '':
                    duplicate_set += ', {}'.format(code)
                    all_duplicate_sets.append(duplicate_set)
            if len(all_duplicate_sets) > 0:
                duplicates = 'n\t'.join(all_duplicate_sets)
                warn(("Current configuration has multiplot set and more than one code "
                      "of interest.  User supplied at least two identical colors for "
                      "different codes, this is most likely a mistake.  Execution will "
                      "continue, but be aware that codes have matching colors and will "
                      "be indistinguishable:\n\n\t{}").format(duplicates))
            # unique colors
            u_colors = [color for color in d_palette if color not in colors.values()]
            colors_f = OrderedDict(
                [(code, colors[code]) if code in colors else (code, u_colors.pop(0))
                 for code in codes]
            )
        else:
            colors_f = OrderedDict(
                [(code, colors[code]) if code in colors else (code, d_colors[0])
                 for code in codes]
            )

    return colors_f


def load_plot_params(deployment: Deployment, model: str, sn: str, codes=None,
                     colors=None, all_samples=False, multifile: bool = False,
                     multiplot: bool = False, dest=None, plot_type: str = ''):

    df = extract_validations(deployment) if not all_samples else deployment.validations
    df = df.loc[np.logical_and(df['model'] == model, df['sn'] == sn), :]
    df['file'] = df['file'].apply(lambda x: os.path.basename(x))

    codes = list(set(df['code'])) if codes is None else codes
    colors = load_colors(colors, codes, multifile, multiplot)

    hydrophone = deployment.hydrophones[sn]
    wav_files = hydrophone.all_wavs

    location = '' if deployment.region is None else deployment.region
    organization = '' if deployment.organization is None else deployment.organization
    utc_offset = set([wf.utc_offset for wf in wav_files])
    if len(utc_offset) > 1:
        warn('Multiple UTC offsets detected for single hydrophone plot.  ')

    # 2) Load the metadata if it is valid
    metadata = dict(
        deployment_date=hydrophone.deployment_start,
        organization_name=organization,
        location_name=location,
        wav_files=[os.path.basename(f.file) for f in wav_files if not f.corrupted],
        utc_offset=utc_offset.pop(),
        deployment_start=hydrophone.deployment_start,
        deployment_end=hydrophone.deployment_end,
    )
    for f in wav_files:
        if not f.corrupted:
            metadata[os.path.basename(f.file)] = dict(
                global_start=f.start,
                serial_number=hydrophone.sn,
                path_to_file=f.file,
                frames=f.frames,
                sr=f.sr,
                duration=f.duration,
                global_end=f.end,
                utc_offset=f.utc_offset
            )

    # 6) Load outfile if valid dropping, outfiles for any non existent codes or setup
    #    with default outfiles
    outfiles = load_outfiles(dest, metadata, codes, multifile, multiplot, plot_type)

    return dict(
        df=df,
        metadata=metadata,
        codes=codes,
        colors=colors,
        multifile=multifile,
        multiplot=multiplot,
        outfiles=outfiles
    )


def write_bar_graph(title: str, bar_labels: pd.DataFrame, bar_sets: list, outfile: str,
                    xlabel: str) -> None:
    """
    Finish the docs here
    """
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Occurrences")

    # set the bar labels (center them on the thickest bar)
    bar_labels['x_pos'] = (bar_labels['x_pos'] +
                           (max([width for _, width, _, _ in bar_sets]) / 2))
    ax.set_xticks(bar_labels['x_pos'])
    ax.set_xticklabels(bar_labels['text'], minor=False)
    plt.xticks(rotation=90, ha='center')

    highest_bar = max([max(heights) for heights, _, _, _ in bar_sets])

    # put these into a list for each subplot
    bars = []
    for bar_set in bar_sets:
        bar_heights, bar_width, bar_color, codes_str = bar_set

        # increase value for number labels (i.e. darker shade, but same hue as bar)
        hsv_bar_color = mplcolors.rgb_to_hsv(bar_color)
        hsv_bar_color[2] = max(hsv_bar_color[2] - 0.25, 0.0)
        label_color = mplcolors.hsv_to_rgb(hsv_bar_color)

        # put our bars into the plot (shift them so label is centered)
        bars.append(ax.bar((bar_labels['x_pos']), bar_heights, bar_width,
                           color=bar_color))
        # offset is relative to the highest of all bars (was 100)
        offset = highest_bar / 30
        for x, y in enumerate(bar_heights):
            ax.text(bar_labels.iloc[x, :]['x_pos'], y+offset, str(y), rotation=90,
                    color=label_color, va='center', ha='center', fontweight='bold')
    if len(bars) > 1:  # if multiplot then create a legend
        ax.legend(bars, [codes_str for _, _, _, codes_str in bar_sets])

    if ok_to_write(outfile):
        plt.savefig(outfile, dpi=300, format='png', bbox_inches='tight')  # dpi=300
    return


def occurance_plot(plot_params: dict) -> None:
    dbw = 0.75  # default bar width
    sbw = 0.1  # skinniest bar width
    mbw = 0.9  # max bar width

    # 1) Load the plot parameters
    (df, metadata, codes, colors, multifile, multiplot, outfiles, xlabel, bar_labels,
     plot_type) = plot_params.values()

    org_name = metadata['organization_name']
    location_name = metadata['location_name']
    years = "-".join(sorted(list({str(metadata['deployment_start'].year),
                                  str(metadata['deployment_end'].year)})))
    serial_nums = ", ".join(
        sorted(list(set([metadata[f]['serial_number'] for f in metadata['wav_files']])))
    )
    # capitalize, change underscore to space, and sort
    codes_str = sorted([code.capitalize().replace('_', ' ') for code in codes])
    all_codes_str = codes_str[0] if len(codes_str) == 1 else f"({', '.join(codes_str)})"
    all_codes_str = balanced_lines(all_codes_str, max_chars=40)  # wraps long strings

    bar_sets = []
    title = ''
    for idx, c in enumerate(codes):
        df_rel = (df.loc[df['code'] == c, :] if multifile or multiplot
                  else df.loc[df['code'].isin(codes), :])
        title_codes_str = codes_str[idx] if multifile else all_codes_str
        ylabel_codes_str = codes_str[idx] if multifile or multiplot else all_codes_str
        # find the closest space character to 50

        # May always want to include the UTC offset in plot title
        title = (f"{org_name} -- {location_name} {years} "
                 f"(SN: {serial_nums})\n{title_codes_str} / {plot_type}")
        # 3) create single bar set
        bar_heights = [df_rel[df_rel['bar'] == row['x_pos']].shape[0]
                       for i, row in bar_labels.iterrows()]
        bar_width = dbw
        bar_color = colors[c]
        bar_sets.append((bar_heights, bar_width, bar_color, ylabel_codes_str))

        if not multiplot:
            # 4) get the outfile
            outfile = outfiles[idx]
            write_bar_graph(title, bar_labels.copy(deep=True), bar_sets, outfile,
                            xlabel=xlabel)
            bar_sets = []
        if not multifile and not multiplot:
            break
    if not multifile and multiplot:
        # 4) sort by occurrences and set bar sizes
        dw = (mbw-sbw) / len(codes)
        bar_sets.sort(key=lambda x: sum(x[0]), reverse=True)
        bar_sets = [(b[0], 0.9-dw*idx, b[2], b[3]) for idx, b in enumerate(bar_sets)]
        write_bar_graph(title, bar_labels.copy(deep=True), bar_sets, outfiles[0],
                        xlabel=xlabel)
    return


def hour_plot(depfile, model, labels, all_samples, multifile, multiplot, colors,
              dest) -> None:
    hours = [u'12:00am', u'1:00am', u'2:00am', u'3:00am', u'4:00am', u'5:00am',
             u'6:00am', u'7:00am', u'8:00am', u'9:00am', u'10:00am', u'11:00am',
             u'12:00pm', u'1:00pm', u'2:00pm', u'3:00pm', u'4:00pm', u'5:00pm',
             u'6:00pm', u'7:00pm', u'8:00pm', u'9:00pm', u'10:00pm', u'11:00pm']
    deployment = load_depfile(depfile)
    df = deployment.validations
    for sn in tqdm(df['sn'].unique()):
        plot_params = load_plot_params(
            deployment, model, sn, labels, colors, all_samples, multifile,
            multiplot, dest, 'events_per_hour'
        )
        utc_o = plot_params['metadata']['utc_offset']
        plot_params['xlabel'] = "Hour of Day"
        # 2) set up bar labels could send this in
        plot_params['bar_labels'] = pd.DataFrame({'x_pos': np.arange(len(hours)),
                                                  'text': hours})
        plot_params['plot_type'] = "Hour [UTC{}]".format(
            str(utc_o) if utc_o < 0 else '+' + str(utc_o)
        )

        # add bar column to dataset (i.e. bar postion corresponds to the hours of day
        # list above in this case)
        def add_bar_column(row):
            return (plot_params['metadata'][row['file']]['global_start'] +
                    dt.timedelta(seconds=row['start'])).hour
        plot_params['df'].loc[:, 'bar'] = plot_params['df'].apply(add_bar_column,
                                                                  axis=1)
        occurance_plot(plot_params)


def weekday_plot(depfile, model, labels, all_samples, multifile, multiplot, colors,
                 dest) -> None:
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                'Sunday']
    deployment = load_depfile(depfile)
    df = deployment.validations
    for sn in tqdm(df['sn'].unique()):
        plot_params = load_plot_params(
            deployment, model, sn, labels, colors, all_samples, multifile,
            multiplot, dest, 'events_per_weekday'
        )
        plot_params['xlabel'] = "Weekday"
        # 2) set up bar labels could send this in
        plot_params['bar_labels'] = pd.DataFrame(
            {'x_pos': np.arange(len(weekdays)), 'text': weekdays})
        plot_params['plot_type'] = "Weekday"

        # add bar column to dataset (i.e. bar postion corresponds to the hours of day
        # list above in this case)
        def add_bar_column(row):
            return (plot_params['metadata'][row['file']]['global_start'] + dt.timedelta(
                seconds=row['start'])).weekday()

        plot_params['df'].loc[:, 'bar'] = plot_params['df'].apply(add_bar_column,
                                                                  axis=1)
        occurance_plot(plot_params)


def date_plot(depfile, model, labels, all_samples, multifile, multiplot, colors,
              dest) -> None:
    deployment = load_depfile(depfile)
    df = deployment.validations
    for sn in tqdm(df['sn'].unique()):
        plot_params = load_plot_params(
            deployment, model, sn, labels, colors, all_samples, multifile,
            multiplot, dest, 'events_per_date'
        )
        st, end = plot_params['metadata']['deployment_start'], plot_params['metadata'][
            'deployment_end']
        dates = [(st + dt.timedelta(days=i)).isoformat().split('T')[0] for i in
                 range((end.date() - st.date()).days + 1)]
        plot_params['xlabel'] = "Date"
        # 2) set up bar labels could send this in
        plot_params['bar_labels'] = pd.DataFrame(
            {'x_pos': np.arange(len(dates)), 'text': dates})
        plot_params['plot_type'] = "Date"

        # add bar column to dataset (i.e. bar postion corresponds to the hours of day
        # list above in this case)
        def add_bar_column(row):
            return dates.index(
                (plot_params['metadata'][row['file']]['global_start'] +
                 dt.timedelta(seconds=row['start'])).isoformat().split('T')[0]
            )
        plot_params['df'].loc[:, 'bar'] = plot_params['df'].apply(add_bar_column,
                                                                  axis=1)
        occurance_plot(plot_params)


def week_plot(depfile, model, labels, all_samples, multifile, multiplot, colors,
              dest) -> None:
    deployment = load_depfile(depfile)
    df = deployment.validations
    for sn in tqdm(df['sn'].unique()):
        plot_params = load_plot_params(
            deployment, model, sn, labels, colors, all_samples, multifile,
            multiplot, dest, 'events_per_date'
        )
        st, end = (plot_params['metadata']['deployment_start'],
                   plot_params['metadata']['deployment_end'])
        num_days = (end.date() - st.date()).days
        sow = (
            [st] + [st.replace(hour=0, minute=0, second=0) + dt.timedelta(days=i)
                    for i in range(1, num_days + 1)
                    if (st.date() + dt.timedelta(days=i)).weekday() == 0]
        )
        eow = (
            [st.replace(hour=23, minute=59, second=59) + dt.timedelta(days=i)
             for i in range(num_days)
             if (st.date() + dt.timedelta(days=i)).weekday() == 6] + [end]
        )
        assert len(sow) == len(eow)
        weeks = [
            sow[i].strftime('%b %d') + "\n(1 day)"
            if sow == eow
            else sow[i].strftime('%b %d') + ' - ' + eow[i].strftime('%b %d') +
            "\n({} days)".format((eow[i].date() - sow[i].date()).days + 1)
            for i in range(len(sow))
        ]
        plot_params['xlabel'] = "Week"
        # 2) set up bar labels could send this in
        plot_params['bar_labels'] = pd.DataFrame(
            {'x_pos': np.arange(len(weeks)), 'text': weeks})
        plot_params['plot_type'] = "Week"

        # add bar column to dataset (i.e. bar position corresponds to the hours of day
        # list above in this case)
        def add_bar_column(row):
            event_time = (
                plot_params['metadata'][row['file']]['global_start'] +
                dt.timedelta(seconds=row['start'])
            )
            for idx in range(len(sow)):
                if sow[idx] <= event_time <= eow[idx]:
                    return idx
            return
        plot_params['df'].loc[:, 'bar'] = plot_params['df'].apply(
            add_bar_column, axis=1
        )
        occurance_plot(plot_params)
