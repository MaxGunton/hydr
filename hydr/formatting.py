# python standard library
import os
import sys
import datetime as dt
from collections import OrderedDict

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

# from this package
from hydr.definitions import LINE_LENGTH, LABEL_WIDTH


def wrap_long_line(line: str) -> str:
    """
    This method is used to wrap dictionary value strings into multiple lines if they are
    longer than `(LINE_LENGTH - LABEL_WIDTH)`.  It will break the string into parts <=
    (LINE_LENGTH - LABEL_WIDTH) either by nearest ',' if it contains any or '/'
    otherwise.  If it contains neither the string will return unchanged.

    > **Note**
    >
    > This method assumes that long lines are going to contain `,` or `/` (i.e. they
    > are filenames or lists/dictionaries).  And it uses these to split up the line in a
    > place where it makes sense and doesn't interfere with readability.

    :param line: str - represents a value for a dictionary key
    :return: str - value line split into pieces with '\n' and LABEL_WIDTH padding at
                   front of subsequent lines.
    """
    lines = line.split("\n")
    new_lines = []
    for line in lines:
        line = line.strip()
        split_on = "," if "," in line else "/"
        lp = line.split(split_on)
        rt = 0
        new_line = []
        while True:
            for idx, p in enumerate(lp):
                rt += len(p) + 1
                if rt > (LINE_LENGTH - LABEL_WIDTH):
                    rt = 0
                    idx = 1 if idx == 0 else idx
                    break
            else:
                idx = len(lp)
            new_line.append(lp[0:idx])
            lp = lp[idx:]
            if not lp:
                break
        new_line = [split_on.join(nl) for nl in new_line]
        new_lines.append(f'{split_on}\n{" ".ljust(LABEL_WIDTH)}'.join(new_line))
    return f'\n{" ".ljust(LABEL_WIDTH)}'.join(new_lines)


def format_file_list(files: str) -> str:
    files = [f.replace('\\', '/')for f in files]
    files = ["None"] if len(files) == 0 else files
    return f'\n{", ".ljust(LABEL_WIDTH)}'.join(files)


def format_utc_offset(offset):
    prefix = "UTC" if offset < 0 else "UTC+"
    return f"{prefix}{offset}".rstrip("0").rstrip("0")


def strformat_dict(layout: OrderedDict) -> str:
    """
    Given the `layout` parameter, this method formats it into a nice human readable
    string to display or write.  Headings are seperated with `LINE_LENGTH` number of '.'
    characters and are underlined with '=' characters.  Items are left justified to be
    `LABEL_WIDTH` characters wide and wrapping if attempted on values longer than
    `LINE_LENGTH` - `LABEL_WIDTH`

    :param layout: OrderedDict - takes the form:
                   OrderedDict[<heading>: str, List[Tuple(<item>: str, <value>: str)]]

    :return: str - human-readable string containing the dictionary contents ready to
                   write or display
    """
    fstring = ""
    for k, v in layout.items():
        fstring += f'{"." * LINE_LENGTH}\n{k}\n{"-" * len(k)}\n' if k != "" else ""
        for h, i in v:
            i = (
                wrap_long_line(str(i))
                if len(str(i)) > (LINE_LENGTH - LABEL_WIDTH)
                else i
            )
            fstring += f"{h.ljust(LABEL_WIDTH - 2)}: {i}\n"
    return fstring


def strformat_seconds(seconds: float) -> str:
    """
    Format seconds from a float to a string of the following form:

        f'{:.2f} seconds / {dt.timedelta(seconds=float)}'

    :param seconds: float - seconds
    :return: str - string representation of seconds
    """
    fs = "{:.2f} seconds / {}" if seconds > 0 else "-{:.2f} seconds / -{}"
    return fs.format(abs(seconds), str(dt.timedelta(seconds=abs(seconds)))[:-4])


def format_deployment_summary(data):
    """
    define the layout as an ordered dictionary of the form:
    dict[<heading>:str, List[Tuple[<item>:str, <value>:str]]]

    :param data:
    :return:
    """
    layout = OrderedDict(
        [
            (
                "General",
                [
                    ("Extension Counts", data['ext_counts']),
                    ("Unused Wavs", format_file_list(data['unused_wavs'])),
                    ("Samplerate(s)", data['sr_counts']),
                    ("UTC Offset(s)", {
                        format_utc_offset(k): v
                        for k, v in data['utc_offset_counts'].items()
                    }),
                    ("Gain(s) Used", data['gain_counts']),
                    ("Audio Start", data['audio_start'].strftime('%Y-%m-%d %H:%M:%S')),
                    ("Audio End", data['audio_end'].strftime('%Y-%m-%d %H:%M:%S')),
                    ("Deployment Start", data['deployment_start'].strftime('%Y-%m-%d %H:%M:%S')),
                    ("Deployment End", data['deployment_end'].strftime('%Y-%m-%d %H:%M:%S')),
                    ("Total Audio", strformat_seconds(data['total_audio'])),
                ],
            ),
            (
                "File Issues",
                [("Wavfiles", format_file_list(data['bad_wavfiles'])),
                 ("Logfiles", format_file_list(data['bad_logfiles']))],
            ),
            (
                "Audio Coverage Gaps",
                [("Mean", strformat_seconds(data['mean_gap'])),
                 ("Median", strformat_seconds(data['median_gap'])),
                 ("Max", strformat_seconds(data['max_gap']))],
            ),
            (
                "Audio File Lengths",
                [("Mean", strformat_seconds(data['mean_duration'])),
                 ("Median", strformat_seconds(data['median_duration']))],
            ),
        ]
    )
    return layout


def summary_report(deployment_details, device_details=None):
    deployment_summary = format_deployment_summary(deployment_details)
    title = f"HYDROPHONE SUMMARY - (SN: {deployment_details['sn']})"
    sections = [("DEPLOYMENT DATA", strformat_dict(deployment_summary))]
    if device_details:
        sections.append(("DEVICE AND CALIBRATION", strformat_dict(device_details)))
    summary = f"{title}\n"
    for heading, contents in sections:
        summary += f'{"-" * (LINE_LENGTH + 12)}\n'
        summary += f'{heading}\n{"=" * len(heading)}\n'
        summary += contents
    summary += f'{"-" * (LINE_LENGTH + 12)}\n'
    return summary


