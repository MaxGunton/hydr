# python standard library
import os
import sys
import datetime as dt
from pickle import load, dump

# if running as script need to add the current directory to the path
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
# from this package
import hydr as hydrophone  # for backward compatibility when unpickling
from hydr.definitions import (ContainsRestrictedCharacter, ALLOWED_FILENAME_CHARACTERS,
                              DEVICES)

# DECORATOR for classes to make them singleton
def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance


def load_depfile(depfile: str):
    with open(depfile, 'rb') as fileobj:
        deployment = load(fileobj)
    return deployment


def save_depfile(deployment, dest: str, check_overwrite=True) -> None:
    if not check_overwrite or ok_to_write(dest):
        with open(dest, 'wb') as fileobj:
            dump(deployment, fileobj)


def answered_yes(r):
    while True:
        if r in ["Y", "y", "N", "n", ""]:
            yes = True if r in ["Y", "y"] else False
            break
        else:
            r = input(f"Unknown response `{r}` please respond with Y/[N]: ")
    return yes


def str_to_dt(d):
    """
    Assumes that d follows the form yyyy-mm-dd HH:MM:SS.******
    """
    return (dt.datetime.strptime(d[:16], '%Y-%m-%d %H:%M') +
            dt.timedelta(seconds=float(d.split(':')[-1])))


def secs_to_frames(sec: float, sr: int) -> int:
    return int(round(sec * sr))


def unique_items(l):
    return sorted(list(set(l)))


def ok_to_write(file: str) -> bool:
    """
    Given the the parameter `file` this method makes sure it is ok to write it by
    returning True if it is and False if it isn't.  It does so by first checking if the
    file exists if it doesn't then it returns True, otherwise it prompts the user to ok
    overwriting the existing file.

    :param file: str - representing the path to a file
    :return: bool - True if the file is ok to write to and False it it exists and the
                    user has opted to not overwrite it.
    """
    # file doesn't exist therefore it is safe to write
    file = os.path.abspath(file)
    if not os.path.exists(file):
        print(f"\nWriting file: `{file}`")
        return True

    # verify with user that it is safe to overwrite the existing file
    r = input(f"\nDo you want to overwrite existing file:\n\t{file}\nY/[N]: ")
    overwrite = answered_yes(r)
    print(f"\nWriting file: `{file}`" if overwrite else f"\nAborting ...")
    return overwrite


# ARGPARSE CHECKS >>>
def existing(file_or_directory: str) -> str:
    """
    Checks that the file or directory exists and raises FileNotFoundError if not.

    :param file_or_directory: str - string representing a file or directory
    :return: str - the input file or directory unchanged
    """
    if not os.path.exists(file_or_directory):
        raise FileNotFoundError(file_or_directory)
    return file_or_directory


def existing_directory(directory: str) -> str:
    """
    Checks that the directory passed in exists and raises NotADirectoryError if not.

    :param directory: str - string representing a directory
    :return: str - the input directory unchanged
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(directory)
    return directory


def existing_parent(directory: str):
    """
    Checks that the parent of the directory passed in exists and raises
    NotADirectoryError if not.

    :param directory: str - string representing a file/directory
    :return: str - the input file/directory unchanged
    """
    parent, _ = os.path.split(os.path.abspath(directory))
    if not os.path.isdir(parent):
        raise NotADirectoryError(parent)
    return directory


def existing_file(file: str) -> str:
    """
    Checks that a file exists and raises FileNotFoundError if not.

    :param file: str - string representing a file
    :return: str - the input file unchanged
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(file)
    return file


def valid_filename(filename: str) -> str:
    """
    Checks that filename is restricted to the following characters ([a-zA-Z1-9_-]) and
    contains at least one of them otherwise it raises
    `hydrophone.definitions.ContainsRestrictedCharacters` exception.

    :param filename: str - string representing a filename
    :return: str - the input filename unchanged
    """
    if (
        not all([f in ALLOWED_FILENAME_CHARACTERS for f in filename])
        or len(filename) == 0
    ):
        raise ContainsRestrictedCharacter(filename)
    return filename


def valid_device(device: str) -> str:
    """
    Checks that `device` parameter is defined in `hydrophone.definitions.DEVICES`, and
    if not it raises a ValueError.  `hydrophone.definitions.DEVICES` currently includes:
    'cpu', 'cuda:0', ..., 'cuda:99'.  Although any of these values won't trigger an
    exception here it doesn't guarantee that the device exists.

    :param device: str - string representing device name
    :return: str - the input device unchanged
    """
    if device not in DEVICES:
        raise ValueError(f"Unsupported device: '{device}'. Try using 'cpu', 'cuda:0' "
                         f"..., 'cuda:99' as device instead")
    return device
# <<< ARGPARSE CHECKS

