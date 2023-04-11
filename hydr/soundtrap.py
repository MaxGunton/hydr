# python standard library
import os
import datetime as dt
import json
from collections import Counter, ChainMap, OrderedDict
from xml.etree import ElementTree
from urllib.request import urlopen
from urllib.error import URLError
from typing import Union, Dict, Any

DEVICE_PROPERTIES = {
    "dateCreated": "",
    "operator": "",
    "modelName": "",
    "serialNo": "",
    "hardwareSerial": "",
    "hpSerial": "",
    "refLevel": "",
    "refSerial": "",
    "highFreq": "",
    "lowFreq": "",
    "tone": "",
    "calType": "",
    "refDevice": "",
    "model": "",
}


def follows_conventions(file: str) -> bool:
    """
    Given input parameter `file` representing a filename, this method will return True
    if the file follows the SoundTrap naming conventions otherwise it returns False.
    Following the SoundTrap naming convention means that the files must pass the
    following 2 checks:

    1) When name is split by `.` it must have at least 3 sections
    2) The second section must represent a valid datetime such that is can be
       interpreted as `yymmddHHMMSS`

    :param file: str - Representing a filename
    :return: bool - reprsenting whether the input parameter follows the conventions
    """

    f = os.path.basename(file).split('.')
    if len(f) < 3:
        return False
    try:
        dt.datetime.strptime(f[1], '%y%m%d%H%M%S')
    except ValueError:
        return False
    return True


def utc_offset_from_logfile(logfile: str) -> Union[float, None]:
    """
    Given the parameter `logfile` this method parses it for the `SamplingStartTimeUTC`
    value and computes the difference between it and the datetime encoded in the
    filename to get the UTC offset.  If it is unable to do so because of any of the
    following:

        - the file doesn't exist
        - the file isn't formatted as expected
        - the file doesn't contain the `SamplingStartTimeUTC` value

    it will simply return None.

    :param logfile: str - path to the logfile
    :return: Union[float, None] - the UTC offset as computed by the difference between
                                  `SamplingStartTimeUTC` in the logfile and the datetime
                                   encoded in the filename or None if there is an issue
                                   collecting it.
    """
    try:
        tree = ElementTree.parse(logfile)
        root = tree.getroot()
        # wav attributes as a dict (ChainMap turns list of dicts into a single dict)
        pes = [
            child for child in root if child.tag == 'PROC_EVENT'
        ]  # and child.attrib["ID"] == "5"]
        wav_attributes = dict(ChainMap(*[wh.attrib for pe in pes for wh in pe]))
        utc_start = wav_attributes['SamplingStartTimeUTC']
        datefmt = '%m/%d/%Y %I:%M:%S %p' if '/' in utc_start else '%Y-%m-%dT%H:%M:%S'
        start_time_utc = dt.datetime.strptime(
            utc_start,
            datefmt
        )
        start_time_from_filename = dt_from_filename(logfile)
        return (
            round(
                (start_time_from_filename - start_time_utc).total_seconds() / (60 * 15)
            )
            / 4.0
        )
    except (KeyError, FileNotFoundError, ElementTree.ParseError, IndexError):
        # print(f'Unable to parse utc offset from logfile: {logfile}')
        return None


def gain_from_logfile(logfile: str) -> Union[str, None]:
    """
    Given the parameter `logfile` this method parses the `Gain` value used from it.  If
    it is unable to do so because of any of the following:

        - the file doesn't exist
        - the file isn't formatted as expected
        - the file doesn't contain the gain value

    it will simply return None.

    :param logfile: str - path to the logfile
    :return: Union[str, None] - the gain value as indicated by the logfile or None if
                                there is an issue collecting it.
    """
    try:
        tree = ElementTree.parse(logfile)
        root = tree.getroot()
        if root[0][0].attrib["STATE"] == "NEW":
            es = [child for child in root if child.tag == "EVENT"]
            return [
                et.attrib["Gain"]
                for e in es
                for et in e
                if et.tag == "AUDIO" and "Gain" in et.attrib.keys()
            ][0]
    except (KeyError, FileNotFoundError, ElementTree.ParseError, IndexError):
        return None


def dt_from_filename(file: str) -> dt.datetime:
    """
    Given the parameter `file` extract the datetime encoding and return it as a
    datetime.datetime object.  This method assumes that the `file` parameter follows the
    SoundTrap naming convention of `<sn>.<yymmddHHMMSS>.<ext>`; if it doesn't this
    method will likely either throw a ValueError or an IndexError.

    :param file: str - SoundTrap hydrophone file
    :return: dt.datetime - representing the datetime encoded in the parameter `file`
    """
    return dt.datetime.strptime(os.path.basename(file).split(".")[1], "%y%m%d%H%M%S")


def sn_from_filename(file: str) -> str:
    """
    Given the parameter `file` extract the serial number encoded and return it as a
    string object.  This method assumes that the `file` parameter follows the
    SoundTrap naming convention of `<sn>.<yymmddHHMMSS>.<ext>`; if it doesn't this
    method will likely either throw a ValueError or an IndexError.

    :param file: str - SoundTrap hydrophone file
    :return: str - representing the serial number encoded in the parameter `file`
    """
    return os.path.basename(file).split(".")[0]


def ext_from_filename(file: str) -> str:
    return ".".join(os.path.basename(file).split(".")[2:])


def extra_wav_details(filename: str) -> Dict[str, Any]:
    utc_offset = utc_offset_from_logfile(filename[:-3] + "log.xml")
    gain = gain_from_logfile(filename[:-3] + "log.xml")
    start = dt_from_filename(filename)
    return dict(start=start, utc_offset=utc_offset, gain=gain)


def _pull_device_specs(sn: str) -> dict:
    """
    This method uses the Ocean Instruments webapp to fetch calibration information for
    SoundTrap hydrophones using the serial number passed in through the parameter `sn`.
    Likely to raise an IndexError if an invalid serial number is passed in or the webapp
    doesn't respond as expected.

    :param sn: str - represents a SoundTrap serial number
    :return: Dict[str, str]
    """
    try:
        # search for the deviceId associated to SoundTrap serial number
        with urlopen(
            f"http://oceaninstruments.azurewebsites.net/api/Devices/Search/{sn}"
        ) as response:
            resp = json.loads(response.read())
        dev_d = resp[0]  # assume the first item in response is most recent
        did = dev_d["deviceId"]

        # use deviceId from the previous response to request the calibration data
        with urlopen(
            f"http://oceaninstruments.azurewebsites.net/api/Calibrations/Device/{did}"
        ) as response:
            resp = json.loads(response.read())
        cal_d = resp[0]  # assume the first item in response is most recent

        # pop device data out of the calibration data and merge with other device data
        dev_d = {**dev_d, **cal_d.pop("device")}

        # merge device data with remaining calibration data (for flat dictionary of all)
        c = {**dev_d, **cal_d}
    except (IndexError, URLError):
        c = DEVICE_PROPERTIES
    return c


def _parse_device_specs(specs: Dict[str, Any]):
    c = specs
    # format the values as strings for display
    c["calType"] = "0" if c.get("calType", "0") is None else str(c.get("calType", "0"))
    c["hpSerial"] = "" if c.get("hpSerial", "") is None else c.get("hpSerial", "")
    c["operator"] = "JA" if c.get("operator", "JA") is None else c.get("operator", "JA")
    c["sourceModel"] = "Center 327" if c.get("model", None) is None else ""
    c["sourceSerial"] = "130307390" if c.get("model", None) is None else ""
    c["sourceFreq"] = "250 Hz" if c.get("model", None) is None else ""
    c["sourceCoupler"] = "OIC1" if c.get("model", None) is None else ""
    c["refSerial"] = (
        "2015497"
        if c.get("refSerial", "2015497") is None
        else c.get("refSerial", "2015497")
    )
    c["refDevice"] = (
        "B&K 2236"
        if c.get("refDevice", "B&K 2236") is None
        else c.get("refDevice", "B&K 2236")
    )
    c["dateCreated"] = c["dateCreated"].split("T")[0] if c != "" else ""
    c["refLevel"] = f"{c['refLevel']} dB re. 1 \u00b5Pa" if c['refLevel'] != '' else ''
    unit = "\u00b5Pa" if c["calType"] == "0" else ("V" if c["calType"] == "1" else "?")
    c["tone"] = f"{c['tone']}  dB re. 1 {unit}" if c['tone'] != '' else ''
    c["highFreq"] = f"{c['highFreq']} dB" if c['highFreq'] != '' else ''
    c["lowFreq"] = f"{c['lowFreq']} dB" if c['highFreq'] != '' else ''
    return c


def device_specs_from_sn(sn: str):
    specs = _pull_device_specs(sn)
    return _parse_device_specs(specs)


def format_specs(specs:Dict[str, Any]):
    # define the layout
    c = specs
    try:
        cid = int(c["calType"]) if 0 <= int(c["calType"]) <= 2 else 2
    except ValueError:
        cid = 2
    layout = [
        ("Test", [("Date", c["dateCreated"]), ("Operator", c["operator"])]),
        (
            "Device",
            [
                ("Model", c["modelName"]),
                ("Serial No", c["serialNo"]),
                ("Hardware Serial No", c["hardwareSerial"]),
                ("Hp Serial No", c["hpSerial"]),
            ],
        ),
        (
            "Source",
            [
                ("Model", c["sourceModel"]),
                ("Serial", c["sourceSerial"]),
                ("Frequency", c["sourceFreq"]),
                ("Coupler", c["sourceCoupler"]),
                ("Level", c["refLevel"]),
            ],
        ),
        ("Reference", [("Model", c["refDevice"]), ("Serial", c["refSerial"])]),
        ("Calibration Tone", [("RTI Level @ 1kHz", c["tone"])]),
    ]
    variable_section = [
        (
            "End-to-End Calibration",
            [("High Gain", c["highFreq"]), ("Low Gain", c["lowFreq"])],
        ),
        ("Sensitivity", [("", c["highFreq"])]),
        (
            "Unknown Calibration Type",
            [("High Gain", c["highFreq"]), ("Low Gain", c["lowFreq"])],
        ),
    ]
    layout = OrderedDict(layout[:-1] + [variable_section[cid]] + layout[-1:])
    return layout
