from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.rs import Imm

if TYPE_CHECKING:
    from rateslib.typing import int_, str_


_Imm: dict[str, Imm] = {
    "imm": Imm.Wed3_HMUZ,
    "serial_imm": Imm.Wed3,
    "credit_imm": Imm.Day20_HMUZ,
    "credit_imm_hu": Imm.Day20_HU,
    "credit_imm_mz": Imm.Day20_MZ,
    "wed3_hmuz": Imm.Wed3_HMUZ,
    "wed3": Imm.Wed3,
    "day20_hmuz": Imm.Day20_HMUZ,
    "day20": Imm.Day20,
    "day20_mz": Imm.Day20_MZ,
    "day20_hu": Imm.Day20_HU,
    "fri2_hmuz": Imm.Fri2_HMUZ,
    "fri2": Imm.Fri2,
    "wed1_post9": Imm.Wed1_Post9,
    "wed1_post9_hmuz": Imm.Wed1_Post9_HMUZ,
    "eom": Imm.Eom,
    "leap": Imm.Leap,
}


def next_imm(start: datetime, definition: str | Imm = Imm.Wed3_HMUZ) -> datetime:
    """Return the next IMM date *after* the given start date.

    Parameters
    ----------
    start : datetime
        The date from which to determine the next IMM.
    definition : Imm, str in {"imm", "serial_imm", "credit_imm", "credit_imm_HU", "credit_imm_MZ"}
        The IMM definition to return the date for. See notes.

    Returns
    -------
    datetime

    Notes
    -----
    The ``definition`` is typically input as an :class:`~rateslib.scheduling.Imm` enum.
    For simpler, and legacy cases, string input options are also available:

    - 'imm': 3rd Wednesday in any month of March, June, September or December.
    - 'serial_imm': 3rd Wednesday in any month of the year.
    - 'credit_imm': 20th of the month in March, June, September or December.
    - 'credit_imm_HU': 20th of the month in March or September, facilitating CDSs that
       rolls on a 6-month basis.
    - 'credit_imm_MZ': 20th of the month in June and December.
    - String representations of all of the underlying enum variants.
    """
    if isinstance(definition, str):
        imm_: Imm = _Imm[definition.lower()]
    else:
        imm_ = definition
    return imm_.next(start)


MONTHS = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def get_imm(
    month: int_ = NoInput(0),
    year: int_ = NoInput(0),
    code: str_ = NoInput(0),
    definition: str | Imm = Imm.Wed3,
) -> datetime:
    """
    Return an IMM date for a specified month.

    Parameters
    ----------
    month: int
        The month of the year in which the IMM date falls.
    year: int
        The year in which the IMM date falls.
    code: str
        Identifier in the form of a one digit month code and 21st century year, e.g. "U29".
        If code is given ``month`` and ``year`` are unused.
    definition: Imm, str
        The IMM definition to return the date for. See :meth:`~rateslib.scheduling.next_imm`.

    Returns
    -------
    datetime
    """
    if isinstance(code, str):
        year = int(code[1:]) + 2000
        month = MONTHS[code[0].upper()]
    elif isinstance(month, NoInput) or isinstance(year, NoInput):
        raise ValueError("`month` and `year` must each be valid integers if `code`not given.")

    if isinstance(definition, str):
        imm_: Imm = _Imm[definition.lower()]
    else:
        imm_ = definition
    return imm_.get(year, month)
