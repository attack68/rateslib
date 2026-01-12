# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput
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
    definition : Imm, str
        The IMM definition to return the date for. This is entered as either an
        :class:`~rateslib.scheduling.Imm` enum, or that enum variant name as sting, e.g. *"Wed3"*.

    Returns
    -------
    datetime

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib import next_imm, Imm, dt

    Get the next quarterly SOFR or ESTR futures date, defined by CME, EUREX, or ICE:

    .. ipython:: python

       next_imm(dt(2000, 1, 1), Imm.Wed3_HMUZ)

    Get the next serial futures contract for a NZD bank bill defined by ASX:

    .. ipython:: python

       next_imm(dt(2000, 1, 1), "Wed1_Post9")

    """
    if isinstance(definition, str):
        d_ = definition.lower()
        if d_ in ["imm", "serial_imm", "credit_imm", "credit_imm_hu", "credit_imm_mz"]:
            warnings.warn(
                f"The given string entry '{d_}' is deprecated and will be removed in "
                f"future releases. Please change the equivalent version in {{'Wed3', 'Wed3_HMUZ', "
                f"'Day20', 'Day20_HMUZ', 'Day20_HU', 'Day20_MZ'}}",
                DeprecationWarning,
            )
        imm_: Imm = _Imm[d_]
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
        The IMM definition to return the date for. This is entered as either an
        :class:`~rateslib.scheduling.Imm` enum, or that enum variant name as sting, e.g. *"Wed3"*.

    Returns
    -------
    datetime

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib import get_imm, Imm, dt

    Get the quarterly SOFR or ESTR futures date, defined by CME, EUREX, or ICE:

    .. ipython:: python

       get_imm(3, 2022, definition=Imm.Wed3_HMUZ)
       get_imm(code="H22", definition="Wed3")

    Get a serial futures contract for a NZD bank bill defined by ASX:

    .. ipython:: python

       get_imm(1, 2023, definition="Wed1_Post9")
    """
    if isinstance(code, str):
        year = int(code[1:]) + 2000
        month = MONTHS[code[0].upper()]
    elif isinstance(month, NoInput) or isinstance(year, NoInput):
        raise ValueError("`month` and `year` must each be valid integers if `code`not given.")

    if isinstance(definition, str):
        d_ = definition.lower()
        if d_ in ["imm", "serial_imm", "credit_imm", "credit_imm_hu", "credit_imm_mz"]:
            warnings.warn(
                f"The given string entry '{d_}' is deprecated and will be removed in "
                f"future releases. Please change the equivalent version in {{'Wed3', 'Wed3_HMUZ', "
                f"'Day20', 'Day20_HMUZ', 'Day20_HU', 'Day20_MZ'}}",
                DeprecationWarning,
            )
        imm_: Imm = _Imm[d_]
    else:
        imm_ = definition
    return imm_.get(year, month)
