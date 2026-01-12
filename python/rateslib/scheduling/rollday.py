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

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.rs import Adjuster, Imm, RollDay

if TYPE_CHECKING:
    from rateslib.typing import CalTypes, int_


def _get_rollday(roll: RollDay | str | int_) -> RollDay | None:
    """Convert a user str or int into a RollDay enum object."""
    if isinstance(roll, RollDay):
        return roll
    elif isinstance(roll, str):
        return {
            "EOM": RollDay.Day(31),
            "SOM": RollDay.Day(1),
            "IMM": RollDay.IMM(),
        }[roll.upper()]
    elif isinstance(roll, int):
        return RollDay.Day(roll)
    return None


def _is_eom_cal(date: datetime, cal: CalTypes) -> bool:
    """Test whether a given date is end of month under a specific calendar"""
    eom_unadjusted = Imm.Eom.get(date.year, date.month)
    eom = Adjuster.Previous().adjust(eom_unadjusted, cal)
    return date == eom
    # end_day = calendar_mod.monthrange(date.year, date.month)[1]
    # eom = datetime(date.year, date.month, end_day)
    # adj_eom = _adjust_date(eom, "P", cal)
    # return date == adj_eom
