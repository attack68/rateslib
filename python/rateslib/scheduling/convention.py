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

from typing import TYPE_CHECKING

from rateslib.rs import Convention

if TYPE_CHECKING:
    pass

_CONVENTIONS_MAP: dict[str, Convention] = {
    "ACT365F": Convention.Act365F,
    "ACT365": Convention.Act365F,
    "ACT360": Convention.Act360,
    "ACT365_25": Convention.Act365_25,
    "ACT364": Convention.Act364,
    ###
    "30360": Convention.Thirty360,
    "THIRTY360": Convention.Thirty360,
    "360360": Convention.Thirty360,
    "BONDBASIS": Convention.Thirty360,
    "30E360": Convention.ThirtyE360,
    "THIRTYE360": Convention.ThirtyE360,
    "EUROBONDBASIS": Convention.ThirtyE360,
    "30E360ISDA": Convention.ThirtyE360ISDA,
    "THIRTYE360ISDA": Convention.ThirtyE360ISDA,
    "30U360": Convention.ThirtyU360,
    "THIRTYU360": Convention.ThirtyU360,
    ###
    "ACT365F+": Convention.YearsAct365F,
    "YEARSACT365F": Convention.YearsAct365F,
    "ACT360+": Convention.YearsAct360,
    "YEARSACT360": Convention.YearsAct360,
    "1+": Convention.YearsMonths,
    "YEARSMONTHS": Convention.YearsMonths,
    ###
    "1": Convention.One,
    "ONE": Convention.One,
    ###
    "ACTACT": Convention.ActActISDA,
    "ACTACTISDA": Convention.ActActISDA,
    "ACTACTICMA": Convention.ActActICMA,
    "ACTACTISMA": Convention.ActActICMA,
    "ACTACTBOND": Convention.ActActICMA,
    ###
    "BUS252": Convention.Bus252,
    ###
    "ACTACTICMA_STUB365F": Convention.ActActICMAStubAct365F,
    "ACTACTICMASTUBACT365F": Convention.ActActICMAStubAct365F,
}


def _get_convention(convention: Convention | str) -> Convention:
    """Convert a user str input into a Convention enum."""
    if isinstance(convention, Convention):
        return convention
    else:
        try:
            return _CONVENTIONS_MAP[convention.upper()]
        except KeyError:
            raise ValueError(f"`convention`: {convention}, is not valid.")


__all__ = ["Convention"]
