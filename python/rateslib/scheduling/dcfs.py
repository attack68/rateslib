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
from functools import partial
from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput, _drb
from rateslib.scheduling import Adjuster, Convention, Frequency, RollDay
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency_none

if TYPE_CHECKING:
    from rateslib.typing import Any, CalInput, Callable, bool_, datetime_, int_, str_


def dcf(
    start: datetime,
    end: datetime,
    convention: Convention | str,
    termination: datetime_ = NoInput(0),  # required for 30E360ISDA and ActActICMA
    frequency: Frequency | str_ = NoInput(0),  # req. ActActICMA = ActActISMA = ActActBond
    stub: bool_ = NoInput(0),  # required for ActActICMA = ActActISMA = ActActBond
    roll: RollDay | str | int_ = NoInput(0),  # required also for ActACtICMA = ...
    calendar: CalInput = NoInput(0),  # required for ActACtICMA = ActActISMA = ActActBond
    adjuster: Adjuster | str_ = NoInput(0),
) -> float:
    """
    Calculate the day count fraction of a period.

    Parameters
    ----------
    start : datetime
        The adjusted start date of the calculation period.
    end : datetime
        The adjusted end date of the calculation period.
    convention : Convention, str
        The day count convention of the calculation period accrual. See notes.
    termination : datetime, optional
        The adjusted termination date of the leg. Required only for some ``convention``.
    frequency : Frequency, str, optional
        The frequency of the period. Required only for some ``convention``.
    stub : bool, optional
        Indicates whether the period is a stub or not. Required only for some ``convention``.
    roll : str, int, optional
        Used only if ``frequency`` is given in string form. Required only for some ``convention``.
    calendar: str, Calendar, optional
        Used only of ``frequency`` is given in string form. Required only for some ``convention``.
    adjuster: Adjuster, str, optional
        The :class:`~rateslib.scheduling.Adjuster` used to convert unadjusted dates to
        adjusted accrual dates on the period. Required only for some ``convention``.

    Returns
    --------
    float

    Notes
    -----
    See :class:`~rateslib.scheduling.Convention` for permissible values and for argument
    related specifics.

    Further information can be found in the
    :download:`2006 ISDA definitions <https://www.rbccm.com/assets/rbccm/docs/legal/doddfrank/Documents/ISDALibrary/2006%20ISDA%20Definitions.pdf>` and
    :download:`2006 ISDA 30360 example <_static/30360isda_2006_example.xls>`, and also in the lower
    level Rust documentation at :rust:`rateslib-rs: Scheduling <scheduling>`.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import dcf

    .. ipython:: python

       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act360")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act365f")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), "Q", False)
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), "Q", True)

    """  # noqa: E501
    convention_ = _get_convention(convention)
    if isinstance(adjuster, NoInput):
        adjuster = Adjuster.Actual()
    frequency_: Frequency | None = _get_frequency_none(frequency, roll, calendar)

    if isinstance(frequency_, Frequency.Zero) and convention_ in [
        Convention.ActActICMA,
        Convention.ActActICMAStubAct365F,
    ]:
        warnings.warn(
            "`frequency` cannot be 'Zero' variant in combination with 'ActActICMA' type"
            "conventions. Internally this will be converted to 'Frequency.Months(12, ...)'",
            UserWarning,
        )

    # delegate simple calculations to Python only for performance gains, otherwise use Rust.
    if convention_ in PERFORMANCE:
        try:
            return PERFORMANCE[convention_](start, end, frequency=frequency_, stub=stub)
        except NotImplementedError:
            pass

    return convention_.dcf(
        start=start,
        end=end,
        termination=_drb(None, termination),
        frequency=frequency_,
        stub=_drb(None, stub),
        calendar=get_calendar(calendar),
        adjuster=_get_adjuster(_drb(Adjuster.Actual(), adjuster)),
    )


def _dcf_numeric(start: datetime, end: datetime, denominator: float, **kwargs: Any) -> float:
    """Calculate the day count fraction of a period using the fixed denominator rule."""
    return (end - start).days / denominator


def _dcf_actacticma_nonstub(
    start: datetime, end: datetime, frequency: Frequency, stub: bool, **kwargs: Any
) -> float:
    """Calculate just the regular frequency part of the dcf for ActActICMA."""
    if not stub:
        return 1.0 / frequency.periods_per_annum()
    else:
        raise NotImplementedError("`stub` must be `False` for `ActActICMA` performance short cut.")


PERFORMANCE: dict[Convention, Callable[..., float]] = {
    Convention.Act365F: partial(_dcf_numeric, denominator=365.0),
    Convention.Act360: partial(_dcf_numeric, denominator=360.0),
    Convention.ActActICMA: _dcf_actacticma_nonstub,
    Convention.ActActICMAStubAct365F: _dcf_actacticma_nonstub,
}


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
