from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.default import NoInput, _drb
from rateslib.scheduling import Adjuster, Convention, Frequency, RollDay
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.frequency import _get_frequency_none

if TYPE_CHECKING:
    from rateslib.typing import CalInput, bool_, datetime_, int_, str_


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
    convention : str
        The day count convention of the calculation period accrual. See notes.
    termination : datetime, optional
        The adjusted termination date of the leg. Required only if ``convention`` is
        one of the following values:

        - `"30E360ISDA"` (since end Feb is adjusted to 30 unless it aligns with
          ``termination`` of a leg)
        - `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"`, (if the period is
          a stub the ``termination`` of the leg is used to assess front or back stubs and
          adjust the calculation accordingly)

    frequency_months : int, optional
        The number of months according to the frequency of the period. Required only
        with specific values for ``convention``.
    stub : bool, optional
        Required for `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"`.
        Non-stub periods will
        return a fraction equal to the frequency, e.g. 0.25 for quarterly.
    roll : str, int, optional
        Used by `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"` to project
        regular periods when calculating stubs.
    calendar: str, Calendar, optional
        Required for `"BUS252"` to count business days in period.

    Returns
    --------
    float

    Notes
    -----
    Permitted values for the convention are:

    - `"1"`: Returns 1 for any period.
    - `"1+"`: Returns the number of months between dates divided by 12.
    - `"Act365F"`: Returns actual number of days divided by a fixed 365 denominator.
    - `"Act365F+"`: Returns the number of years and the actual number of days in the fractional year
      divided by a fixed 365 denominator.
    - `"Act360"`: Returns actual number of days divided by a fixed 360 denominator.
    - `"30E360"`, `"EuroBondBasis"`: Months are treated as having 30 days and start
      and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, end day).

    - `"30360"`, `"360360"`, `"BondBasis"`: Months are treated as having 30 days
      and start and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, end day) if start day >= 30.

    - `"30U360"`: Months are treated as having 30 days and start and end dates are converted
      under the following rules in order:

      * If the ``roll`` is EoM and ``start`` is end-Feb then:

         - start day is 30.
         - end day is 30 ``end`` is also end-Feb.

      * If start day is 30 or 31 then it is converted to 30.
      * End day is converted to 30 if it is 31 and start day is 30.

    - `"30360ISDA"`: Months are treated as having 30 days and start and end dates are
      converted under the rule:

      * start day is converted to 30 if it is a month end.
      * end day is converted to 30 if it is a month end.
      * end day is not converted if it coincides with the leg termination and is
        in February.

    - `"ActAct"`, `"ActActISDA"`: Calendar days between start and end are divided
      by 365 or 366 dependent upon whether they fall within a leap year or not.
    - `"ActActICMA"`, `"ActActISMA"`, `"ActActBond"`, `"ActActICMA_stub365f"`: Returns a fraction
      relevant to the frequency of the schedule if a regular period. If a stub then projects
      a regular period and returns a fraction of that period.
    - `"Bus252"`: Business days between start and end divided by 252. If business days, `start` is
      included whilst `end` is excluded.

    Further information can be found in the
    :download:`2006 ISDA definitions <https://www.rbccm.com/assets/rbccm/docs/legal/doddfrank/Documents/ISDALibrary/2006%20ISDA%20Definitions.pdf>` and
    :download:`2006 ISDA 30360 example <_static/30360isda_2006_example.xls>`.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import dcf

    .. ipython:: python

       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act360")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act365f")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, False)
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, True)

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

    return convention_.dcf(
        start=start,
        end=end,
        termination=_drb(None, termination),
        frequency=frequency_,
        stub=_drb(None, stub),
        calendar=get_calendar(calendar),
        adjuster=_get_adjuster(_drb(Adjuster.Actual(), adjuster)),
    )


CONVENTIONS_MAP: dict[str, Convention] = {
    "ACT365F": Convention.Act365F,
    "ACT360": Convention.Act360,
    ###
    "30360": Convention.Thirty360,
    "360360": Convention.Thirty360,
    "BONDBASIS": Convention.Thirty360,
    "30E360": Convention.ThirtyE360,
    "EUROBONDBASIS": Convention.ThirtyE360,
    "30E360ISDA": Convention.ThirtyE360ISDA,
    "30U360": Convention.ThirtyU360,
    ###
    "ACT365F+": Convention.YearsAct365F,
    "ACT360+": Convention.YearsAct360,
    "1+": Convention.YearsMonths,
    ###
    "1": Convention.One,
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
}


def _get_convention(convention: Convention | str) -> Convention:
    """Convert a user str input into a Convention enum."""
    if isinstance(convention, Convention):
        return convention
    else:
        try:
            return CONVENTIONS_MAP[convention.upper()]
        except KeyError:
            raise ValueError(f"`convention`: {convention}, is not valid.")
            # raise ValueError(
            #     "`convention` must be in {'Act365f', '1', '1+', 'Act360', "
            #     "'30360' '360360', 'BondBasis', '30U360', '30E360', 'EuroBondBasis', "
            #     "'30E360ISDA', 'ActAct', 'ActActISDA', 'ActActICMA', "
            #     "'ActActISMA', 'ActActBond'}",
            # )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
