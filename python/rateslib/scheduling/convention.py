from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.default import NoInput, _drb
from rateslib.rs import Adjuster, Convention, Frequency, RollDay
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import CalInput, bool_, datetime, datetime_, str_

_CONVENTIONS_MAP: dict[str, Convention] = {
    "ACT365F": Convention.Act365F,
    "ACT360": Convention.Act360,
    "THIRTY360": Convention.Thirty360,
    "THIRTYE360": Convention.ThirtyE360,
    "THIRTYU360": Convention.ThirtyU360,
    "THIRTYE360ISDA": Convention.ThirtyE360ISDA,
    "YEARSACT365F": Convention.YearsAct365F,
    "YEARSACT360": Convention.YearsAct360,
    "YEARSMONTHS": Convention.YearsMonths,
    "ONE": Convention.One,
    "ACTACTISDA": Convention.ActActISDA,
    "ACTACTICMA": Convention.ActActICMA,
    "BUS252": Convention.Bus252,
}


def _get_convention(convention: str | Convention) -> Convention:
    if isinstance(convention, Convention):
        return convention
    return _CONVENTIONS_MAP[convention.upper()]


def dcf(
    start: datetime,
    end: datetime,
    convention: str | Convention,
    termination: datetime_ = NoInput(0),
    frequency: Frequency | str_ = NoInput(0),
    roll: RollDay | int | str_ = NoInput(0),
    stub: bool_ = NoInput(0),
    calendar: CalInput = NoInput(0),
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
    convention_: Convention = _get_convention(convention)
    if isinstance(frequency, NoInput):
        frequency_: Frequency | None = None
    else:
        frequency_ = _get_frequency(frequency, roll, calendar)

    if isinstance(adjuster, NoInput):
        adjuster_: Adjuster | None = None
    else:
        adjuster_ = _get_adjuster(adjuster)
    try:
        return convention_.dcf(
            start=start,
            end=end,
            termination=_drb(None, termination),
            frequency=frequency_,
            stub=_drb(None, stub),
            calendar=_drb(None, calendar),
            adjuster=adjuster_,
        )
    except KeyError:
        raise ValueError(
            "`convention` must be in {'Act365f', '1', '1+', 'Act360', "
            "'30360' '360360', 'BondBasis', '30U360', '30E360', 'EuroBondBasis', "
            "'30E360ISDA', 'ActAct', 'ActActISDA', 'ActActICMA', "
            "'ActActISMA', 'ActActBond'}",
        )


__all__ = ["Convention"]
