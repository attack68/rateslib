from typing import Optional, Union, Any
import calendar as calendar_mod
from itertools import product
from collections.abc import Iterator
from datetime import datetime, timedelta
from pandas import DataFrame
from pandas.tseries.offsets import CustomBusinessDay

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.calendars import (
    get_calendar,
    _is_holiday,
    _is_eom_cal,
    add_tenor,
    _add_months,
    _adjust_date,
    _is_eom,
    _is_imm,
    _is_som,
    _get_roll,
)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Schedule:
    """
    Generate a schedule of dates according to a regular pattern and calendar inference.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}, optional
        The frequency of the schedule where the options are: M(onthly), B(i-monthly),
        T(hirdly), Q(uarterly), S(emi-annually), A(nnually), Z(ero-coupon).
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}, optional
        The stub type to enact on the swap. Can provide two types, for
        example "SHORTFRONTLONGBACK".
    front_stub : datetime, optional
        An adjusted or unadjusted date for the first stub period.
    back_stub : datetime, optional
        An adjusted or unadjusted date for the back stub period.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. Inferred if not given.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for inference. Set by
        default. Not required if ``roll`` is specified.
    modifier : str, optional
        The modification rule, in {"NONE", "F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If string will call
        :meth:`~rateslib.calendars.get_calendar`.
    payment_lag: int, optional
        The number of business days to lag payments by.
    eval_date: datetime, optional
        Only required if ``effective`` is given as a string tenor, to provide a point of reference.
    eval_mode: str in {"swaps_align", "swaptions_align"}
        The method for determining the ``effective`` and ``termination`` dates if both are provided
        as string tenors. See notes.

    Attributes
    ----------
    ueffective : datetime
    effective : datetime
    utermination : datetime
    termination : datetime
    frequency : str
    stub : str
    front_stub : datetime
    back_stub : datetime
    roll : int or str
    eom : bool
    modifier : str
    calendar : calendar
    payment_lag : int
    uschedule : list[datetime]
    aschedule : list[datetime]
    pschedule : list[datetime]
    stubs : list[bool]
    eval_date : datetime
    eval_mode : str

    Notes
    -----
    **Zero coupon schedules**

    If ``frequency`` is *Z* then stub arguments are ignored.

    **Inferred termination date and roll from tenor - The 1Y1Y problem**

    When generating schedules implied from tenor ``effective`` and ``termination`` dates there
    exist two methods for doing this. **Both** have practical reasons to exist. This results in
    the ``eval_mode`` argument which allows either *"swaps_align"* or *"swaptions_align"*. So,
    what is the difference and the purpose?

    **Swaps Align**

    When a EUR swap dealer trades a 1Y1Y swap he will hedge it in the interbank market with a 1Y and
    a 2Y swap. 1Y and 2Y swaps have roll days that are generated from the same evaluation date.
    For a perfect hedge the 1Y1Y swap should also have the same roll day and its periods should
    align with the second half of the 2Y swap. To achieve this, the ``effective`` date is
    calculated **unadjusted** and the ``termination`` date is derived from that unadjusted date.
    Then under *rateslib* inferral rules this will produce the correct schedule.

    For example, today is Tue 15th Aug '23 and spot is Thu 17th Aug '23:

    - A 1Y trade has effective, termination and roll of: Tue 17th Aug '23, Mon 19th Aug '24, 17.
    - A 2Y trade has effective, termination and roll of: Tue 17th Aug '23, Mon 18th Aug '25, 17.
    - A 1Y1Y trade has effective, termination and roll of: Mon 19th Aug '24, Mon 18th Aug '25, 17.

    .. ipython:: python

       sch = Schedule(
           effective="1Y",
           termination="1Y",
           frequency="S",
           calendar="tgt",
           eval_date=dt(2023, 8, 17),
           eval_mode="swaps_align",
       )
       sch

    **Swaptions Align**

    When a swaptions dealer trades a 1Y1Y swaption, that trade will settle against the 1Y swap
    evaluated as of the expiry date (in 1Y) against the swap ISDA fixing.
    The delta exposure the swaption trader experiences is best hedged with a swap matching those
    dates. This means that the effective date of the swap should be derived from an **adjusted**
    date.

    For example, today is Tue 15th Aug '23:

    - A 1Y expiring swaption has an expiry on Thu 15th Aug '24.
    - At expiry a spot starting 1Y swap has effective, termination, and roll of:
      Mon 19th Aug '24, Tue 19th Aug '25, 19.

    .. ipython:: python

       sch = Schedule(
           effective="1Y",
           termination="1Y",
           frequency="S",
           calendar="tgt",
           eval_date=dt(2023, 8, 17),
           eval_mode="swaptions_align",
       )
       sch

    .. note::
       To avoid these, it is recommended to provide ``effective``, ``termination``, as
       **unadjusted dates** (and also ``front_stub`` and ``back_stub``) since this eliminates the
       combinatorial aspect of date inference. Also providing ``roll`` is more
       explicit.

    **Inferred stub dates from inputs**

    The following input arguments are provided; ``stub``, ``front_stub`` and
    ``back_stub``. These are optional and in the case one or more are *NoInput*, then
    the code will attempt to infer stub scheduling.

    .. list-table:: Inference when ``stub`` is *None* in combination with stub dates
       :widths: 20 20 20 20 20
       :header-rows: 2

       * - ``front_stub``:
         - datetime
         - datetime
         - *NoInput*
         - *NoInput*
       * - ``back_stub``:
         - datetime
         - *NoInput*
         - datetime
         - *NoInput*
       * - ``stub`` defaults to:
         - *"FRONTBACK"*
         - *"FRONT"*
         - *"BACK"*
         - ``defaults.stub`` (*"SHORTFRONT"*)
       * - Method called:
         - :meth:`_check_regular_swap`
         - :meth:`_check_regular_swap`
         - :meth:`_check_regular_swap`
         - :meth:`_infer_stub_date`

    In the case that :meth:`_check_regular_swap` is called this will attempt to ensure
    that the given stub dates align either with each other, or with the associated
    ``effective`` or ``termination`` dates. If they do not align then an error is
    raised.

    In the case all are *NoInput* and :meth:`_infer_stub_date` is called this will first
    check for a regular swap to ensure a stub is required and if so will generate the
    appropriate stub at the front or back as necessary.


    .. list-table:: Inference when ``stub`` is given in combination with stub dates
       :widths: 20 20 20 20 20
       :header-rows: 2

       * - ``front_stub``:
         - datetime
         - datetime
         - *NoInput*
         - *NoInput*
       * - ``back_stub``:
         - datetime
         - *NoInput*
         - datetime
         - *NoInput*
       * - ``stub`` is dual sided
         - :meth:`_check_regular_swap`
         - :meth:`_infer_stub_date`
         - :meth:`_infer_stub_date`
         - ValueError
       * - ``stub`` is front sided
         - ValueError
         - :meth:`_check_regular_swap`
         - ValueError
         - :meth:`_infer_stub_date`
       * - ``stub`` is back sided
         - ValueError
         - ValueError
         - :meth:`_check_regular_swap`
         - :meth:`_infer_stub_date`

    **Handling stubs and rolls**

    If ``front_stub`` and ``back_stub`` are given, the ``stub`` is not used. There is
    **no validation** check to ensure that the given dated stubs conform to the stub
    type, for example if the front stub conforms to a long front or a short front.

    **Dates only object**

    A ``Schedule`` is a dates only object, meaning the attributes necessary for
    cashflow calculation and generation, such as day count convention and notional
    amounts are not attributed to this object. Those will be handled by the appropriate
    ``Leg`` object.

    """

    def __init__(
        self,
        effective: Union[datetime, str],
        termination: Union[datetime, str],
        frequency: str,
        stub: Union[str, NoInput] = NoInput(0),
        front_stub: Union[datetime, NoInput] = NoInput(0),
        back_stub: Union[datetime, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        eval_date: Union[datetime, NoInput] = NoInput(0),
        eval_mode: Union[str, NoInput] = NoInput(0),
    ):
        # Arg validation
        eom = defaults.eom if eom is NoInput.blank else eom
        self.eom = eom

        self.eval_date = eval_date
        self.eval_mode = defaults.eval_mode if eval_mode is NoInput.blank else eval_mode.lower()

        if modifier is NoInput.blank:  # then get default
            modifier_: str = defaults.modifier
        else:
            modifier_ = modifier
        self.modifier = modifier_

        if payment_lag is NoInput.blank:
            payment_lag_: int = defaults.payment_lag
        else:
            payment_lag_ = payment_lag
        self.payment_lag = payment_lag_

        self.calendar = get_calendar(calendar)

        frequency = frequency.upper()
        if frequency not in ["M", "B", "Q", "T", "S", "A", "Z"]:
            raise ValueError("`frequency` must be in {M, B, Q, T, S, A, Z}.")
        self.frequency = frequency

        if isinstance(effective, str):
            if self.eval_date is NoInput.blank:
                raise ValueError(
                    "For `effective` given as string tenor, must also supply a `base_eval` date."
                )
            if self.eval_mode == "swaps_align":
                # effective date is calculated as unadjusted
                effective_: datetime = add_tenor(
                    self.eval_date, effective, "NONE", NoInput(0), roll
                )
            elif self.eval_mode == "swaptions_align":
                effective_ = add_tenor(
                    self.eval_date, effective, self.modifier, self.calendar, roll
                )
        else:
            effective_ = effective
        self.effective: datetime = effective_

        if isinstance(termination, str):
            # if termination is string the end date is calculated as unadjusted
            if self.eom and roll is NoInput.blank and _is_eom_cal(self.effective, self.calendar):
                roll_ = 31
            else:
                roll_ = roll
            termination_: datetime = add_tenor(
                self.effective, termination, "NONE", NoInput(0), roll_
            )
        else:
            termination_ = termination
        self.termination: datetime = termination_

        if self.termination <= self.effective:
            raise ValueError("`termination` must be after `effective`.")

        if frequency == "Z":
            self.ueffective = NoInput(0)
            self.utermination = NoInput(0)
            self.stub = NoInput(0)
            self.front_stub = NoInput(0)
            self.back_stub = NoInput(0)
            self.roll = NoInput(0)
            self.uschedule = [self.effective, self.termination]
            self._attribute_schedules()
            return None

        if stub is NoInput.blank:
            # if specific stub dates are given we cannot know if these are long or short
            if front_stub is NoInput.blank:
                stub = defaults.stub if back_stub is NoInput.blank else "BACK"
            else:
                stub = "FRONT" if back_stub is NoInput.blank else "FRONTBACK"
        else:
            stub = stub.upper()
        self.stub = stub

        if "FRONT" in stub and "BACK" in stub:
            self._dual_sided_stub_parsing(front_stub, back_stub, roll)
        elif "FRONT" in stub:
            self._front_sided_stub_parsing(front_stub, back_stub, roll)
        elif "BACK" in stub:
            self._back_sided_stub_parsing(front_stub, back_stub, roll)
        else:
            raise ValueError(
                "`stub` should be combinations of {'SHORT', 'LONG'} with {'FRONT', 'BACK'}."
            )

        self.uschedule = list(
            _generate_irregular_schedule_unadjusted(
                self.ueffective,
                self.utermination,
                self.frequency,
                self.roll,
                self.front_stub,
                self.back_stub,
            )
        )
        self._attribute_schedules()
        return None

    def _dual_sided_stub_parsing(self, front_stub, back_stub, roll):
        """This is called when the provided `stub` argument implies dual sided stubs."""
        if front_stub is NoInput.blank and back_stub is NoInput.blank:
            raise ValueError(
                "Must supply at least one stub date with dual sided stub type.\n"
                "Require `front_stub` or `back_stub` or both."
            )
        elif front_stub is NoInput.blank:
            valid, parsed_args = _infer_stub_date(
                self.effective,
                self.termination,
                self.frequency,
                self.stub,
                front_stub,
                back_stub,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = parsed_args["ueffective"]
                self.utermination = parsed_args["utermination"]
                self.front_stub = parsed_args["front_stub"]
                self.back_stub = parsed_args["back_stub"]
                self.roll = parsed_args["roll"]
        elif back_stub is NoInput.blank:
            valid, parsed_args = _infer_stub_date(
                self.effective,
                self.termination,
                self.frequency,
                self.stub,
                front_stub,
                back_stub,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = parsed_args["ueffective"]
                self.utermination = parsed_args["utermination"]
                self.front_stub = parsed_args["front_stub"]
                self.back_stub = parsed_args["back_stub"]
                self.roll = parsed_args["roll"]
        else:
            # check regular swap and populate attributes
            valid, parsed_args = _check_regular_swap(
                front_stub,
                back_stub,
                self.frequency,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = self.effective
                self.utermination = self.termination
                self.front_stub = parsed_args["ueffective"]
                self.back_stub = parsed_args["utermination"]
                self.roll = parsed_args["roll"]

    def _front_sided_stub_parsing(self, front_stub, back_stub, roll):
        if back_stub is not NoInput.blank:
            raise ValueError("`stub` is only front sided but `back_stub` given.")
        if front_stub is NoInput.blank:
            valid, parsed_args = _infer_stub_date(
                self.effective,
                self.termination,
                self.frequency,
                self.stub,
                front_stub,
                back_stub,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = parsed_args["ueffective"]
                self.utermination = parsed_args["utermination"]
                self.front_stub = parsed_args["front_stub"]
                self.back_stub = parsed_args["back_stub"]
                self.roll = parsed_args["roll"]
        else:
            # check regular swap and populate attibutes
            valid, parsed_args = _check_regular_swap(
                front_stub,
                self.termination,
                self.frequency,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                # stub inference is not required, no stubs are necessary
                self.ueffective = self.effective
                self.utermination = parsed_args["utermination"]
                self.front_stub = parsed_args["ueffective"]
                self.back_stub = NoInput(0)
                self.roll = parsed_args["roll"]

    def _back_sided_stub_parsing(self, front_stub, back_stub, roll):
        if front_stub is not NoInput.blank:
            raise ValueError("`stub` is only back sided but `front_stub` given.")
        if back_stub is NoInput.blank:
            valid, parsed_args = _infer_stub_date(
                self.effective,
                self.termination,
                self.frequency,
                self.stub,
                front_stub,
                back_stub,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = parsed_args["ueffective"]
                self.utermination = parsed_args["utermination"]
                self.front_stub = parsed_args["front_stub"]
                self.back_stub = parsed_args["back_stub"]
                self.roll = parsed_args["roll"]
        else:
            # check regular swap and populate attributes
            valid, parsed_args = _check_regular_swap(
                self.effective,
                back_stub,
                self.frequency,
                self.modifier,
                self.eom,
                roll,
                self.calendar,
            )
            if not valid:
                raise ValueError("date, stub and roll inputs are invalid")
            else:
                self.ueffective = parsed_args["ueffective"]
                self.utermination = self.termination
                self.front_stub = NoInput(0)
                self.back_stub = parsed_args["utermination"]
                self.roll = parsed_args["roll"]

    def _attribute_schedules(self):
        """Attributes additional schedules according to date adjust and payment lag."""
        self.aschedule = [_adjust_date(dt, self.modifier, self.calendar) for dt in self.uschedule]
        self.pschedule = [
            add_tenor(dt, f"{self.payment_lag}B", "NONE", self.calendar, NoInput(0))
            for dt in self.aschedule
        ]
        self.stubs = [False] * (len(self.uschedule) - 1)
        if self.front_stub is not NoInput(0):
            self.stubs[0] = True
        if self.back_stub is not NoInput(0):
            self.stubs[-1] = True

    def __repr__(self):
        str = (
            f"freq: {self.frequency},  stub: {self.stub},  roll: {self.roll}"
            f",  pay lag: {self.payment_lag},  modifier: {self.modifier}\n"
        )
        return str + self.table.__repr__()

    @property
    def table(self):
        """
        DataFrame : Rows of schedule dates and information.
        """
        df = DataFrame(
            {
                defaults.headers["stub_type"]: ["Stub" if _ else "Regular" for _ in self.stubs],
                defaults.headers["u_acc_start"]: self.uschedule[:-1],
                defaults.headers["u_acc_end"]: self.uschedule[1:],
                defaults.headers["a_acc_start"]: self.aschedule[:-1],
                defaults.headers["a_acc_end"]: self.aschedule[1:],
                defaults.headers["payment"]: self.pschedule[1:],
            }
        )
        return df

    @property
    def n_periods(self):
        """
        int : Number of periods contained in the schedule.
        """
        return len(self.aschedule[1:])


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _is_divisible_months(date1: datetime, date2: datetime, frequency_months: int):
    """
    Test whether two dates' months define a period divisible by frequency months.

    Parameters
    ----------
    date1 : datetime,
        Start date.
    date2 : datetime,
        End date.
    frequency_months : datetime,
        Number of months within a period.

    Returns
    -------
    bool

    Notes
    -----
    All default frequencies divide into one year so the number of years between dates
    is not relevant to this calculation
    """
    months = date2.month - date1.month
    # months += (date2.year - date1.year) * 12
    return months % frequency_months == 0


def _get_unadjusted_roll(ueffective: datetime, utermination: datetime, eom: bool):
    """
    Infer a roll day from given effective and termination dates of a regular swap.

    Parameters
    ----------
    ueffective : Datetime
        The unadjusted effective date.
    utermination: Datetime
        The unadjusted termination date.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for inference.
        False by default.

    Returns
    -------
    str, int : Roll day, zero if invalid.

    Notes
    -----
    If dates are not month end this will return 0 if the day numbers are not equal,
    otherwise it must return the day number.

    If dates are month end then based on the date category the inferred roll
    date is extracted from the below static mappings.

    An ``eom`` preference would classify 30 March to 30 September as *"eom"* roll day
    and not 30. This has relevance for a periods where the number of days in the month
    is 31.
    """
    if ueffective.day < 28 or utermination.day < 28:
        if ueffective.day == utermination.day:
            return ueffective.day
        else:
            return 0

    e_cat = _get_date_category(ueffective)
    t_cat = _get_date_category(utermination)

    non_eom_map = [
        [28, 28, 29, 30, 31, 30, 29, 28],
        [28, 28, 0, 0, 0, 0, 0, 28],
        [29, 0, 29, 30, 31, 30, 29, 0],
        [30, 0, 30, 30, 31, 30, 0, 0],
        [31, 0, 31, 31, 31, 0, 0, 0],
        [30, 0, 30, 30, 0, 30, 0, 0],
        [29, 0, 29, 0, 0, 0, 29, 0],
        [28, 28, 0, 0, 0, 0, 0, 28],
    ]
    eom_map = [
        ["eom", 28, "eom", "eom", "eom", 30, 29, 28],
        [28, 28, 0, 0, 0, 0, 0, 28],
        ["eom", 0, "eom", "eom", "eom", 30, 29, 0],
        ["eom", 0, "eom", "eom", "eom", 30, 0, 0],
        ["eom", 0, "eom", "eom", "eom", 0, 0, 0],
        [30, 0, 30, 30, 0, 30, 0, 0],
        [29, 0, 29, 0, 0, 0, 29, 0],
        [28, 28, 0, 0, 0, 0, 0, 28],
    ]
    if eom:
        return eom_map[e_cat][t_cat]
    else:
        return non_eom_map[e_cat][t_cat]


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_date_category(date: datetime):
    """
    Assign a date to a specific category for roll parsing.

    Parameters
    ----------
    date : Datetime
        The date to categorise.

    Returns
    -------
    int : category

    Notes
    -----
    Categories are:

     - 0: Month is February, non-leap year, day is 28, roll -> {28, 29, 30, 31, eom}
     - 1: Month is February, leap year, day is 28, roll -> {28}
     - 2: Month is February, day is 29, roll -> {29, 30, 31, eom}
     - 3: Month is in [Apr, Jun, Sep, Nov], day is 30, roll -> {30, 31, eom}
     - 4: Month is in [Jan, Mar, May, Jul, Aug, Oct, Dec], day is 31, roll -> {31, eom}
     - 5: Month is in [Jan, Mar, May, Jul, Aug, Oct, Dec], day is 30, roll -> {30}
     - 6: Month is not February, day is 29, roll -> {29}
     - 7: Month is not February, day is 28, roll -> {28}

    Raises if cannot categorise, i.e. if day < 28.
    """
    if date.month == 2:
        if calendar_mod.monthrange(date.year, 2)[1] == 28 and date.day == 28:
            return 0
        elif calendar_mod.monthrange(date.year, 2)[1] == 29 and date.day == 28:
            return 1
        elif calendar_mod.monthrange(date.year, 2)[1] == 29 and date.day == 29:
            return 2
    else:
        if date.month in [4, 6, 9, 11] and date.day == 30:
            return 3
        elif date.month in [1, 3, 5, 7, 8, 10, 12] and date.day == 31:
            return 4
        elif date.month in [1, 3, 5, 7, 8, 10, 12] and date.day == 30:
            return 5
        elif date.day == 29:
            return 6
        elif date.day == 28:
            return 7
    raise ValueError("Category not defined for day < 28.")


def _check_unadjusted_regular_swap(
    ueffective: datetime,
    utermination: datetime,
    frequency: str,
    eom: bool,
    roll: Optional[Union[str, int]],
):
    """
    Test whether given parameters define a regular leg without stubs.

    Parameters
    ----------
    ueffective : Datetime
        The unadjusted effective date.
    utermination: Datetime
        The unadjusted termination date.
    frequency : str
        The frequency of the leg.
    eom : bool
        Use an end of month preference rather than regular rolls for inference. Set by
        default.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. Inferred if not given.

    Returns
    -------
    tuple : bool, and either kwargs or error message.

    Notes
    -----
    This calculation is performed all relative to the roll. If the effective and
    terminations do not align with it under any combination then a regular swap
    cannot be created.
    """
    frequency_months = defaults.frequency_months[frequency.upper()]
    freq_check = _is_divisible_months(ueffective, utermination, frequency_months)
    if not freq_check:
        return False, "Months date separation not aligned with frequency."

    roll = "eom" if roll == 31 else roll
    for roll_, _is_roll in [("eom", _is_eom), ("imm", _is_imm), ("som", _is_som)]:
        if str(roll).lower() == roll_:
            if not _is_roll(ueffective):
                return False, f"Non-{roll_} effective date with {roll_} rolls."
            if not _is_roll(utermination):
                return False, f"Non-{roll_} termination date with {roll_} rolls."

    if isinstance(roll, int):
        if roll in [29, 30]:
            if ueffective.day != roll and not (ueffective.month == 2 and _is_eom(ueffective)):
                return False, f"Effective date not aligned with {roll} rolls."
            if utermination.day != roll and not (utermination.month == 2 and _is_eom(utermination)):
                return False, f"Termination date not aligned with {roll} rolls."
        else:
            if ueffective.day != roll:
                return False, f"Effective date not aligned with {roll} rolls."
            if utermination.day != roll:
                return False, f"Termination date not aligned with {roll} rolls."

    if roll is NoInput.blank:
        roll = _get_unadjusted_roll(ueffective, utermination, eom)
        if roll == 0:
            return False, "Roll day could not be inferred from given dates."

    return True, {
        "ueffective": ueffective,
        "utermination": utermination,
        "frequency": frequency,
        "roll": roll,
        "eom": eom,
    }


def _check_regular_swap(
    effective: datetime,
    termination: datetime,
    frequency: str,
    modifier: str,
    eom: bool,
    roll: Optional[Union[str, int]],
    calendar: CustomBusinessDay,
):
    """
    Tests whether the given the parameters define a regular leg schedule without stubs.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime
        The adjusted or unadjusted termination date.
    frequency : str in {}, optional
        The frequency of the schedule.
    modifier : str,
        The date modification rule in {'NONE', 'F', 'MF', 'P', 'MP'}.
    eom : bool
        Use an end of month preference for rolls instead of 28, 29, or 30.
    roll : str, int, optional, set by Default
        The roll day for the schedule in [0, 31] or {"eom", "som", "imm"}.
    calendar : Calendar, optional, set by Default
         The holiday calendar used for adjusting dates.

    Notes
    -----
    This function first assumes that the given effective and termination dates are
    unadjusted and checks if it is a regular swap. If not the function attempts to
    reverse-adjust the dates to their possible unadjusted values and re check the
    validity of a regular swap. This is only possible if a given effective or
    termination date is surrounded by a holiday which could have been modified to
    yield the given date.

    Priority is always given to test the given dates first and any values close to
    those. The termination dates are trialed ahead of the effective dates, thus if a
    roll day is not specified it will likely be inferred from the effective date.

    When trading interbank swaps there is an ambiguous situation that can arise in
    regards to forward starting tenors. For example suppose that the current effective
    date is Fri 4th March 2022 and a 1Y1Y swap is traded. This can have the
    following implications:

    - Sat 4th March 2023 (1Y forward) is modified to Mon 6th March effective date and
      the swap is then defined as 1Y out of 6th March making the termination date
      Wed 6th March 2024. The swap is defined with a 6th roll as per the effective
      date.
    - Sat 4th March 2023 is modified to Mon 6th March but the termination date remains
      as the valid date measured 1Y from 4th March, i.e. Mon 4th March 2024. The
      swap is set to have 4th rolls. (These 4th rolls would even be valid if the
      termination date had to be modified to the 5th, or 6th, say).

    Although the first bullet is prevalent in the GBP market, other
    markets such as EUR adopt the second approach, and the second also provides a more
    consistent framework with which to hedge using par tenors so we adopt the second
    in this method.
    """
    _ueffectives = _get_unadjusted_date_alternatives(effective, modifier, calendar)
    _uterminations = _get_unadjusted_date_alternatives(termination, modifier, calendar)

    err_str = ""
    for _ueff, _uterm in product(_ueffectives, _uterminations):
        ret = _check_unadjusted_regular_swap(_ueff, _uterm, frequency, eom, roll)
        if ret[0]:
            return ret
        else:
            err_str += ret[1] + "\n"
    return False, f"All unadjusted date combinations exhuasted:\n{err_str}"


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _is_invalid_very_short_stub(
    date_to_modify: datetime,
    date_fixed: datetime,
    modifier: str,
    calendar: CustomBusinessDay,
):
    """
    This tests that a very short, i.e. 1 to a few days, stub has not been erroneously
    generated. Short stubs are invalid if there is one genuine business day in the
    window.
    """
    # _ = date_range(start=date1, end=date2, freq=calendar)
    if _is_holiday(date_to_modify, calendar):
        date1_ = add_tenor(date_to_modify, "1b", modifier, calendar)
    else:
        date1_ = date_to_modify

    if _is_holiday(date_fixed, calendar):
        date2_ = add_tenor(date_fixed, "1b", modifier, calendar)
    else:
        date2_ = date_fixed

    if date1_ == date2_:
        return True  # date range created by stubs is too small and is invalid
    return False


def _infer_stub_date(
    effective: datetime,
    termination: datetime,
    frequency: str,
    stub: str,
    front_stub: Union[datetime, NoInput],
    back_stub: Union[datetime, NoInput],
    modifier: str,
    eom: bool,
    roll: Union[str, int, NoInput],
    calendar: CustomBusinessDay,
) -> tuple[bool, Any]:
    """
    Attempts to infer either a front or back stub in an unspecified schedule.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime
        The adjusted or unadjusted termination date.
    frequency : str in {}, optional
        The frequency of the schedule.
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}
        The stub type to enact on the swap. Can provide two types, for
        example "SHORTFRONTLONGBACK".
    front_stub : datetime, optional
        An adjusted or unadjusted date for the first stub period.
    back_stub : datetime, optional
        An adjusted or unadjusted date for the back stub period.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    modifier : str
        The date modification rule in {'NONE', 'F', 'MF', 'P', 'MP'}.
    eom : bool
        Use an end of month preference for rolls instead of 28, 29, or 30.
    roll : str, int, optional, set by Default
        The roll day for the schedule in [0, 31] or {"eom", "som", "imm"}.
    calendar : Calendar, optional, set by Default
         The holiday calendar used for adjusting dates.

    Returns
    -------
    tuple : bool, and either kwargs or error message.

    Notes
    -----
    Only the following inferences are possible:

    - ``front_stub``: when ``stub`` is front sided only and ``back_stub`` is *NoInput*.
    - ``front_stub``: when ``stub`` is dual sided and ``back_stub`` is specified.
    - ``back_stub``: when ``stub`` is back sided only and ``front_stub`` is *NoInput*.
    - ``back_stub``: when ``stub`` is dual sided and ``front_stub`` is specified.

    """
    if "FRONT" in stub and "BACK" in stub:  # stub is dual sided
        dead_front_stub, dead_back_stub = False, False
        if front_stub is NoInput.blank:
            assert isinstance(back_stub, datetime)
            valid, parsed_args = _check_regular_swap(
                effective, back_stub, frequency, modifier, eom, roll, calendar
            )
            if valid:  # no front stub is required
                return True, {
                    "ueffective": parsed_args["ueffective"],
                    "utermination": termination,
                    "front_stub": NoInput(0),
                    "back_stub": parsed_args["utermination"],
                    "roll": parsed_args["roll"],
                    "frequency": parsed_args["frequency"],
                    "eom": parsed_args["eom"],
                }
            else:
                stub_ = _get_default_stub("FRONT", stub)
                front_stub = _get_unadjusted_stub_date(
                    effective, back_stub, frequency, stub_, eom, roll
                )
                dead_front_stub = _is_invalid_very_short_stub(
                    effective, front_stub, modifier, calendar
                )
        else:
            valid, parsed_args = _check_regular_swap(
                front_stub, termination, frequency, modifier, eom, roll, calendar
            )
            if valid:  # no back stub is required
                return True, {
                    "ueffective": effective,
                    "utermination": parsed_args["utermination"],
                    "front_stub": parsed_args["ueffective"],
                    "back_stub": NoInput(0),
                    "roll": parsed_args["roll"],
                    "frequency": parsed_args["frequency"],
                    "eom": parsed_args["eom"],
                }
            else:
                stub_ = _get_default_stub("BACK", stub)
                back_stub = _get_unadjusted_stub_date(
                    front_stub, termination, frequency, stub_, eom, roll
                )
                dead_back_stub = _is_invalid_very_short_stub(
                    back_stub, termination, modifier, calendar
                )
        valid, parsed_args = _check_regular_swap(
            front_stub, back_stub, frequency, modifier, eom, roll, calendar
        )
        if not valid:
            return valid, parsed_args
        else:
            return True, {
                "ueffective": effective if not dead_front_stub else parsed_args["ueffective"],
                "utermination": termination if not dead_back_stub else parsed_args["utermination"],
                "front_stub": parsed_args["ueffective"] if not dead_front_stub else NoInput(0),
                "back_stub": parsed_args["utermination"] if not dead_back_stub else NoInput(0),
                "roll": parsed_args["roll"],
                "frequency": parsed_args["frequency"],
                "eom": parsed_args["eom"],
            }
    elif "FRONT" in stub:
        valid, parsed_args = _check_regular_swap(
            effective, termination, frequency, modifier, eom, roll, calendar
        )
        if valid:  # no front stub is required
            return True, {
                "ueffective": parsed_args["ueffective"],
                "utermination": parsed_args["utermination"],
                "front_stub": NoInput(0),
                "back_stub": NoInput(0),
                "roll": parsed_args["roll"],
                "frequency": parsed_args["frequency"],
                "eom": parsed_args["eom"],
            }
        else:
            stub_ = _get_default_stub("FRONT", stub)
            front_stub = _get_unadjusted_stub_date(
                effective, termination, frequency, stub_, eom, roll
            )

            # The following check prohibits stubs that are too short under calendar,
            # e.g. 2 May 27 is a Sunday and 3 May 27 is a Monday => dead_stub is True
            dead_stub = _is_invalid_very_short_stub(effective, front_stub, modifier, calendar)

            valid, parsed_args = _check_regular_swap(
                front_stub, termination, frequency, modifier, eom, roll, calendar
            )
            if not valid:
                return valid, parsed_args
            else:
                return True, {
                    "ueffective": effective if not dead_stub else parsed_args["ueffective"],
                    "utermination": parsed_args["utermination"],
                    "front_stub": parsed_args["ueffective"] if not dead_stub else NoInput(0),
                    "back_stub": NoInput(0),
                    "roll": parsed_args["roll"],
                    "frequency": parsed_args["frequency"],
                    "eom": parsed_args["eom"],
                }
    else:  # schedule is "BACK" sided
        valid, parsed_args = _check_regular_swap(
            effective, termination, frequency, modifier, eom, roll, calendar
        )
        if valid:  # no back stub is required
            return True, {
                "ueffective": parsed_args["ueffective"],
                "utermination": parsed_args["utermination"],
                "front_stub": NoInput(0),
                "back_stub": NoInput(0),
                "roll": parsed_args["roll"],
                "frequency": parsed_args["frequency"],
                "eom": parsed_args["eom"],
            }
        else:
            stub_ = _get_default_stub("BACK", stub)
            back_stub = _get_unadjusted_stub_date(
                effective, termination, frequency, stub_, eom, roll
            )

            # The following check prohibits stubs that are too short under calendar,
            # 19 Oct 47 is a Saturday and 20 Oct 47 is a Sunday => dead_stub is True
            dead_stub = _is_invalid_very_short_stub(back_stub, termination, modifier, calendar)

            valid, parsed_args = _check_regular_swap(
                effective, back_stub, frequency, modifier, eom, roll, calendar
            )
            if not valid:
                return valid, parsed_args
            else:
                return True, {
                    "ueffective": parsed_args["ueffective"],
                    "utermination": termination if not dead_stub else parsed_args["utermination"],
                    "front_stub": NoInput(0),
                    "back_stub": parsed_args["utermination"] if not dead_stub else NoInput(0),
                    "roll": parsed_args["roll"],
                    "frequency": parsed_args["frequency"],
                    "eom": parsed_args["eom"],
                }


def _get_default_stub(side: str, stub: str) -> str:
    if f"SHORT{side}" in stub:
        return f"SHORT{side}"
    elif f"LONG{side}" in stub:
        return f"LONG{side}"
    else:
        return f"{defaults.stub_length}{side}"


def _get_unadjusted_stub_date(
    ueffective: datetime,
    utermination: datetime,
    frequency: str,
    stub: str,
    eom: bool,
    roll: Union[int, str, NoInput],
) -> datetime:
    """
    Return an unadjusted stub date inferred from the dates and frequency.

    Parameters
    ----------
    ueffective : datetime
        The unadjusted effective date.
    utermination : datetime
        The unadjusted termination date.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}
        The frequency of the schedule.
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}
        The specification of the stub type..
    eom : bool
        Use an end of month preference for rolls instead of 28, 29, or 30.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day for the schedule.

    Returns
    -------
    datetime
    """
    # frequency_months = defaults.frequency_months[frequency]
    stub_side = "FRONT" if "FRONT" in stub else "BACK"
    if "LONG" in stub:
        _ = _get_unadjusted_short_stub_date(
            ueffective, utermination, frequency, stub_side, eom, roll
        )
        if "FRONT" in stub:
            ueffective = _ + timedelta(days=1)
        else:  # "BACK" in stub
            utermination = _ - timedelta(days=1)

    return _get_unadjusted_short_stub_date(
        ueffective,
        utermination,
        frequency,
        stub_side,
        eom,
        roll,
    )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_unadjusted_short_stub_date(
    ueffective: datetime,
    utermination: datetime,
    frequency: str,
    stub_side: str,
    eom: bool,
    roll: Union[int, str, NoInput],
):
    """
    Return an unadjusted short stub date inferred from the dates and frequency.

    Parameters
    ----------
    ueffective : datetime
        The unadjusted effective date.
    utermination : datetime
        The unadjusted termination date.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}
        The frequency of the schedule.
    stub_side : str in {"FRONT", "BACK"}
        The side of the schedule to infer a short stub.
    eom : bool
        Use an end of month preference for rolls instead of 28, 29, or 30.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day for the schedule.

    Returns
    -------
    datetime
    """
    if stub_side == "FRONT":
        stub_side_dt, reg_side_dt, direction = ueffective, utermination, 1
    else:  # stub_side == "BACK":
        stub_side_dt, reg_side_dt, direction = utermination, ueffective, -1

    if roll is NoInput.blank:
        roll = "eom" if (eom and _is_eom(reg_side_dt)) else reg_side_dt.day

    frequency_months = defaults.frequency_months[frequency]

    if _is_divisible_months(ueffective, utermination, frequency_months):
        if stub_side == "FRONT":
            comparison = _get_roll(ueffective.month, ueffective.year, roll)
            if ueffective.day > comparison.day:
                _ = _add_months(ueffective, frequency_months * direction, "NONE", NoInput(0), roll)
                _ = _get_roll(_.month, _.year, roll)
            else:
                _ = ueffective
                _ = _get_roll(_.month, _.year, roll)

        else:  # stub_side == "BACK"
            comparison = _get_roll(utermination.month, utermination.year, roll)
            if utermination.day < comparison.day:
                _ = _add_months(
                    utermination, frequency_months * direction, "NONE", NoInput(0), roll
                )
                _ = _get_roll(_.month, _.year, roll)
            else:
                _ = utermination
                _ = _get_roll(_.month, _.year, roll)

    else:
        for month_offset in range(1, 12):
            stub_date = _add_months(
                stub_side_dt, month_offset * direction, "NONE", NoInput(0), roll
            )
            if _is_divisible_months(stub_date, reg_side_dt, frequency_months):
                break
        # _ = _get_roll(stub_date.month, stub_date.year, roll)
        _ = stub_date

    return _


# Book coded - This defines all the unadjusted date scheduling - no inference

# The code below this line should be tested and is confident that it functions


def _generate_irregular_schedule_unadjusted(
    ueffective: datetime,
    utermination: datetime,
    frequency: str,
    roll: Union[int, str],
    ufront_stub: Optional[datetime],
    uback_stub: Optional[datetime],
) -> Iterator[datetime]:
    """
    Generate unadjusted dates defining an irregular swap schedule.

    Parameters
    ----------
    ueffective : datetime
        The unadjusted effective date.
    utermination : datetime
        The unadjusted termination date.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}
        The frequency of the schedule.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}
        The roll day for the schedule.
    ufront_stub : datetime, optional
        The unadjusted front stub date.
    uback_stub : datetime, optional
        The unadjusted back stub date.

    Yields
    ------
    datetime
    """
    if ufront_stub is NoInput(0):
        yield from _generate_regular_schedule_unadjusted(
            ueffective, utermination if uback_stub is NoInput.blank else uback_stub, frequency, roll
        )
    else:
        yield ueffective
        yield from _generate_regular_schedule_unadjusted(
            ufront_stub,
            utermination if uback_stub is NoInput.blank else uback_stub,
            frequency,
            roll,
        )
    if uback_stub is not NoInput(0):
        yield utermination


def _generate_regular_schedule_unadjusted(
    ueffective: datetime, utermination: datetime, frequency: str, roll: Union[int, str]
) -> Iterator[datetime]:
    """
    Generates unadjusted dates defining a regular swap schedule.

    Parameters
    ----------
    ueffective : datetime
        The unadjusted effective date.
    utermination : datetime
        The unadjusted termination date, which aligns in a regular swap sense with
        ueffective.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}
        The frequency of the schedule.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}
        The roll day for the schedule.

    Yields
    ------
    datetime

    Notes
    -----
    ``roll`` of 1 and "som" are semantically identical.
    ``roll`` of 31 and "eom" are semantically identical.

    Errors are not raised if ``utermination`` does not define a regular swap
    associated with ``ueffective``.
    """
    n_periods = _get_n_periods_in_regular(ueffective, utermination, frequency)
    _ = ueffective
    yield _
    for i in range(int(n_periods)):
        _ = _add_months(_, defaults.frequency_months[frequency], "NONE", NoInput(0), roll)
        # _ = _get_roll(_.month, _.year, roll)
        yield _


# Utility Functions


def _get_unadjusted_date_alternatives(date: datetime, modifier: str, cal: CustomBusinessDay):
    """
    Return all possible unadjusted dates that result in given date under modifier/cal.

    Parameters
    ----------
    date : Datetime
        Adjusted date for which unadjusted dates can be modified to.
    modifier : str
        |modifier|
    calendar : Calendar
        |calendar|

    Returns
    -------
    list : of valid unadjusted dates
    """
    unadj_dates = [date]
    if _is_holiday(date, cal):
        return unadj_dates  # no other unadjusted date can adjust to a holiday.
    for days in range(1, 20):
        possible_unadjusted_date = date + timedelta(days=days)
        if not _is_holiday(possible_unadjusted_date, cal):
            break  # if not a holiday no later date will adjust back to date.
        if date == _adjust_date(possible_unadjusted_date, modifier, cal):
            unadj_dates.append(possible_unadjusted_date)
    for days in range(1, 20):
        possible_unadjusted_date = date - timedelta(days=days)
        if not _is_holiday(possible_unadjusted_date, cal):
            break  # if not a holiday no previous date will adjust back to date.
        if date == _adjust_date(possible_unadjusted_date, modifier, cal):
            unadj_dates.append(possible_unadjusted_date)
    return unadj_dates


def _get_n_periods_in_regular(
    effective: datetime,
    termination: datetime,
    frequency: str,
) -> int:
    """
    Determine the number of regular periods between effective and termination.

    .. warning::
       This method should only be used for dates known to define a regular schedule.

    Parameters
    ----------
    effective : datetime
        The effective date of the schedule.
    termination : datetime
        The termination date of the schedule which aligns in a regular swap sense
        with ``effective``.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}
        The frequency of the schedule.

    Returns
    -------
    int
    """
    if frequency == "Z":
        return 1
    frequency_months = defaults.frequency_months[frequency]
    n_months = (termination.year - effective.year) * 12 + termination.month - effective.month
    if n_months % frequency_months != 0:
        raise ValueError("Regular schedule not implied by `frequency` and dates.")
    return int(n_months / frequency_months)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
