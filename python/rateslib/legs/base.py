from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from pandas import DataFrame, Series, concat

from rateslib import defaults
from rateslib.calendars import add_tenor
from rateslib.default import NoInput, _drb
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    IndexCashflow,
    IndexFixedPeriod,
)
from rateslib.scheduling import Schedule

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CalInput,
        Curve,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        Period,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class BaseLeg(metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``Leg`` subclasses.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A", "Z"}, optional
        The frequency of the schedule.
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
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data. See :meth:`~rateslib.calendars.get_calendar`.
    payment_lag : int, optional
        The number of business days to lag payments by on regular accrual periods.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    initial_exchange : bool
        Whether to also include an initial notional exchange.
    final_exchange : bool
        Whether to also include a final notional exchange and interim amortization
        notional exchanges.

    Notes
    -----
    See also :class:`~rateslib.scheduling.Schedule` for a more thorough description
    of some of these scheduling arguments.

    The (optional) initial cashflow notional is set as the negative of the notional.
    The payment date is set equal to the accrual start date adjusted by
    the ``payment_lag_exchange``.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accrual date adjusted by ``payment_lag_exchange``.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods as interim exchanges if
    ``final_exchange`` is *True*. Payment dates adhere to the ``payment_lag_exchange``.

    Examples
    --------
    See :ref:`Leg Examples<legs-doc>`

    Attributes
    ----------
    schedule : Schedule
    currency : str
    convention : str
    periods : list
    initial_exchange : bool
    final_exchange : bool
    payment_lag_exchange : int

    See Also
    --------
    FixedLeg : Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.
    FloatLeg : Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.
    IndexFixedLeg : Create a fixed leg composed of :class:`~rateslib.periods.IndexFixedPeriod` s.
    ZeroFixedLeg : Create a zero coupon leg composed of a :class:`~rateslib.periods.FixedPeriod`.
    ZeroFloatLeg : Create a zero coupon leg composed of a :class:`~rateslib.periods.FloatPeriod` s.
    ZeroIndexLeg : Create a zero coupon leg composed of :class:`~rateslib.periods.IndexFixedPeriod`.
    CustomLeg : Create a leg composed of user specified periods.
    """

    periods: list[Period]

    @abstractmethod
    def __init__(
        self,
        effective: datetime,
        termination: datetime | str,
        frequency: str,
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: int | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        convention: str_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
    ) -> None:
        self.schedule = Schedule(
            effective,
            termination,
            frequency,
            stub,
            front_stub,
            back_stub,
            roll,
            eom,
            modifier,
            calendar,
            payment_lag,
        )
        self.convention: str = _drb(defaults.convention, convention)
        self.currency: str = _drb(defaults.base_currency, currency).lower()
        self.payment_lag_exchange: int = _drb(defaults.payment_lag_exchange, payment_lag_exchange)
        self.initial_exchange: bool = initial_exchange
        self.final_exchange: bool = final_exchange
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._amortization: DualTypes = _drb(0.0, amortization)
        if getattr(self, "_delay_set_periods", False):
            pass
        else:
            self._set_periods()

    @property
    def notional(self) -> DualTypes:
        return self._notional

    @notional.setter
    def notional(self, value: DualTypes) -> None:
        self._notional = value
        self._set_periods()

    @property
    def amortization(self) -> DualTypes:
        return self._amortization

    @amortization.setter
    def amortization(self, value: float) -> None:
        self._amortization = value
        self._set_periods()

    @abstractmethod
    def _set_periods(self) -> None:
        """Combine all period collection types into an ordered list."""
        self._set_exchange_periods()  # initial and final exchange
        self._set_regular_periods()  # normal coupon periods
        self._set_interim_exchange_periods()  # amortisation exchanges

        periods_: list[Period] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])

        if self._interim_exchange_periods is not None:
            interleaved_periods_: list[Period] = [
                val
                for pair in zip(self._regular_periods, self._interim_exchange_periods, strict=False)
                for val in pair
            ]
            interleaved_periods_.append(self._regular_periods[-1])  # add last regular period
        else:
            interleaved_periods_ = list(self._regular_periods)
        periods_.extend(interleaved_periods_)

        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        self.periods = periods_

    def _set_exchange_periods(self) -> None:
        """Set default cashflow exchanges on Legs with `initial_exchange` or `final_exchange`."""

        periods_: list[Cashflow | None] = [None, None]

        if self.initial_exchange:
            periods_[0] = Cashflow(
                notional=-self.notional,
                payment=self.schedule.calendar.lag(
                    self.schedule.aschedule[0],
                    self.payment_lag_exchange,
                    True,
                ),
                currency=self.currency,
                stub_type="Exchange",
            )

        if self.final_exchange:
            periods_[1] = Cashflow(
                notional=self.notional - self.amortization * (self.schedule.n_periods - 1),
                payment=self.schedule.calendar.lag(
                    self.schedule.aschedule[-1],
                    self.payment_lag_exchange,
                    True,
                ),
                currency=self.currency,
                stub_type="Exchange",
            )

        self._exchange_periods: tuple[Cashflow | None, Cashflow | None] = tuple(periods_)  # type: ignore[assignment]

    def _set_interim_exchange_periods(self) -> None:
        """Set cashflow exchanges if `amortization` and `final_exchange` are present."""
        if not self.final_exchange or self.amortization == 0:
            self._interim_exchange_periods: tuple[Cashflow, ...] | None = None
        else:
            periods_ = [
                Cashflow(
                    notional=self.amortization,
                    payment=self.schedule.calendar.lag(
                        self.schedule.aschedule[i + 1],
                        self.payment_lag_exchange,
                        True,
                    ),
                    currency=self.currency,
                    stub_type="Amortization",
                )
                for i in range(self.schedule.n_periods - 1)
            ]
            self._interim_exchange_periods = tuple(periods_)

    def _set_regular_periods(self) -> None:
        self._regular_periods: tuple[Period, ...] = tuple(
            [
                self._regular_period(
                    start=period[defaults.headers["a_acc_start"]],
                    end=period[defaults.headers["a_acc_end"]],
                    payment=period[defaults.headers["payment"]],
                    stub=period[defaults.headers["stub_type"]] == "Stub",
                    notional=self.notional - self.amortization * i,
                    iterator=i,
                )
                for i, period in enumerate(self.schedule.table.to_dict(orient="index").values())
            ]
        )

    @abstractmethod
    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> Period:
        # implemented by individual legs to satisfy generic `set_periods` methods
        pass  # pragma: no cover

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    # def _regular_period(self, *args: Any, **kwargs: Any) -> Any:
    #    pass

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *Leg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        _ = (period.analytic_delta(*args, **kwargs) for period in self.periods)
        ret: DualTypes = sum(_)
        return ret

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *Leg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        seq = [period.cashflows(*args, **kwargs) for period in self.periods]
        return DataFrame.from_records(seq)

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        """
        Return the NPV of the *Leg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        _is_local = (len(args) >= 5 and args[4]) or kwargs.get("local", False)
        if _is_local:
            _ = (period.npv(*args, **kwargs)[self.currency] for period in self.periods)  # type: ignore[index]
            return {self.currency: sum(_)}
        else:
            _ = (period.npv(*args, **kwargs) for period in self.periods)
            ret: DualTypes = sum(_)
            return ret

    # @property
    # def _is_linear(self) -> bool:
    #     """
    #     Tests if analytic delta spread is a linear function affecting NPV.
    #
    #     This is non-linear if the spread is itself compounded, which only occurs
    #     on RFR trades with *"isda_compounding"* or *"isda_flat_compounding"*, which
    #     should typically be avoided anyway.
    #
    #     Returns
    #     -------
    #     bool
    #     """
    #     # ruff: noqa: SIM103
    #     return True

    def _spread(
        self,
        target_npv: DualTypes,
        fore_curve: CurveOption_,
        disc_curve: CurveOption_,
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculates an adjustment to the ``fixed_rate`` or ``float_spread`` to match
        a specific target NPV.

        Parameters
        ----------
        target_npv : float, Dual or Dual2
            The target NPV that an adjustment to the parameter will achieve. **Must
            be in local currency of the leg.**
        fore_curve : Curve or LineCurve
            The forecast curve passed to analytic delta calculation.
        disc_curve : Curve
            The discounting curve passed to analytic delta calculation.
        fx : FXForwards, optional
            Required for multi-currency legs which are MTM exchanged.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        ``FixedLeg`` and ``FloatLeg`` with a *"none_simple"* spread compound method have
        linear sensitivity to the spread. This can be calculated directly and
        exactly using an analytic delta calculation.

        *"isda_compounding"* and *"isda_flat_compounding"* spread compound methods
        have non-linear sensitivity to the spread. This requires a root finding,
        iterative algorithm, which, coupled with very poor performance of calculating
        period rates under this method is exceptionally slow. We approximate this
        using first and second order AD and extrapolate a solution as a Taylor
        expansion. This results in approximation error.

        Examples
        --------
        """
        a_delta = self.analytic_delta(fore_curve, disc_curve, fx, self.currency)
        return -target_npv / a_delta

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"


class _FixedLegMixin:
    """
    Add the functionality to add and retrieve ``fixed_rate`` on
    :class:`~rateslib.periods.FixedPeriod` s.
    """

    convention: str
    schedule: Schedule
    currency: str
    _fixed_rate: DualTypes | NoInput

    @property
    def fixed_rate(self) -> DualTypes_:
        """
        float or NoInput : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes) -> None:
        self._fixed_rate = value
        for period in getattr(self, "periods", []):
            if isinstance(period, FixedPeriod | CreditPremiumPeriod):
                period.fixed_rate = value

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        notional: DualTypes,
        stub: bool,
        iterator: int,
    ) -> FixedPeriod:
        return FixedPeriod(
            fixed_rate=self.fixed_rate,
            start=start,
            end=end,
            payment=payment,
            frequency=self.schedule.frequency,
            notional=notional,
            currency=self.currency,
            convention=self.convention,
            termination=self.schedule.termination,
            stub=stub,
            roll=self.schedule.roll,
            calendar=self.schedule.calendar,
        )


class _FloatLegMixin:
    """
    Add the functionality to add and retrieve ``float_spread`` on
    :class:`~rateslib.periods.FloatPeriod` s and a
    :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
    """

    convention: str
    schedule: Schedule
    currency: str
    _float_spread: DualTypes_
    fixing_method: str
    spread_compound_method: str
    method_param: int
    fixings: list[DualTypes | list[DualTypes] | Series[DualTypes] | NoInput]  # type: ignore[type-var]
    periods: list[Period]

    def _get_fixings_from_series(
        self,
        ser: Series[DualTypes],  # type: ignore[type-var]
        ini_period: int = 0,
    ) -> list[Series[DualTypes] | NoInput]:  # type: ignore[type-var]
        """
        Determine which fixings can be set for Periods with the given Series.
        """
        last_fixing_dt = ser.index[-1]
        if self.fixing_method in [
            "rfr_payment_delay",
            "rfr_lockout",
            "rfr_payment_delay_avg",
            "rfr_lockout_avg",
        ]:
            adj_days = 0
        else:
            adj_days = self.method_param
        first_required_day = [
            add_tenor(
                self.schedule.aschedule[i],
                f"-{adj_days}B",
                "NONE",
                self.schedule.calendar,
            )
            for i in range(ini_period, self.schedule.n_periods)
        ]
        return [ser if last_fixing_dt >= day else NoInput(0) for day in first_required_day]

    def _set_fixings(
        self,
        fixings: FixingsRates_,  # type: ignore[type-var]
    ) -> None:
        """
        Re-organises the fixings input to list structure for each period.
        Requires a ``schedule`` object and ``float_args``.
        """
        if isinstance(fixings, NoInput):
            fixings_: list[DualTypes | list[DualTypes] | Series[DualTypes] | NoInput] = []  # type: ignore[type-var]
        elif isinstance(fixings, Series):
            # oldest fixing at index 0: latest -1
            sorted_fixings: Series[DualTypes] = fixings.sort_index()  # type: ignore[type-var, assignment]
            fixings_ = self._get_fixings_from_series(sorted_fixings)  # type: ignore[assignment]
        elif isinstance(fixings, tuple):
            fixings_ = [fixings[0]]
            fixings_.extend(self._get_fixings_from_series(fixings[1], 1))
        elif not isinstance(fixings, list):
            fixings_ = [fixings]
        else:  # fixings as a list should be remaining
            fixings_ = fixings

        self.fixings = fixings_ + [NoInput(0)] * (self.schedule.n_periods - len(fixings_))

    # def _spread_isda_dual2(
    #     self,
    #     target_npv,
    #     fore_curve,
    #     disc_curve,
    #     fx=NoInput(0),
    # ):  # pragma: no cover
    #     # This method is unused and untested, superseded by _spread_isda_approx_rate
    #
    #     # This method creates a dual2 variable for float spread + obtains derivativs automatically
    #     _fs = self.float_spread
    #     self.float_spread = Dual2(0.0 if _fs is None else float(_fs), "spread_z")
    #
    #     # This method uses ad-hoc AD to solve a specific problem for which
    #     # there is no closed form solution. Calculating NPV is very inefficient
    #     # so, we only do this once as opposed to using a root solver algo
    #     # which would otherwise converge to the exact solution but is
    #     # practically not workable.
    #
    #     # This method is more accurate than the 'spread through approximated
    #     # derivatives' method, but it is a more costly and less robust method
    #     # due to its need to work in second order mode.
    #
    #     fore_ad = fore_curve.ad
    #     fore_curve._set_ad_order(2)
    #
    #     disc_ad = disc_curve.ad
    #     disc_curve._set_ad_order(2)
    #
    #     if isinstance(fx, FXRates | FXForwards):
    #         _fx = None if fx is None else fx._ad
    #         fx._set_ad_order(2)
    #
    #     npv = self.npv(fore_curve, disc_curve, fx, self.currency)
    #     b = gradient(npv, "spread_z", order=1)[0]
    #     a = 0.5 * gradient(npv, "spread_z", order=2)[0][0]
    #     c = -target_npv
    #
    #     # Perform quadratic solution
    #     _1 = -c / b
    #     if abs(a) > 1e-14:
    #         _2a = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
    #         _2b = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)  # alt quadratic soln
    #         if abs(_1 - _2a) < abs(_1 - _2b):
    #             _ = _2a
    #         else:
    #             _ = _2b  # select quadratic soln
    #     else:  # pragma: no cover
    #         # this is to avoid divide by zero errors and return an approximation
    #         _ = _1
    #         warnings.warn(
    #             "Divide by zero encountered and the spread is approximated to " "first order.",
    #             UserWarning,
    #         )
    #
    #     # This is required by the Dual2 AD approach to revert to original order.
    #     self.float_spread = _fs
    #     fore_curve._set_ad_order(fore_ad)
    #     disc_curve._set_ad_order(disc_ad)
    #     if isinstance(fx, FXRates | FXForwards):
    #         fx._set_ad_order(_fx)
    #     _ = set_order(_, disc_ad)  # use disc_ad: credit spread from disc curve
    #
    #     return _

    def _spread_isda_approximated_rate(
        self,
        target_npv: DualTypes,
        fore_curve: Curve,  # TODO: use CurveOption_ and handle dict[str, Curve]
        disc_curve: Curve,  # TODO: use CurveOption_ and handle dict[str, Curve]
    ) -> DualTypes:
        """
        Use approximated derivatives through geometric averaged 1day rates to derive the
        spread
        """
        a: DualTypes = 0.0
        b: DualTypes = 0.0
        for period in [_ for _ in self.periods if isinstance(_, FloatPeriod)]:
            a_, b_ = period._get_analytic_delta_quadratic_coeffs(fore_curve, disc_curve)
            a += a_
            b += b_

        c = -target_npv

        # perform the quadratic solution
        _1 = -c / b
        if abs(a) > 1e-14:
            _2a = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
            _2b = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)  # alt quadratic soln
            if abs(_1 - _2a) < abs(_1 - _2b):
                _: DualTypes = _2a
            else:
                _ = _2b  # select quadratic soln
        else:
            # this is to avoid divide by zero errors and return an approximation
            # also isda_flat_compounding has a=0
            _ = _1

        return _

    @property
    def float_spread(self) -> DualTypes | NoInput:
        """
        float or NoInput : If set will also set the ``float_spread`` of contained
            :class:`~rateslib.periods.FloatPeriod` s.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self._float_spread = value
        if value is NoInput(0):
            _ = 0.0
        else:
            _ = value
        for period in self.periods:
            if isinstance(period, FloatPeriod):
                period.float_spread = _

    # def fixings_table(self, curve: Curve):
    #     """
    #     Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.
    #
    #     Parameters
    #     ----------
    #     curve : Curve
    #         The forecast needed to calculate rates which affect compounding and
    #         dependent notional exposure.
    #
    #     Returns
    #     -------
    #     DataFrame
    #
    #     Notes
    #     -----
    #     The fixing dates given in the table are publication times which is assumed to
    #     be in arrears, i.e. for a fixing period of 4th Jan 2022 to 5th Jan 2022, the
    #     fixing will be published on the 5th Jan 2022. This is universally applied to
    #     all RFR calculations.
    #     """
    #     df = self.periods[0].fixings_table(curve)
    #     for i in range(1, self.schedule.n_periods):
    #         df = pd.concat([df, self.periods[i].fixings_table(curve)])
    #     return df

    def _fixings_table(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        See :meth:`~rateslib.periods.FloatPeriod.fixings_table` for arguments.

        Returns
        -------
        DataFrame
        """
        dfs = []
        for period in self.periods:
            if isinstance(period, FloatPeriod):
                dfs.append(period.fixings_table(*args, **kwargs))

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            return concat(dfs)

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> FloatPeriod:
        return FloatPeriod(
            float_spread=self.float_spread,
            start=start,
            end=end,
            payment=payment,
            frequency=self.schedule.frequency,
            notional=notional,
            currency=self.currency,
            convention=self.convention,
            termination=self.schedule.termination,
            stub=stub,
            roll=self.schedule.roll,
            calendar=self.schedule.calendar,
            fixings=self.fixings[iterator],
            fixing_method=self.fixing_method,
            method_param=self.method_param,
            spread_compound_method=self.spread_compound_method,
        )

    @property
    def _is_linear(self) -> bool:
        """
        Tests if analytic delta spread is a linear function affecting NPV.

        This is non-linear if the spread is itself compounded, which only occurs
        on RFR trades with *"isda_compounding"* or *"isda_flat_compounding"*, which
        should typically be avoided anyway.

        Returns
        -------
        bool
        """
        # ruff: noqa: SIM103
        if "rfr" in self.fixing_method and self.spread_compound_method != "none_simple":
            return False
        return True

    def _spread(
        self,
        target_npv: DualTypes,
        fore_curve: CurveOption_,
        disc_curve: CurveOption_,
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculates an adjustment to the ``fixed_rate`` or ``float_spread`` to match
        a specific target NPV.

        Parameters
        ----------
        target_npv : float, Dual or Dual2
            The target NPV that an adjustment to the parameter will achieve. **Must
            be in local currency of the leg.**
        fore_curve : Curve or LineCurve
            The forecast curve passed to analytic delta calculation.
        disc_curve : Curve
            The discounting curve passed to analytic delta calculation.
        fx : FXForwards, optional
            Required for multi-currency legs which are MTM exchanged.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        ``FixedLeg`` and ``FloatLeg`` with a *"none_simple"* spread compound method have
        linear sensitivity to the spread. This can be calculated directly and
        exactly using an analytic delta calculation.

        *"isda_compounding"* and *"isda_flat_compounding"* spread compound methods
        have non-linear sensitivity to the spread. This requires a root finding,
        iterative algorithm, which, coupled with very poor performance of calculating
        period rates under this method is exceptionally slow. We approximate this
        using first and second order AD and extrapolate a solution as a Taylor
        expansion. This results in approximation error.

        Examples
        --------
        """
        if self._is_linear:
            a_delta: DualTypes = self.analytic_delta(fore_curve, disc_curve, fx, self.currency)  # type: ignore[attr-defined]
            return -target_npv / a_delta
        else:
            return self._spread_isda_approximated_rate(target_npv, fore_curve, disc_curve)  # type: ignore[arg-type]
            # _ = self._spread_isda_dual2(target_npv, fore_curve, disc_curve, fx)


class CustomLeg(BaseLeg):
    """
    Create a leg contained of user specified ``Periods``.

    Useful for crafting amortising swaps with custom notional and date schedules.

    Parameters
    ----------
    periods : iterable of ``Period`` types
        A sequence of ``Periods`` to attach to the leg.

    Attributes
    ----------
    periods : list[Periods]

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.legs import CustomLeg
       from rateslib.periods import FixedPeriod

    .. ipython:: python

       fp1 = FixedPeriod(dt(2021,1,1), dt(2021,7,1), dt(2021,7,2), "Q", 1e6, "Act365F", fixed_rate=2.10)
       fp2 = FixedPeriod(dt(2021,3,7), dt(2021,9,7), dt(2021,9,8), "Q", -5e6, "Act365F", fixed_rate=3.10)
       custom_leg = CustomLeg(periods=[fp1, fp2])
       custom_leg.cashflows(curve)

    """  # noqa: E501

    def __init__(self, periods: list[Period]) -> None:
        if not all(
            isinstance(
                p,
                FloatPeriod
                | FixedPeriod
                | IndexFixedPeriod
                | Cashflow
                | IndexCashflow
                | CreditPremiumPeriod
                | CreditProtectionPeriod,
            )
            for p in periods
        ):
            raise ValueError(
                "Each object in `periods` must be a specific `Period` type.",
            )
        self._set_periods(periods)

    def _set_periods(self, periods: list[Period]) -> None:  # type: ignore[override]
        self.periods: list[Any] = periods

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CustomLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *CustomLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *CustomLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def _regular_period(self, *args: Any, **kwargs: Any) -> Any:
        pass
