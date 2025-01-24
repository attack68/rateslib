from __future__ import annotations

import abc
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from math import prod
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.calendars import add_tenor
from rateslib.curves import Curve, index_left
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    IndexCashflow,
    IndexFixedPeriod,
    IndexMixin,
)
from rateslib.periods.utils import _get_fx_and_base, _validate_float_args
from rateslib.scheduling import Schedule

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        CalInput,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FixingsFx_,
        FixingsRates_,
        Period,
        bool_,
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

    @abc.abstractmethod
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
        return sum(_)

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
            return sum(_)

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
    def fixed_rate(self) -> DualTypes | NoInput:
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


class FixedLeg(_FixedLegMixin, BaseLeg):  # type: ignore[misc]
    """
    Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The rate applied to determine cashflows in % (i.e 5.0 = 5%). Can be left unset and
        designated later, perhaps after a mid-market rate for all periods has been calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a fixed leg is the sum of the period NPVs.

    .. math::

       P = \\underbrace{- R \\sum_{i=1}^n {N_i d_i v_i(m_i)}}_{\\text{regular flows}} \\underbrace{+ N_1 v(m_0) - \\sum_{i=1}^{n-1}v(m_i)(N_{i}-N_{i+1})  - N_n v(m_n)}_{\\text{exchange flows}}

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial R} = \\sum_{i=1}^n {N_i d_i v_i(m_i)}

    Examples
    --------

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       fixed_leg_exch = FixedLeg(
           dt(2022, 1, 1), "9M", "Q",
           fixed_rate=2.0,
           notional=1000000,
           amortization=200000,
           initial_exchange=True,
           final_exchange=True,
       )
       fixed_leg_exch.cashflows(curve)
       fixed_leg_exch.npv(curve)
    """  # noqa: E501

    periods: list[FixedPeriod | Cashflow]  # type: ignore[assignment]
    _regular_periods: tuple[FixedPeriod, ...]

    def __init__(
        self, *args: Any, fixed_rate: DualTypes | NoInput = NoInput(0), **kwargs: Any
    ) -> None:
        self._fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *FixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *FixedLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *FixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def _set_periods(self) -> None:
        return super()._set_periods()


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class _FloatLegMixin:
    """
    Add the functionality to add and retrieve ``float_spread`` on
    :class:`~rateslib.periods.FloatPeriod` s and a
    :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
    """

    convention: str
    schedule: Schedule
    currency: str
    _float_spread: DualTypes | NoInput
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
            sorted_fixings: Series[DualTypes] = fixings.sort_index()  # type: ignore[attr-defined, type-var]
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
            return pd.concat(dfs)

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


class FloatLeg(_FloatLegMixin, BaseLeg):
    """
    Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    float_spread : float, optional
        The spread applied to determine cashflows in bps (i.e. 100 = 1%). Can be set to `None`
        and designated later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, Series, 2-tuple, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``. If a 2-tuple of value and *Series*, the first
        scalar value is applied to the first period and latter periods handled as with *Series*.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a *FloatLeg* is the sum of the period NPVs.

    .. math::

       P = \\underbrace{- \\sum_{i=1}^n {N_i r_i(r_j, z) d_i v_i(m_i)}}_{\\text{regular flows}} \\underbrace{+ N_1 v(m_0) - \\sum_{i=1}^{n-1}v(m_i)(N_{i}-N_{i+1})  - N_n v(m_n)}_{\\text{exchange flows}}

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial z} = \\sum_{i=1}^n {\\frac{\\partial r_i}{\\partial z} N_i d_i v_i(m_i)}


    .. warning::

       When floating rates are determined from historical fixings the forecast
       ``Curve`` ``calendar`` will be used to determine fixing dates.
       If this calendar does not align with the ``Leg`` ``calendar`` then
       spurious results or errors may be generated.

       Including the curve calendar within a *Leg* multi-holiday calendar
       is acceptable, i.e. a *Leg* calendar of *"nyc,ldn,tgt"* and a curve
       calendar of *"ldn"* is valid. A *Leg* calendar of just *"nyc,tgt"* may
       give errors.

    Examples
    --------
    Set the first fixing on an historic IBOR leg.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 12, 1),
           termination="9M",
           frequency="Q",
           fixing_method="ibor",
           fixings=2.00,
       )
       float_leg.cashflows(curve)

    Set multiple fixings on an historic IBOR leg.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 9, 1),
           termination="12M",
           frequency="Q",
           fixing_method="ibor",
           fixings=[1.00, 2.00],
       )
       float_leg.cashflows(curve)

    It is **not** best practice to supply fixings as a list of values. It is better to supply
    a *Series* indexed by IBOR publication date (in this case lagged by zero days).

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 9, 1),
           termination="12M",
           frequency="Q",
           fixing_method="ibor",
           method_param=0,
           fixings=Series([1.00, 2.00], index=[dt(2021, 9, 1), dt(2021, 12, 1)])
       )
       float_leg.cashflows(curve)

    Set the initial RFR fixings in the first period of an RFR leg (notice the sublist
    and the implied -10% year end turn spread).

    .. ipython:: python

       swestr_curve = Curve({dt(2023, 1, 2): 1.0, dt(2023, 7, 2): 0.99}, calendar="stk")
       float_leg = FloatLeg(
           effective=dt(2022, 12, 28),
           termination="2M",
           frequency="M",
           fixings=[[1.19, 1.19, -8.81]],
           currency="SEK",
           calendar="stk"
       )
       float_leg.cashflows(swestr_curve)
       float_leg.fixings_table(swestr_curve)[dt(2022,12,28):dt(2023,1,4)]

    Again, this is poor practice. It is **best practice** to supply a *Series* of RFR rates by
    reference value date.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2022, 12, 28),
           termination="2M",
           frequency="M",
           fixings=Series([1.19, 1.19, -8.81], index=[dt(2022, 12, 28), dt(2022, 12, 29), dt(2022, 12, 30)]),
           currency="SEK",
           calendar="stk",
       )
       float_leg.cashflows(swestr_curve)
       float_leg.fixings_table(swestr_curve)[dt(2022,12,28):dt(2023,1,4)]
    """  # noqa: E501

    _delay_set_periods: bool = True  # do this to set fixings first
    _regular_periods: tuple[FloatPeriod, ...]

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)
        self._set_fixings(fixings)
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *FloatLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        """
        Return the NPV of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def fixings_table(
        self,
        curve: CurveOption_,
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given and ``curve`` is discount factor based.
        fx : float, FXRates, FXForwards, optional
            Only used in the case of :class:`~rateslib.legs.FloatLegMtm` to derive FX fixings.
        base : str, optional
            Not used by ``fixings_table``.
        approximate: bool
            Whether to use a faster (3x) but marginally less accurate (0.1% error) calculation.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        return super()._fixings_table(
            curve=curve, disc_curve=disc_curve, approximate=approximate, right=right
        )

    def _set_periods(self) -> None:
        return super(_FloatLegMixin, self)._set_periods()

    # @property
    # def _is_complex(self):
    #     """
    #     A complex float leg is one which is RFR based and for which each individual
    #     RFR fixing is required is order to calculate correctly. This occurs in the
    #     following cases:
    #
    #     1) The ``fixing_method`` is *"lookback"* - since fixing have arbitrary
    #        weightings misaligned with their standard weightings due to
    #        arbitrary shifts.
    #     2) The ``spread_compound_method`` is not *"none_simple"* - this is because the
    #        float spread is compounded alongside the rates so there is a non-linear
    #        relationship. Note if spread is zero this is negated and can be ignored.
    #     3) The ``fixing_method`` is *"lockout"* - technically this could be made semi
    #        efficient by splitting calculations into two parts. As of now it
    #        remains within the inefficient complex section.
    #     4) ``fixings`` are given which need to be incorporated into the calculation.
    #
    #
    #     """
    #     if self.fixing_method in ["rfr_payment_delay", "rfr_observation_shift"]:
    #         if self.fixings is not None:
    #             return True
    #         elif abs(self.float_spread) < 1e-9 or \
    #                 self.spread_compound_method == "none_simple":
    #             return False
    #         else:
    #             return True
    #     elif self.fixing_method == "ibor":
    #         return False
    #     return True


class _IndexLegMixin:
    schedule: Schedule
    index_method: str
    _index_fixings: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput = NoInput(0)  # type: ignore[type-var]
    _index_base: DualTypes | NoInput = NoInput(0)
    periods: list[IndexFixedPeriod | IndexCashflow | Cashflow]
    index_lag: int

    # def _set_index_fixings_on_periods(self):
    #     """
    #     Re-organises the fixings input to list structure for each period.
    #     Requires a ``schedule`` object and ``float_args``.
    #     """
    #     if self.index_fixings is None:
    #         pass  # do nothing
    #     elif isinstance(self.index_fixings, Series):
    #         for period in self.periods:
    #             period.index_fixings = IndexMixin._index_value(
    #                 i_method=self.index_method,
    #                 i_lag=self.index_lag,
    #                 i_curve=None,
    #                 i_date=period.end,
    #                 i_fixings=self.index_fixings,
    #             )
    #     elif isinstance(self.index_fixings, list):
    #         for i in range(len(self.index_fixings)):
    #             self.periods[i].index_fixings = self.index_fixings[i]
    #     else:  # index_fixings is float
    #         if type(self) is ZeroFixedLeg:
    #             self.periods[0].index_fixings = self.index_fixings
    #             self.periods[1].index_fixings = self.index_fixings
    #         elif type(self) is IndexFixedLegExchange and self.inital_exchange is False:
    #             self.periods[0].index_fixings = self.index_fixings
    #         else:
    #             self.periods[0].index_fixings = self.index_fixings
    #         # TODO index_fixings as a list cannot handle amortization. Use a Series.

    @property
    def index_fixings(self) -> DualTypes | list[DualTypes] | Series[DualTypes] | NoInput:  # type: ignore[type-var]
        return self._index_fixings

    @index_fixings.setter
    def index_fixings(
        self,
        value: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput,  # type: ignore[type-var]
    ) -> None:
        self._index_fixings: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput = value  # type: ignore[type-var]

        def _index_from_series(ser: Series[DualTypes], end: datetime) -> DualTypes | NoInput:  # type: ignore[type-var]
            val: DualTypes | None = IndexMixin._index_value(
                i_fixings=ser,
                i_method=self.index_method,
                i_lag=self.index_lag,
                i_date=end,
                i_curve=NoInput(0),  # ! NoInput returned for periods beyond Series end.
            )
            if val is None:
                _: DualTypes | NoInput = NoInput(0)
            else:
                _ = val
            return _

        def _index_from_list(ls: list[DualTypes], i: int) -> DualTypes | NoInput:
            return NoInput(0) if i >= len(ls) else ls[i]

        if isinstance(value, NoInput):
            for p in [_ for _ in self.periods if type(_) is not Cashflow]:
                p.index_fixings = NoInput(0)
        elif isinstance(value, Series):
            for p in [_ for _ in self.periods if type(_) is not Cashflow]:
                date_: datetime = p.end if type(p) is IndexFixedPeriod else p.payment
                p.index_fixings = _index_from_series(value, date_)
        elif isinstance(value, list):
            for i, p in enumerate([_ for _ in self.periods if type(_) is not Cashflow]):
                p.index_fixings = _index_from_list(value, i)
        else:
            self.periods[0].index_fixings = value  # type: ignore[union-attr]
            for p in [_ for _ in self.periods[1:] if type(_) is not Cashflow]:
                p.index_fixings = NoInput(0)

    @property
    def index_base(self) -> DualTypes | NoInput:
        return self._index_base

    @index_base.setter
    def index_base(self, value: DualTypes | Series[DualTypes] | NoInput) -> None:  # type: ignore[type-var]
        if isinstance(value, Series):
            _: DualTypes | None = IndexMixin._index_value(
                i_fixings=value,
                i_method=self.index_method,
                i_lag=self.index_lag,
                i_date=self.schedule.effective,
                i_curve=NoInput(0),  # not required because i_fixings is Series
            )
            if _ is None:
                ret: DualTypes | NoInput = NoInput(0)
            else:
                ret = _
        else:
            ret = value
        self._index_base = ret
        # if value is not None:
        for period in self.periods:
            if isinstance(period, IndexFixedPeriod | IndexCashflow):
                period.index_base = self._index_base

    # def _regular_period(self, *args: Any, **kwargs: Any) -> Period:  # type: ignore[empty-body]
    #     pass  # pragma: no cover


class ZeroFloatLeg(_FloatLegMixin, BaseLeg):
    """
    Create a zero coupon floating leg composed of
    :class:`~rateslib.periods.FloatPeriod` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    float_spread : float, optional
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Applies only to
        rates within *Periods*. This does **not** apply to compounding of *Periods* within the
        *Leg*. Compounding of *Periods* is done using the ISDA compounding method. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a *ZeroFloatLeg* is:

    .. math::

       P = -N v(m_n) \\left ( \\prod_{i=1}^n (1 + d_i r_i(r_j, z)) - 1 \\right )

    The analytic delta of a *ZeroFloatLeg* is:

    .. math::

      A = N v(m_n) \\sum_{k=1}^n d_k \\frac{\\partial r_k}{\\partial z} \\prod_{i=1, i \\ne k}^n (1 + d_i r_i(r_j, z))

    .. warning::

       When floating rates are determined from historical fixings the forecast
       ``Curve`` ``calendar`` will be used to determine fixing dates.
       If this calendar does not align with the leg ``calendar`` then
       spurious results or errors may be generated. Including the curve calendar in
       the leg is acceptable, i.e. a leg calendar of *"nyc,ldn,tgt"* and a curve
       calendar of *"ldn"* is valid, whereas only *"nyc,tgt"* may give errors.

    Examples
    --------
    .. ipython:: python

       zfl = ZeroFloatLeg(
           effective=dt(2022, 1, 1),
           termination="3Y",
           frequency="S",
           fixing_method="ibor",
           method_param=0,
           float_spread=100.0
       )
       zfl.cashflows(curve)
    """  # noqa: E501

    _delay_set_periods: bool = True
    _regular_periods: tuple[FloatPeriod, ...]

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes | NoInput = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)
        if self.schedule.frequency == "Z":
            raise ValueError(
                "`frequency` for a ZeroFloatLeg should not be 'Z'. The Leg is zero frequency by "
                "construction. Set the `frequency` equal to the compounding frequency of the "
                "expressed fixed rate, e.g. 'S' for semi-annual compounding.",
            )
        if abs(_dual_float(self.amortization)) > 1e-8:
            raise ValueError("`ZeroFloatLeg` cannot be defined with `amortization`.")
        if self.initial_exchange or self.final_exchange:
            raise ValueError("`initial_exchange` or `final_exchange` not allowed on ZeroFloatLeg.")
        self._set_fixings(fixings)
        self._set_periods()

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> FloatPeriod:
        return super()._regular_period(
            start=start,
            end=end,
            payment=self.schedule.pschedule[-1],
            notional=notional,
            stub=stub,
            iterator=iterator,
        )

    def _set_periods(self) -> None:
        return super(_FloatLegMixin, self)._set_periods()

    @property
    def dcf(self) -> float:
        _ = [period.dcf for period in self.periods if isinstance(period, FloatPeriod)]
        return sum(_)

    def rate(self, curve: CurveOption_) -> DualTypes:
        """
        Calculate a simple period type floating rate for the zero coupon leg.

        Parameters
        ----------
        curve : Curve, LineCurve
            The forecasting curve object.

        Returns
        -------
        float, Dual, Dual2
        """
        rates = (
            (1.0 + p.dcf * p.rate(curve) / 100) for p in self.periods if isinstance(p, FloatPeriod)
        )
        compounded_rate: DualTypes = prod(rates)  # type: ignore[arg-type]
        return 100 * (compounded_rate - 1.0) / self.dcf

    def npv(
        self,
        curve: CurveOption_,
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> dict[str, DualTypes] | DualTypes:
        """
        Return the NPV of the *ZeroFloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = (
            self.rate(curve)
            / 100
            * self.dcf
            * disc_curve_[self.schedule.pschedule[-1]]
            * -self.notional
        )
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def fixings_table(
        self,
        curve: CurveOption_,
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.ZeroFloatLeg`.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given and ``curve`` is discount factor based.
        fx : float, FXRates, FXForwards, optional
            Only used in the case of :class:`~rateslib.legs.FloatLegMtm` to derive FX fixings.
        base : str, optional
            Not used by ``fixings_table``.
        approximate: bool
            Whether to use a faster (3x) but marginally less accurate (0.1% error) calculation.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)

        if self.fixing_method == "ibor":
            dfs = []
            prod = 1 + self.dcf * self.rate(curve) / 100.0
            prod *= -self.notional * disc_curve_[self.schedule.pschedule[-1]]
            for period in self.periods:
                if not isinstance(period, FloatPeriod):
                    continue
                scalar = period.dcf / (1 + period.dcf * period.rate(curve) / 100.0)
                risk = prod * scalar
                dfs.append(period._ibor_fixings_table(curve, disc_curve_, right, risk))
        else:
            dfs = []
            prod = 1 + self.dcf * self.rate(curve) / 100.0
            for period in [_ for _ in self.periods if isinstance(_, FloatPeriod)]:
                # TODO: handle interpolated fixings and curve as dict.
                df = period.fixings_table(curve, approximate, disc_curve_)
                scalar = prod / (1 + period.dcf * period.rate(curve) / 100.0)
                df[(curve.id, "risk")] *= scalar  # type: ignore[operator, union-attr]
                df[(curve.id, "notional")] *= scalar  # type: ignore[operator, union-attr]
                dfs.append(df)

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            return pd.concat(dfs)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *ZeroFloatLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx_, base = _get_fx_and_base(self.currency, fx, base)

        float_periods: list[FloatPeriod] = [_ for _ in self.periods if isinstance(_, FloatPeriod)]
        rates = ((1 + p.dcf * p.rate(curve) / 100) for p in float_periods)
        compounded_rate: DualTypes = prod(rates)  # type: ignore[arg-type]

        a_sum: DualTypes = 0.0
        for period in float_periods:
            _ = period.analytic_delta(curve, disc_curve_, fx_, base) / disc_curve_[period.payment]
            _ *= compounded_rate / (1 + period.dcf * period.rate(curve) / 100)
            a_sum += _
        a_sum *= disc_curve_[self.schedule.pschedule[-1]] * fx_
        return a_sum

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of the *ZeroFloatLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if isinstance(curve, NoInput):
            rate, cashflow = None, None
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            rate = _dual_float(self.rate(curve))
            cashflow = -_dual_float(self.notional * self.dcf * rate / 100)
            if not isinstance(disc_curve_, NoInput):
                npv = _dual_float(self.npv(curve, disc_curve_))  # type: ignore[arg-type]
                npv_fx = npv * _dual_float(fx)
                df = _dual_float(disc_curve_[self.schedule.pschedule[-1]])
                collateral = disc_curve_.collateral
            else:
                npv, npv_fx, df, collateral = None, None, None, None

        spread = 0.0 if isinstance(self.float_spread, NoInput) else _dual_float(self.float_spread)
        seq = [
            {
                defaults.headers["type"]: type(self).__name__,
                defaults.headers["stub_type"]: None,
                defaults.headers["currency"]: self.currency.upper(),
                defaults.headers["a_acc_start"]: self.schedule.aschedule[0],
                defaults.headers["a_acc_end"]: self.schedule.aschedule[-1],
                defaults.headers["payment"]: self.schedule.pschedule[-1],
                defaults.headers["convention"]: self.convention,
                defaults.headers["dcf"]: self.dcf,
                defaults.headers["notional"]: _dual_float(self.notional),
                defaults.headers["df"]: df,
                defaults.headers["rate"]: rate,
                defaults.headers["spread"]: spread,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: _dual_float(fx),
                defaults.headers["npv_fx"]: npv_fx,
                defaults.headers["collateral"]: collateral,
            },
        ]
        return DataFrame.from_records(seq)


class ZeroFixedLeg(_FixedLegMixin, BaseLeg):  # type: ignore[misc]
    """
    Create a zero coupon fixed leg composed of a single
    :class:`~rateslib.periods.FixedPeriod` .

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The IRR rate applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market rate for all periods has been calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    .. warning::

       The ``fixed_rate`` in this calculation is not a period rate but an IRR
       defining the cashflow as follows,

       .. math::

          C = -N \\left ( \\left (1 + \\frac{R^{irr}}{f} \\right ) ^ {df} - 1 \\right )

    The NPV of a *ZeroFixedLeg* is:

    .. math::

       P = -N v(m) \\left ( \\left (1+\\frac{R^{irr}}{f} \\right )^{df} - 1 \\right )

    The analytic delta of a *ZeroFixedLeg* is:

    .. math::

      A = N d v(m) \\left ( 1+ \\frac{R^{irr}}{f} \\right )^{df -1}

    Examples
    --------
    .. ipython:: python

       zfl = ZeroFixedLeg(
           effective=dt(2022, 1, 1),
           termination="3Y",
           frequency="S",
           convention="1+",
           fixed_rate=5.0
       )
       zfl.cashflows(curve)

    """

    periods: list[FixedPeriod]  # type: ignore[assignment]

    def __init__(
        self, *args: Any, fixed_rate: DualTypes | NoInput = NoInput(0), **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_rate = fixed_rate
        if self.schedule.frequency == "Z":
            raise ValueError(
                "`frequency` for a ZeroFixedLeg should not be 'Z'. The Leg is zero frequency by "
                "construction. Set the `frequency` equal to the compounding frequency of the "
                "expressed fixed rate, e.g. 'S' for semi-annual compounding.",
            )
        if abs(_dual_float(self.amortization)) > 1e-8:
            raise ValueError("`ZeroFixedLeg` cannot be defined with `amortization`.")

    def _set_periods(self) -> None:
        self.periods = [
            FixedPeriod(
                fixed_rate=NoInput(0),
                start=self.schedule.effective,
                end=self.schedule.termination,
                payment=self.schedule.pschedule[-1],
                notional=self.notional,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                frequency=self.schedule.frequency,
                stub=False,
                roll=self.schedule.roll,
                calendar=self.schedule.calendar,
            ),
        ]

    @property
    def fixed_rate(self) -> DualTypes | NoInput:
        """
        float or None : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes | NoInput) -> None:
        # overload the setter for a zero coupon to convert from IRR to period rate.
        # the headline fixed_rate is the IRR rate but the rate attached to Periods is a simple
        # rate in order to determine cashflows according to the normal cashflow logic.
        self._fixed_rate = value
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        if not isinstance(value, NoInput):
            period_rate = 100 * (1 / self.dcf) * ((1 + value / (100 * f)) ** (self.dcf * f) - 1)
        else:
            period_rate = NoInput(0)

        for period in self.periods:
            if isinstance(period, FixedPeriod):  # there should only be one FixedPeriod in a Zero
                period.fixed_rate = period_rate

    @property
    def dcf(self) -> float:
        """
        The DCF of a *ZeroFixedLeg* is defined as DCF of the single *FixedPeriod*
        spanning the *Leg*.
        """
        _ = [period.dcf for period in self.periods]
        return sum(_)

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the cashflows of the *ZeroFixedLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx_, base = _get_fx_and_base(self.currency, fx, base)
        rate = self.fixed_rate
        cashflow = self.periods[0].cashflow

        if isinstance(disc_curve_, NoInput) or isinstance(rate, NoInput):
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            npv = _dual_float(self.npv(curve, disc_curve_))  # type: ignore[arg-type]
            npv_fx = npv * _dual_float(fx_)
            df = _dual_float(disc_curve_[self.schedule.pschedule[-1]])
            collateral = disc_curve_.collateral

        seq = [
            {
                defaults.headers["type"]: type(self).__name__,
                defaults.headers["stub_type"]: None,
                defaults.headers["currency"]: self.currency.upper(),
                defaults.headers["a_acc_start"]: self.schedule.aschedule[0],
                defaults.headers["a_acc_end"]: self.schedule.aschedule[-1],
                defaults.headers["payment"]: self.schedule.pschedule[-1],
                defaults.headers["convention"]: self.convention,
                defaults.headers["dcf"]: self.dcf,
                defaults.headers["notional"]: _dual_float(self.notional),
                defaults.headers["df"]: df,
                defaults.headers["rate"]: self.fixed_rate,
                defaults.headers["spread"]: None,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: _dual_float(fx_),
                defaults.headers["npv_fx"]: npv_fx,
                defaults.headers["collateral"]: collateral,
            },
        ]
        return DataFrame.from_records(seq)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *ZeroFixedLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        if isinstance(self.fixed_rate, NoInput):
            raise ValueError("Must have `fixed_rate` on ZeroFixedLeg for analytic delta.")

        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _: DualTypes = self.notional * self.dcf * disc_curve_[self.periods[0].payment]
        _ *= (1 + self.fixed_rate / (100 * f)) ** (self.dcf * f - 1)
        return _ / 10000 * fx

    def _analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Analytic delta based on period rate and not IRR.
        """
        _ = [period.analytic_delta(*args, **kwargs) for period in self.periods]
        return sum(_)

    def _spread(
        self,
        target_npv: DualTypes,
        fore_curve: CurveOption_,
        disc_curve: CurveOption_,
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        """
        Overload the _spread calc to use analytic delta based on period rate
        """
        a_delta = self._analytic_delta(fore_curve, disc_curve, fx, self.currency)
        period_rate = -target_npv / (a_delta * 100)
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _: DualTypes = f * ((1 + period_rate * self.dcf / 100) ** (1 / (self.dcf * f)) - 1)
        return _ * 10000

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *ZeroFixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)


class ZeroIndexLeg(_IndexLegMixin, BaseLeg):
    """
    Create a zero coupon index leg composed of a single
    :class:`~rateslib.periods.IndexFixedPeriod` and
    a :class:`~rateslib.periods.Cashflow`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    index_base : float or None, optional
        The base index applied to all periods.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the first
        period.
        If a list of *n* fixings will be used as the index fixings for the first *n*
        periods.
        If a datetime indexed ``Series`` will use the fixings that are available in
        that object, and derive the rest from the ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    .. warning::

       Setting ``convention`` for a *ZeroIndexLeg* has no effect because the determination of the
       cashflow within the *IndexFixedPeriod* will always have a DCF of 1.0.

    The fixed rate of the *IndexFixedPeriod* is set to 100% to index up the
    complete the notional. The offsetting *Cashflow* deducts the real notional.

    The NPV of a *ZeroIndexLeg* is the sum of the period NPVs.

    .. math::

       P = - v(m_n) N \\left ( I(m_n) - 1 \\right )

    The analytic delta is defined as zero due to the lack of rates related attributes.

    .. math::

       A = 0

    Examples
    --------
    .. ipython:: python

       index_curve = Curve({dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.95}, index_base=100.0)
       zil = ZeroIndexLeg(
           effective=dt(2022, 1, 15),
           termination="3Y",
           frequency="S",
           index_method="monthly",
           index_base=100.25,
       )
       zil.cashflows(index_curve, curve)

    """

    periods: list[IndexFixedPeriod | Cashflow]  # type: ignore[assignment]

    def __init__(
        self,
        *args: Any,
        index_base: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_fixings: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.index_method = _drb(defaults.index_method, index_method).lower()
        self.index_lag = _drb(defaults.index_lag, index_lag)
        super().__init__(*args, **kwargs)
        self.index_fixings = index_fixings  # set index fixings after periods init
        # set after periods initialised
        self.index_base = index_base  # type: ignore[assignment]

    def _regular_period(  # type: ignore[empty-body]
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> IndexFixedPeriod:
        # set_periods has override
        pass

    def _set_periods(self) -> None:
        self.periods = [
            IndexFixedPeriod(
                fixed_rate=100.0,
                start=self.schedule.effective,
                end=self.schedule.termination,
                payment=self.schedule.pschedule[-1],
                convention="1",
                frequency=self.schedule.frequency,
                notional=self.notional,
                currency=self.currency,
                termination=self.schedule.termination,
                stub=False,
                roll=self.schedule.roll,
                calendar=self.schedule.calendar,
                index_base=self.index_base,
                index_fixings=NoInput(0),  # set during init
                index_lag=self.index_lag,
                index_method=self.index_method,
            ),
            Cashflow(
                notional=-self.notional,
                payment=self.schedule.pschedule[-1],
                currency=self.currency,
                stub_type=NoInput(0),
                rate=NoInput(0),
            ),
        ]

    def cashflow(self, curve: Curve | NoInput = NoInput(0)) -> DualTypes:
        """Aggregate the cashflows on the *IndexFixedPeriod* and *Cashflow* period using a
        *Curve*."""
        _: DualTypes = self.periods[0].cashflow(curve) + self.periods[1].cashflow  # type: ignore[operator]
        return _

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *ZeroIndexLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        cfs = super().cashflows(*args, **kwargs)
        _: DataFrame = cfs.iloc[[0]].copy()
        for attr in ["Cashflow", "NPV", "NPV Ccy"]:
            _[attr] += cfs.iloc[1][attr]
        _["Type"] = "ZeroIndexLeg"
        _["Period"] = None
        return _

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return 0.0

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)


class IndexFixedLeg(_IndexLegMixin, _FixedLegMixin, BaseLeg):  # type: ignore[misc]
    """
    Create a leg of :class:`~rateslib.periods.IndexFixedPeriod` s and initial and
    final :class:`~rateslib.periods.IndexCashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    index_base : float or None, optional
        The base index to determine the cashflow.
    index_fixings : float, list or Series, optional
        If a float scalar, will be applied as the index fixing for the first period.
        If a datetime indexed ``Series``, will use the fixings that are available
        in that object for relevant periods, and derive the rest from the ``curve``.
        If a list, will apply those values as the fixings for the first set of periods
        and derive the rest from the ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       An initial exchange is not currently implemented for this leg.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accrual date adjusted by ``payment_lag_exchange``.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods. For similar examples see
    :class:`~rateslib.legs.FloatLeg`.

    The NPV of a *IndexFixedLeg* is the sum of the period NPVs.

    .. math::

       P = - R \\sum_{i=1}^n N_i d_i v(m_i) I(m_i) - \\sum_{i=1}^{n-1}(N_{i}-N_{i+1})v(m_i)I(m_i)  - N_n v(m_n)I(m_n)

    The analytic delta is defined as that of a *FixedLeg*.

    .. math::

       A = \\sum_{i=1}^n N_i d_i v(m_i) I(m_i)

    Examples
    --------

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       index_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_base=100.0)
       index_leg_exch = IndexFixedLeg(
           dt(2022, 1, 1), "9M", "Q",
           notional=1000000,
           amortization=200000,
           index_base=100.0,
           initial_exchange=False,
           final_exchange=True,
           fixed_rate=1.0,
       )
       index_leg_exch.cashflows(index_curve, curve)
       index_leg_exch.npv(index_curve, curve)

    """  # noqa: E501

    periods: list[IndexCashflow | IndexFixedPeriod]  # type: ignore[assignment]
    _regular_periods: tuple[IndexFixedPeriod, ...]

    # TODO: spread calculations to determine the fixed rate on this leg do not work.
    def __init__(
        self,
        *args: Any,
        index_base: DualTypes,
        index_fixings: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        fixed_rate: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._fixed_rate = fixed_rate
        self.index_lag: int = _drb(defaults.index_lag, index_lag)
        self.index_method = _drb(defaults.index_method, index_method).lower()
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super().__init__(*args, **kwargs)
        self.index_fixings = index_fixings  # set index fixings after periods init
        self.index_base = index_base  # set after periods initialised

    def _set_exchange_periods(self) -> None:
        """Set default cashflow exchanges on Legs with `initial_exchange` or `final_exchange`."""

        periods_: list[IndexCashflow | None] = [None, None]

        # initial exchange
        if self.initial_exchange:
            raise NotImplementedError(
                "Cannot construct `IndexFixedLeg` with `initial_exchange` "
                "due to not implemented `index_fixings` input argument applicable to "
                "the indexing-up the initial exchange.",
            )
            # self.periods.append(
            #     IndexCashflow(
            #         notional=-self.notional,
            #         payment=add_tenor(
            #             self.schedule.aschedule[0],
            #             f"{self.payment_lag_exchange}B",
            #             None,
            #             self.schedule.calendar,
            #         ),
            #         currency=self.currency,
            #         stub_type="Exchange",
            #         rate=None,
            #         index_base=self.index_base,
            #         index_fixings=self.index_fixings,
            #         index_method=self.index_method,
            #     )
            # )

        # final cashflow
        if self.final_exchange:
            periods_[1] = IndexCashflow(
                notional=self.notional - self.amortization * (self.schedule.n_periods - 1),
                payment=self.schedule.calendar.lag(
                    self.schedule.aschedule[-1],
                    self.payment_lag_exchange,
                    True,
                ),
                currency=self.currency,
                stub_type="Exchange",
                rate=NoInput(0),
                index_base=self.index_base,
                index_fixings=self.index_fixings[-1]
                if isinstance(self.index_fixings, list)
                else self.index_fixings,
                index_method=self.index_method,
            )

        self._exchange_periods: tuple[IndexCashflow | None, IndexCashflow | None] = tuple(periods_)  # type: ignore[assignment]

    def _set_interim_exchange_periods(self) -> None:
        """Set cashflow exchanges if `amortization` and `final_exchange` are present."""
        if not self.final_exchange or self.amortization == 0:
            self._interim_exchange_periods: tuple[IndexCashflow, ...] | None = None
        else:
            periods_ = [
                IndexCashflow(
                    notional=self.amortization,
                    payment=self.schedule.pschedule[1 + i],
                    currency=self.currency,
                    stub_type="Amortization",
                    rate=NoInput(0),
                    index_base=self.index_base,
                    index_fixings=self.index_fixings[i]
                    if isinstance(self.index_fixings, list)
                    else self.index_fixings,
                    index_method=self.index_method,
                )
                for i in range(self.schedule.n_periods - 1)
            ]
            self._interim_exchange_periods = tuple(periods_)

    def _set_regular_periods(self) -> None:
        self._regular_periods: tuple[IndexFixedPeriod, ...] = tuple(
            [
                IndexFixedPeriod(
                    fixed_rate=self.fixed_rate,
                    start=period[defaults.headers["a_acc_start"]],
                    end=period[defaults.headers["a_acc_end"]],
                    payment=period[defaults.headers["payment"]],
                    notional=self.notional - self.amortization * i,
                    convention=self.convention,
                    currency=self.currency,
                    termination=self.schedule.termination,
                    frequency=self.schedule.frequency,
                    stub=period[defaults.headers["stub_type"]] == "Stub",
                    roll=self.schedule.roll,
                    calendar=self.schedule.calendar,
                    index_base=self.index_base,
                    index_method=self.index_method,
                    index_fixings=self.index_fixings[i]
                    if isinstance(self.index_fixings, list)
                    else self.index_fixings,
                )
                for i, period in enumerate(self.schedule.table.to_dict(orient="index").values())
            ]
        )

    def _set_periods(self) -> None:
        return super(_FixedLegMixin, self)._set_periods()

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        return super().npv(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        return super().cashflows(*args, **kwargs)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        return super().analytic_delta(*args, **kwargs)


class CreditPremiumLeg(_FixedLegMixin, BaseLeg):
    """
    Create a credit premium leg composed of :class:`~rateslib.periods.CreditPremiumPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The credit spread applied to determine cashflows in percentage points (i.e 50bps = 0.50).
        Can be left unset and
        designated later, perhaps after a mid-market rate for all periods has been calculated.
    premium_accrued : bool, optional
        Whether the premium is accrued within the period to default.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a credit premium leg is the sum of the period NPVs.

    .. math::

       P = \\sum_{i=1}^n P_i

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial S} = \\sum_{i=1}^n -\\frac{\\partial P_i}{\\partial S}

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.curves import Curve
       from rateslib.legs import CreditPremiumLeg
       from datetime import datetime as dt

    .. ipython:: python

       disc_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       hazard_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
       premium_leg = CreditPremiumLeg(
           dt(2022, 1, 1), "9M", "Q",
           fixed_rate=2.60,
           notional=1000000,
       )
       premium_leg.cashflows(hazard_curve, disc_curve)
       premium_leg.npv(hazard_curve, disc_curve)
    """  # noqa: E501

    periods: list[CreditPremiumPeriod]  # type: ignore[assignment]

    _regular_periods: tuple[CreditPremiumPeriod, ...]

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes | NoInput = NoInput(0),
        premium_accrued: bool | NoInput = NoInput(0),
        **kwargs: Any,
    ):
        self._fixed_rate = fixed_rate
        self.premium_accrued = _drb(defaults.cds_premium_accrued, premium_accrued)
        super().__init__(*args, **kwargs)
        if self.initial_exchange or self.final_exchange:
            raise ValueError(
                "`initial_exchange` and `final_exchange` cannot be True on CreditPremiumLeg."
            )
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *CreditPremiumLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *CreditPremiumLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditPremiumLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def accrued(self, settlement: datetime) -> DualTypes | None:
        """
        Calculate the amount of premium accrued until a specific date within the relevant *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float
        """
        _ = index_left(
            self.schedule.uschedule,
            len(self.schedule.uschedule),
            settlement,
        )
        # This index is valid because this Leg only contains CreditPremiumPeriods and no exchanges.
        return self.periods[_].accrued(settlement)

    def _set_periods(self) -> None:
        return super()._set_periods()

    def _regular_period(  # type: ignore[override]
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        notional: DualTypes,
        stub: bool,
        iterator: int,
    ) -> CreditPremiumPeriod:
        return CreditPremiumPeriod(
            fixed_rate=self.fixed_rate,
            premium_accrued=self.premium_accrued,
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


class CreditProtectionLeg(BaseLeg):
    """
    Create a credit protection leg composed of :class:`~rateslib.periods.CreditProtectionPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    recovery_rate : float, Dual, Dual2, optional
        The assumed recovery rate that defines payment on credit default. Set by ``defaults``.
    discretization : int, optional
        The number of days to discretize the numerical integration over possible credit defaults.
        Set by ``defaults``.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a credit protection leg is the sum of the period NPVs.

    .. math::

       P = \\sum_{i=1}^n P_i

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial S} = \\sum_{i=1}^n -\\frac{\\partial P_i}{\\partial S}

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.curves import Curve
       from rateslib.legs import CreditProtectionLeg
       from datetime import datetime as dt

    .. ipython:: python

       disc_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       hazard_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995})
       protection_leg = CreditProtectionLeg(
           dt(2022, 1, 1), "9M", "Z",
           recovery_rate=0.40,
           notional=1000000,
       )
       protection_leg.cashflows(hazard_curve, disc_curve)
       protection_leg.npv(hazard_curve, disc_curve)
    """  # noqa: E501

    periods: list[CreditProtectionPeriod]  # type: ignore[assignment]

    def __init__(
        self,
        *args: Any,
        recovery_rate: DualTypes | NoInput = NoInput(0),
        discretization: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._recovery_rate: DualTypes = _drb(defaults.cds_recovery_rate, recovery_rate)
        self.discretization: int = _drb(defaults.cds_protection_discretization, discretization)
        super().__init__(*args, **kwargs)
        if self.initial_exchange or self.final_exchange:
            raise ValueError(
                "`initial_exchange` and `final_exchange` cannot be True on CreditProtectionLeg."
            )
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def analytic_rec_risk(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic recovery risk of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        _ = (period.analytic_rec_risk(*args, **kwargs) for period in self.periods)
        return sum(_)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *CreditProtectionLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditProtectionLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def _set_periods(self) -> None:
        return super()._set_periods()

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> CreditProtectionPeriod:
        return CreditProtectionPeriod(
            recovery_rate=self.recovery_rate,
            discretization=self.discretization,
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

    @property
    def recovery_rate(self) -> DualTypes:
        return self._recovery_rate

    @recovery_rate.setter
    def recovery_rate(self, value: DualTypes) -> None:
        self._recovery_rate = value
        for _ in self.periods:
            if isinstance(_, CreditProtectionPeriod):
                _.recovery_rate = value


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class BaseLegMtm(BaseLeg, metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``LegMtm`` subclasses.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fx_fixings : float, Dual, Dual2 or list or Series of such
        Define the known FX fixings for each period which affects the mark-the-market
        (MTM) notional exchanges after each period. If not given, or only some
        FX fixings are given, the remaining unknown fixings will be forecast
        by a provided :class:`~rateslib.fx.FXForwards` object later. If a Series must be indexed
        by the date of the notional exchange considering ``payment_lag_exchange``.
    alt_currency : str
        The alternative reference currency against which FX fixings are measured
        for MTM notional exchanges (3-digit code).
    alt_notional : float
        The notional expressed in the alternative currency which will be used to
        determine the notional for this leg using the ``fx_fixings`` as FX rates.
    kwargs : dict
        Required keyword args to :class:`BaseLeg`.

    See Also
    --------
    FixedLegExchangeMtm: Create a fixed leg with notional and Mtm exchanges.
    FloatLegExchangeMtm : Create a floating leg with notional and Mtm exchanges.
    """

    _do_not_repeat_set_periods: bool = False
    _is_mtm: bool = True
    _delay_set_periods: bool = True

    def __init__(
        self,
        *args: Any,
        fx_fixings: NoInput  # type: ignore[type-var]
        | DualTypes
        | list[DualTypes]
        | Series[DualTypes]
        | tuple[DualTypes, Series[DualTypes]] = NoInput(0),
        alt_currency: str | NoInput = NoInput(0),
        alt_notional: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        if isinstance(alt_currency, NoInput):
            raise ValueError("`alt_currency` and `currency` must be supplied for MtmLeg.")
        self.alt_currency: str = alt_currency.lower()
        self.alt_notional: DualTypes = _drb(defaults.notional, alt_notional)
        if "initial_exchange" not in kwargs:
            kwargs["initial_exchange"] = True
        kwargs["final_exchange"] = True
        super().__init__(*args, **kwargs)
        if self.amortization != 0:
            raise ValueError("`amortization` cannot be supplied to a `FixedLegExchangeMtm` type.")

        # calls the fixings setter, will convert the input types to list
        self.fx_fixings = fx_fixings  # type: ignore[assignment]

    @property
    def notional(self) -> DualTypes:
        return self._notional

    @notional.setter
    def notional(self, value: DualTypes) -> None:
        self._notional = value

    def _get_fx_fixings_from_series(
        self,
        ser: Series[DualTypes],  # type: ignore[type-var]
        ini_period: int = 0,
    ) -> list[DualTypes]:
        last_fixing_date = ser.index[-1]
        fixings_list: list[DualTypes] = []
        for i in range(ini_period, self.schedule.n_periods):
            required_date = self.schedule.calendar.lag(
                self.schedule.aschedule[i], self.payment_lag_exchange, True
            )
            if required_date > last_fixing_date:
                break
            else:
                try:
                    fixings_list.append(ser[required_date])
                except KeyError:
                    raise ValueError(
                        "A Series is provided for FX fixings but the required exchange "
                        f"settlement date, {required_date.strftime('%Y-%d-%m')}, is not "
                        f"available within the Series.",
                    )
        return fixings_list

    @property
    def fx_fixings(self) -> list[DualTypes]:
        """
        list : FX fixing values input by user and attached to the instrument.
        """
        return self._fx_fixings

    @fx_fixings.setter
    def fx_fixings(self, value: FixingsFx_) -> None:  # type: ignore[type-var]
        """
        Parse a 'FixingsFx_' object to convert to a list[DualTypes] attached to _fx_fixings attr.
        """
        if isinstance(value, NoInput):
            self._fx_fixings: list[DualTypes] = []
        elif isinstance(value, list):
            self._fx_fixings = value
        elif isinstance(value, float | Dual | Dual2 | Variable):
            self._fx_fixings = [value]
        elif isinstance(value, Series):
            self._fx_fixings = self._get_fx_fixings_from_series(value)
        elif isinstance(value, tuple):
            self._fx_fixings = [value[0]]
            self._fx_fixings.extend(self._get_fx_fixings_from_series(value[1], ini_period=1))
        else:
            raise TypeError("`fx_fixings` should be scalar value, list or Series of such.")

        # if self._initialised:
        #     self._set_periods(None)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _get_fx_fixings(self, fx: FX_) -> list[DualTypes]:
        """
        Return the calculated FX fixings.

        Initialise with the fx fixings already provided statically.
        Use an FXForwards object to determine the additionally required fixings.
        If FXForwards object not available repeat the final given fixing.
        If no fixings are known default to 1.0.

        Parameters
        ----------
        fx : FXForwards, optional
            The object to derive FX fixings that are not otherwise given in
            ``fx_fixings``.
        """
        n_given, n_req = len(self.fx_fixings), self.schedule.n_periods
        fx_fixings_: list[DualTypes] = self.fx_fixings.copy()

        # Only FXForwards can correctly forecast rates. Other inputs may raise Errros or Warnings.
        if isinstance(fx, FXForwards):
            for i in range(n_given, n_req):
                fx_fixings_.append(
                    fx.rate(
                        self.alt_currency + self.currency,
                        self.schedule.calendar.lag(
                            self.schedule.aschedule[i],
                            self.payment_lag_exchange,
                            True,
                        ),
                    ),
                )
        elif n_req > 0:  # only check if unknown fixings are required
            if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                raise ValueError(
                    "`fx` is required when `fx_fixings` are not pre-set and "
                    "if rateslib option `no_fx_fixings_for_xcs` is set to "
                    "'raise'.\nFurther info: You are trying to value a mark-to-market "
                    "leg on a multi-currency derivative.\nThese require FX fixings and if "
                    "those are not given then an FXForwards object should be provided which "
                    "will calculate the relevant FX rates."
                )
            if n_given == 0:
                if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                    warnings.warn(
                        "Using 1.0 for FX, no `fx` or `fx_fixing` given and "
                        "the option `defaults.no_fx_fixings_for_xcs` is set to "
                        "'warn'.\nFurther info: You are trying to value a mark-to-market "
                        "leg on a multi-currency derivative.\nThese require FX fixings and if "
                        "those are not given then an FXForwards object should be provided which "
                        "will calculate the relevant FX rates.",
                        UserWarning,
                    )
                fx_fixings_ = [1.0] * n_req
            else:
                if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                    warnings.warn(
                        "Using final FX fixing given for missing periods, "
                        "rateslib option `no_fx_fixings_for_xcs` is set to "
                        "'warn'.\nFurther info: You are trying to value a mark-to-market "
                        "leg on a multi-currency derivative.\nThese require FX fixings and if "
                        "those are not given then an FXForwards object should be provided which "
                        "will calculate the relevant FX rates.",
                        UserWarning,
                    )
                fx_fixings_.extend([fx_fixings_[-1]] * (n_req - n_given))
        return fx_fixings_

    def _set_periods(self) -> None:
        raise NotImplementedError("Mtm Legs do not implement this. Look for _set_periods_mtm().")

    def _set_periods_mtm(self, fx: FX_) -> None:
        fx_fixings_: list[DualTypes] = self._get_fx_fixings(fx)
        self.notional = fx_fixings_[0] * self.alt_notional
        notionals = [self.alt_notional * fx_fixings_[i] for i in range(len(fx_fixings_))]

        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[0],
                        self.payment_lag_exchange,
                        True,
                    ),
                    self.currency,
                    "Exchange",
                    fx_fixings_[0],
                ),
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            self._regular_period(
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                stub=period[defaults.headers["stub_type"]] == "Stub",
                notional=notionals[i],
                iterator=i,
            )
            for i, period in enumerate(self.schedule.table.to_dict(orient="index").values())
        ]
        mtm_flows = [
            Cashflow(
                -notionals[i + 1] + notionals[i],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[i + 1],
                    self.payment_lag_exchange,
                    True,
                ),
                self.currency,
                "Mtm",
                fx_fixings_[i + 1],
            )
            for i in range(len(fx_fixings_) - 1)
        ]
        interleaved_periods = [
            val for pair in zip(regular_periods, mtm_flows, strict=False) for val in pair
        ]
        interleaved_periods.append(regular_periods[-1])
        self.periods.extend(interleaved_periods)

        # final cashflow
        self.periods.append(
            Cashflow(
                notionals[-1],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[-1],
                    self.payment_lag_exchange,
                    True,
                ),
                self.currency,
                "Exchange",
                fx_fixings_[-1],
            ),
        )

    def npv(
        self,
        curve: Curve,
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().npv(curve, disc_curve, fx, base, local)
        # self._is_set_periods_fx = False
        return ret

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().cashflows(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().analytic_delta(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret


class FixedLegMtm(_FixedLegMixin, BaseLegMtm):  # type: ignore[misc]
    """
    Create a leg of :class:`~rateslib.periods.FixedPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    fx_fixings : float, Dual, Dual2, list of such
        Specify a known initial FX fixing or a list of such for historical legs.
        Fixings that are not specified will be calculated at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    alt_currency : str
        The alternative currency against which mark-to-market fixings and payments
        are made. This is considered as the domestic currency in FX fixings.
    alt_notional : float, optional
        The notional of the alternative currency from which to calculate ``notional``
        under the determined ``fx_fixings``. If `None` sets a
        default for ``alt_notional``.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       ``amortization`` is currently **not implemented** for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

       ``currency`` and ``alt_currency`` are required in order to determine FX fixings
       from an :class:`~rateslib.fx.FXForwards` object at pricing time.

    Examples
    --------
    For an example see :ref:`Mtm Legs<mtm-legs>`.
    """

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._fixed_rate = fixed_rate
        super().__init__(
            *args,
            **kwargs,
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatLegMtm(_FloatLegMixin, BaseLegMtm):
    """
    Create a leg of :class:`~rateslib.periods.FloatPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    float_spread : float or None
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the **first**
        whole period of the leg. If a list of *n* items, each successive item will be
        passed to the ``fixing`` argument of the first *n* periods of the leg.
        A list within the list is accepted if it contains a set of RFR fixings that
        will be applied to any individual RFR period.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    fx_fixings : float, Dual, Dual2, list of such
        Specify a known initial FX fixing or a list of such for historical legs.
        Fixings that are not specified will be calculated at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    alt_currency : str
        The alternative currency against which mark-to-market fixings and payments
        are made. This is considered as the domestic currency in FX fixings.
    alt_notional : float, optional
        The notional of the alternative currency from which to calculate ``notional``
        under the determined ``fx_fixings``. If `None` sets a
        default for ``alt_notional``.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       ``amortization`` is currently **not implemented** for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

       ``currency`` and ``alt_currency`` are required in order to determine FX fixings
       from an :class:`~rateslib.fx.FXForwards` object at pricing time.

    Examples
    --------
    For an example see :ref:`Mtm Legs<mtm-legs>`.
    """

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(
            *args,
            **kwargs,
        )

        self._set_fixings(fixings)
        self.fx_fixings = self.fx_fixings  # sets fx_fixings and periods after initialising

    def fixings_table(
        self,
        curve: CurveOption_,
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLegMtm`.

        For arguments see
        :meth:`FloatLeg.fixings_table()<rateslib.legs.FloatLeg.fixings_table>`.
        """
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        return super()._fixings_table(
            curve=curve, disc_curve=disc_curve, approximate=approximate, right=right
        )


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


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


__all__ = [
    "CustomLeg",
    "BaseLeg",
    "BaseLegMtm",
    "FixedLeg",
    "IndexFixedLeg",
    "FloatLeg",
    "FixedLegMtm",
    "FloatLegMtm",
    "ZeroFixedLeg",
    "ZeroFloatLeg",
    "ZeroIndexLeg",
    "CreditPremiumLeg",
    "CreditProtectionLeg",
]
