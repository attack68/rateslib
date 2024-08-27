# This is a dependent of instruments.py

"""
.. ipython:: python
   :suppress:

   from rateslib.legs import *
   from rateslib.curves import Curve
   from datetime import datetime as dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
   )
"""

from __future__ import annotations

import abc
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime

import pandas as pd
from pandas import DataFrame, Series
from pandas.tseries.offsets import CustomBusinessDay

from rateslib import defaults
from rateslib.calendars import add_tenor
from rateslib.curves import Curve, IndexCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, DualTypes, gradient, set_order
from rateslib.fx import FXForwards, FXRates
from rateslib.periods import (
    Cashflow,
    FixedPeriod,
    FloatPeriod,
    IndexCashflow,
    IndexFixedPeriod,
    IndexMixin,
    _disc_from_curve,
    _disc_maybe_from_curve,
    _get_fx_and_base,
    _validate_float_args,
)
from rateslib.scheduling import Schedule

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

    _is_mtm = False

    @abc.abstractmethod
    def __init__(
        self,
        effective: datetime,
        termination: datetime | str,
        frequency: str,
        *,
        stub: str | NoInput = NoInput(0),
        front_stub: datetime | NoInput = NoInput(0),
        back_stub: datetime | NoInput = NoInput(0),
        roll: str | int | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CustomBusinessDay | str | NoInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        amortization: float | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        payment_lag_exchange: int | NoInput = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
    ):
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
        self.convention = defaults.convention if convention is NoInput.blank else convention
        self.currency = defaults.base_currency if currency is NoInput.blank else currency.lower()

        self.payment_lag_exchange = (
            defaults.payment_lag_exchange
            if payment_lag_exchange is NoInput.blank
            else payment_lag_exchange
        )
        self.initial_exchange = initial_exchange
        self.final_exchange = final_exchange

        self._notional = defaults.notional if notional is NoInput.blank else notional
        self._amortization = 0 if amortization is NoInput.blank else amortization
        if getattr(self, "_delay_set_periods", False):
            pass
        else:
            self._set_periods()

    @property
    def notional(self):
        return self._notional

    @notional.setter
    def notional(self, value):
        self._notional = value
        self._set_periods()

    @property
    def amortization(self):
        return self._amortization

    @amortization.setter
    def amortization(self, value):
        self._amortization = value
        self._set_periods()

    @abstractmethod
    def _set_periods(self) -> None:
        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[0], self.payment_lag_exchange, True
                    ),
                    self.currency,
                    "Exchange",
                )
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            self._regular_period(
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                notional=self.notional - self.amortization * i,
                iterator=i,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        if self.final_exchange and self.amortization != 0:
            amortization = [
                Cashflow(
                    self.amortization,
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[i + 1], self.payment_lag_exchange, True
                    ),
                    self.currency,
                    "Amortization",
                )
                for i in range(self.schedule.n_periods - 1)
            ]
            interleaved_periods = [
                val for pair in zip(regular_periods, amortization) for val in pair
            ]
            interleaved_periods.append(regular_periods[-1])  # add last regular period
        else:
            interleaved_periods = regular_periods
        self.periods.extend(interleaved_periods)

        # final cashflow
        if self.final_exchange:
            self.periods.append(
                Cashflow(
                    self.notional - self.amortization * (self.schedule.n_periods - 1),
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[-1], self.payment_lag_exchange, True
                    ),
                    self.currency,
                    "Exchange",
                )
            )

    # @abstractmethod
    # def _regular_period(self):
    #     pass  # pragma: no cover

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *Leg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        _ = (period.analytic_delta(*args, **kwargs) for period in self.periods)
        return sum(_)

    def cashflows(self, *args, **kwargs) -> DataFrame:
        """
        Return the properties of the *Leg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        seq = [period.cashflows(*args, **kwargs) for period in self.periods]
        return DataFrame.from_records(seq)

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *Leg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        _is_local = (len(args) == 5 and args[4]) or kwargs.get("local", False)
        if _is_local:
            _ = (period.npv(*args, **kwargs)[self.currency] for period in self.periods)
            return {self.currency: sum(_)}
        else:
            _ = (period.npv(*args, **kwargs) for period in self.periods)
            return sum(_)

    @property
    def _is_linear(self):
        """
        Tests if analytic delta spread is a linear function affecting NPV.

        This is non-linear if the spread is itself compounded, which only occurs
        on RFR trades with *"isda_compounding"* or *"isda_flat_compounding"*, which
        should typically be avoided anyway.

        Returns
        -------
        bool
        """
        if "Float" in type(self).__name__:
            if "rfr" in self.fixing_method and self.spread_compound_method != "none_simple":
                return False
        return True

    def _spread_isda_approximated_rate(self, target_npv, fore_curve, disc_curve):
        """
        Use approximated derivatives through geometric averaged 1day rates to derive the
        spread
        """
        a, b = 0.0, 0.0
        for period in self.periods:
            try:
                a_, b_ = period._get_analytic_delta_quadratic_coeffs(fore_curve, disc_curve)
                a += a_
                b += b_
            except AttributeError:  # the period might be of wrong kind: TODO: better filter
                pass
        c = -target_npv

        # perform the quadratic solution
        _1 = -c / b
        if abs(a) > 1e-14:
            _2a = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
            _2b = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)  # alt quadratic soln
            if abs(_1 - _2a) < abs(_1 - _2b):
                _ = _2a
            else:
                _ = _2b  # select quadratic soln
        else:
            # this is to avoid divide by zero errors and return an approximation
            # also isda_flat_compounding has a=0
            _ = _1

        return _

    def _spread_isda_dual2(
        self, target_npv, fore_curve, disc_curve, fx=NoInput(0)
    ):  # pragma: no cover
        # This method is unused and untested, superseded by _spread_isda_approx_rate

        # This method creates a dual2 variable for float spread + obtains derivatives automatically
        _fs = self.float_spread
        self.float_spread = Dual2(0.0 if _fs is None else float(_fs), "spread_z")

        # This method uses ad-hoc AD to solve a specific problem for which
        # there is no closed form solution. Calculating NPV is very inefficient
        # so, we only do this once as opposed to using a root solver algo
        # which would otherwise converge to the exact solution but is
        # practically not workable.

        # This method is more accurate than the 'spread through approximated
        # derivatives' method, but it is a more costly and less robust method
        # due to its need to work in second order mode.

        fore_ad = fore_curve.ad
        fore_curve._set_ad_order(2)

        disc_ad = disc_curve.ad
        disc_curve._set_ad_order(2)

        if isinstance(fx, (FXRates, FXForwards)):
            _fx = None if fx is None else fx._ad
            fx._set_ad_order(2)

        npv = self.npv(fore_curve, disc_curve, fx, self.currency)
        b = gradient(npv, "spread_z", order=1)[0]
        a = 0.5 * gradient(npv, "spread_z", order=2)[0][0]
        c = -target_npv

        # Perform quadratic solution
        _1 = -c / b
        if abs(a) > 1e-14:
            _2a = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
            _2b = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)  # alt quadratic soln
            if abs(_1 - _2a) < abs(_1 - _2b):
                _ = _2a
            else:
                _ = _2b  # select quadratic soln
        else:  # pragma: no cover
            # this is to avoid divide by zero errors and return an approximation
            _ = _1
            warnings.warn(
                "Divide by zero encountered and the spread is approximated to " "first order.",
                UserWarning,
            )

        # This is required by the Dual2 AD approach to revert to original order.
        self.float_spread = _fs
        fore_curve._set_ad_order(fore_ad)
        disc_curve._set_ad_order(disc_ad)
        if isinstance(fx, (FXRates, FXForwards)):
            fx._set_ad_order(_fx)
        _ = set_order(_, disc_ad)  # use disc_ad: credit spread from disc curve

        return _

    def _spread(self, target_npv, fore_curve, disc_curve, fx=NoInput(0)):
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
            a_delta = self.analytic_delta(fore_curve, disc_curve, fx, self.currency)
            return -target_npv / a_delta
        else:
            _ = self._spread_isda_approximated_rate(target_npv, fore_curve, disc_curve)
            # _ = self._spread_isda_dual2(target_npv, fore_curve, disc_curve, fx)
            return _


class FixedLegMixin:
    """
    Add the functionality to add and retrieve ``fixed_rate`` on
    :class:`~rateslib.periods.FixedPeriod` s.
    """

    @property
    def fixed_rate(self):
        """
        float or NoInput : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        self._fixed_rate = value
        for period in getattr(self, "periods", []):
            if isinstance(period, FixedPeriod):
                period.fixed_rate = value

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        notional: float,
        stub: bool,
        iterator: int,
    ):
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


class FixedLeg(BaseLeg, FixedLegMixin):
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

    def __init__(self, *args, fixed_rate: float | NoInput = NoInput(0), **kwargs):
        self._fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)
        self._set_periods()

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *FixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args, **kwargs) -> DataFrame:
        """
        Return the properties of the *FixedLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args, **kwargs):
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


class FloatLegMixin:
    """
    Add the functionality to add and retrieve ``float_spread`` on
    :class:`~rateslib.periods.FloatPeriod` s and a
    :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
    """

    def _get_fixings_from_series(self, ser: Series, ini_period: int = 0) -> list:
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
                None,
                self.schedule.calendar,
            )
            for i in range(ini_period, self.schedule.n_periods)
        ]
        return [ser if last_fixing_dt >= day else NoInput(0) for day in first_required_day]

    def _set_fixings(
        self,
        fixings,
    ):
        """
        Re-organises the fixings input to list structure for each period.
        Requires a ``schedule`` object and ``float_args``.
        """
        if fixings is NoInput.blank:
            fixings_ = []
        elif isinstance(fixings, Series):
            fixings_ = fixings.sort_index()  # oldest fixing at index 0: latest -1
            fixings_ = self._get_fixings_from_series(fixings_)
        elif isinstance(fixings, tuple):
            fixings_ = [fixings[0]] + self._get_fixings_from_series(fixings[1], 1)
        elif not isinstance(fixings, list):
            fixings_ = [fixings]
        else:  # fixings as a list should be remaining
            fixings_ = fixings

        self.fixings = fixings_ + [NoInput(0)] * (self.schedule.n_periods - len(fixings_))

    @property
    def float_spread(self):
        """
        float or NoInput : If set will also set the ``float_spread`` of contained
            :class:`~rateslib.periods.FloatPeriod` s.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value):
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

    def _fixings_table(self, *args, **kwargs):
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        See :meth:`~rateslib.periods.FloatPeriod.fixings_table` for arguments.

        Returns
        -------
        DataFrame
        """
        df, counter = None, 0
        while df is None:
            if type(self.periods[counter]) is FloatPeriod:
                df = self.periods[counter].fixings_table(*args, **kwargs)
            counter += 1

        n = len(self.periods)
        for i in range(counter, n):
            if type(self.periods[i]) is FloatPeriod:
                df = pd.concat([df, self.periods[i].fixings_table(*args, **kwargs)])
        return df

    def _regular_period(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        notional: float,
        stub: bool,
        iterator: int,
    ):
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


class FloatLeg(BaseLeg, FloatLegMixin):
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

    def __init__(
        self,
        *args,
        float_spread: float | NoInput = NoInput(0),
        fixings: float | list | Series | tuple | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        self._delay_set_periods = True  # do this to set fixings first
        super().__init__(*args, **kwargs)
        self._set_fixings(fixings)
        self._set_periods()

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args, **kwargs) -> DataFrame:
        """
        Return the properties of the *FloatLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def fixings_table(self, *args, **kwargs) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        For arguments see
        :meth:`FloatPeriod.fixings_table()<rateslib.periods.FloatPeriod.fixings_table>`.
        """
        return super()._fixings_table(*args, **kwargs)

    def _set_periods(self) -> None:
        return super()._set_periods()

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


class IndexLegMixin:
    schedule = None
    index_method = None
    _index_fixings = None
    _index_base = None

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
    def index_fixings(self):
        return self._index_fixings

    @index_fixings.setter
    def index_fixings(self, value):
        self._index_fixings = value
        for i, period in enumerate(self.periods):
            if isinstance(period, (IndexFixedPeriod, IndexCashflow)):
                if isinstance(value, Series):
                    _ = IndexMixin._index_value(
                        i_fixings=value,
                        i_method=self.index_method,
                        i_lag=self.index_lag,
                        i_date=period.end,
                        i_curve=NoInput(0),  # ! NoInput returned for periods beyond Series end.
                    )
                elif isinstance(value, list):
                    if i >= len(value):
                        _ = NoInput(0)  # some fixings are unknown, list size is limited
                    else:
                        _ = value[i]
                else:
                    # value is float or NoInput
                    _ = value if i == 0 else NoInput(0)
                period.index_fixings = _

    @property
    def index_base(self):
        return self._index_base

    @index_base.setter
    def index_base(self, value):
        if isinstance(value, Series):
            value = IndexMixin._index_value(
                i_fixings=value,
                i_method=self.index_method,
                i_lag=self.index_lag,
                i_date=self.schedule.effective,
                i_curve=NoInput(0),  # not required because i_fixings is Series
            )
        self._index_base = value
        # if value is not None:
        for period in self.periods:
            if isinstance(period, (IndexFixedPeriod, IndexCashflow)):
                period.index_base = value


class ZeroFloatLeg(BaseLeg, FloatLegMixin):
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
        The method to use for adding a floating spread to compounded rates. Available
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

    def __init__(
        self,
        *args,
        float_spread: float | NoInput = NoInput(0),
        fixings: float | list | Series | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        self._delay_set_periods = True
        super().__init__(*args, **kwargs)
        if abs(float(self.amortization)) > 1e-2:
            raise NotImplementedError("`ZeroFloatLeg` cannot accept `amortization`.")
        self._set_fixings(fixings)
        self._set_periods()

    def _set_periods(self):
        self.periods = [
            FloatPeriod(
                float_spread=self.float_spread,
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                notional=self.notional - self.amortization * i,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                frequency=self.schedule.frequency,
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                roll=self.schedule.roll,
                calendar=self.schedule.calendar,
                fixing_method=self.fixing_method,
                fixings=self.fixings[i],
                method_param=self.method_param,
                spread_compound_method=self.spread_compound_method,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]

    @property
    def dcf(self):
        _ = [period.dcf for period in self.periods]
        return sum(_)

    def rate(self, curve):
        """
        Calculating a period type floating rate for the zero coupon leg.

        Parameters
        ----------
        curve : Curve, LineCurve
            The forecasting curve object.

        Returns
        -------
        float, Dual, Dual2
        """
        compounded_rate, total_dcf = 1.0, 0.0
        for period in self.periods:
            compounded_rate *= 1 + period.dcf * period.rate(curve) / 100
            total_dcf += period.dcf
        return 100 * (compounded_rate - 1.0) / total_dcf

    def npv(
        self,
        curve: Curve,
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the *ZeroFloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        disc_curve_: Curve = _disc_from_curve(curve, disc_curve)
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

    def fixings_table(self, curve: Curve):  # pragma: no cover
        """Not yet implemented for ZeroFloatLeg"""
        # TODO: fixing table for ZeroFloatLeg
        raise NotImplementedError("fixings table on ZeroFloatLeg.")

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the analytic delta of the *ZeroFloatLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        compounded_rate = 1.0
        for period in self.periods:
            compounded_rate *= 1 + period.dcf * period.rate(curve) / 100

        a_sum = 0.0
        for period in self.periods:
            _ = period.analytic_delta(curve, disc_curve_, fx, base) / disc_curve_[period.payment]
            _ *= compounded_rate / (1 + period.dcf * period.rate(curve) / 100)
            a_sum += _
        a_sum *= disc_curve_[self.schedule.pschedule[-1]] * fx
        return a_sum

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the properties of the *ZeroFloatLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if curve is NoInput.blank:
            rate, cashflow = None, None
            if disc_curve_ is NoInput.blank:
                npv, npv_fx, df, collateral = None, None, None, None
        else:
            rate = float(self.rate(curve))
            cashflow = -float(self.notional * self.dcf * rate / 100)
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)
            df = float(disc_curve_[self.schedule.pschedule[-1]])
            collateral = disc_curve_.collateral

        spread = 0.0 if self.float_spread is NoInput.blank else float(self.float_spread)
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
                defaults.headers["notional"]: float(self.notional),
                defaults.headers["df"]: df,
                defaults.headers["rate"]: rate,
                defaults.headers["spread"]: spread,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: float(fx),
                defaults.headers["npv_fx"]: npv_fx,
                defaults.headers["collateral"]: collateral,
            }
        ]
        return DataFrame.from_records(seq)


class ZeroFixedLeg(BaseLeg, FixedLegMixin):
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

    def __init__(self, *args, fixed_rate: float | NoInput = NoInput(0), **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_rate = fixed_rate
        if self.schedule.frequency == "Z":
            raise ValueError(
                "`frequency` for a ZeroFixedLeg should not be 'Z'. The Leg is zero frequency by "
                "construction. Set the `frequency` equal to the compounding frequency of the "
                "expressed fixed rate, e.g. 'S' for semi-annual compounding."
            )

    def _set_periods(self):
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
            )
        ]

    @property
    def fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        # overload the setter for a zero coupon to convert from IRR to period rate.
        # the headline fixed_rate is the IRR rate but the rate attached to Periods is a simple
        # rate in order to determine cashflows according to the normal cashflow logic.
        self._fixed_rate = value
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        if value is not NoInput.blank:
            period_rate = 100 * (1 / self.dcf) * ((1 + value / (100 * f)) ** (self.dcf * f) - 1)
        else:
            period_rate = NoInput(0)

        for period in self.periods:
            if isinstance(period, FixedPeriod):  # there should only be one FixedPeriod in a Zero
                period.fixed_rate = period_rate

    @property
    def dcf(self):
        """
        The DCF of a *ZeroFixedLeg* is defined as DCF of the single *FixedPeriod*
        spanning the *Leg*.
        """
        _ = [period.dcf for period in self.periods]
        return sum(_)

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the cashflows of the *ZeroFixedLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        rate = self.fixed_rate
        cashflow = self.periods[0].cashflow

        if disc_curve is NoInput.blank or rate is NoInput.blank:
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)
            df = float(disc_curve_[self.schedule.pschedule[-1]])
            collateral = disc_curve_.collateral

        seq = [
            {
                defaults.headers["type"]: type(self).__name__,
                defaults.headers["stub_type"]: None,
                defaults.headers["currency"]: self.currency.upper(),
                defaults.headers["a_acc_start"]: self.schedule.effective,
                defaults.headers["a_acc_end"]: self.schedule.termination,
                defaults.headers["payment"]: self.schedule.pschedule[-1],
                defaults.headers["convention"]: self.convention,
                defaults.headers["dcf"]: self.dcf,
                defaults.headers["notional"]: float(self.notional),
                defaults.headers["df"]: df,
                defaults.headers["rate"]: self.fixed_rate,
                defaults.headers["spread"]: None,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: float(fx),
                defaults.headers["npv_fx"]: npv_fx,
                defaults.headers["collateral"]: collateral,
            }
        ]
        return DataFrame.from_records(seq)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *ZeroFixedLeg* from all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        if self.fixed_rate is NoInput.blank:
            return None

        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _ = self.notional * self.dcf * disc_curve_[self.periods[0].payment]
        _ *= (1 + self.fixed_rate / (100 * f)) ** (self.dcf * f - 1)
        return _ / 10000 * fx

    def _analytic_delta(self, *args, **kwargs) -> DualTypes:
        """
        Analytic delta based on period rate and not IRR.
        """
        _ = [period.analytic_delta(*args, **kwargs) for period in self.periods]
        return sum(_)

    def _spread(self, target_npv, fore_curve, disc_curve, fx=NoInput(0)):
        """
        Overload the _spread calc to use analytic delta based on period rate
        """
        a_delta = self._analytic_delta(fore_curve, disc_curve, fx, self.currency)
        period_rate = -target_npv / (a_delta * 100)
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _ = f * ((1 + period_rate * self.dcf / 100) ** (1 / (self.dcf * f)) - 1)
        return _ * 10000

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *ZeroFixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)


class ZeroIndexLeg(BaseLeg, IndexLegMixin):
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

       index_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.95}, index_base=100.0)
       zil = ZeroIndexLeg(
           effective=dt(2022, 1, 15),
           termination="3Y",
           frequency="S",
           index_method="monthly",
           index_base=100.25,
       )
       zil.cashflows(index_curve, curve)

    """

    def __init__(
        self,
        *args,
        index_base: float | Series | NoInput = NoInput(0),
        index_fixings: float | Series | NoInput = NoInput(0),
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        **kwargs,
    ):
        self.index_method = (
            defaults.index_method if index_method is NoInput.blank else index_method.lower()
        )
        self.index_lag = defaults.index_lag if index_lag is NoInput.blank else index_lag
        super().__init__(*args, **kwargs)
        self.index_fixings = index_fixings  # set index fixings after periods init
        self.index_base = index_base  # set after periods initialised

    def _set_periods(self):
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
                index_fixings=self.index_fixings,
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

    def cashflow(self, curve: IndexCurve | None = None):
        """Aggregate the cashflows on the *IndexFixedPeriod* and *Cashflow* period using an
        *IndexCurve*."""
        _ = self.periods[0].cashflow(curve) + self.periods[1].cashflow
        return _

    def cashflows(self, *args, **kwargs):
        """
        Return the properties of the *ZeroIndexLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        cfs = super().cashflows(*args, **kwargs)
        _ = cfs.iloc[[0]].copy()
        for attr in ["Cashflow", "NPV", "NPV Ccy"]:
            _[attr] += cfs.iloc[1][attr]
        _["Type"] = "ZeroIndexLeg"
        _["Period"] = None
        return _

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return 0.0

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class IndexFixedLeg(IndexLegMixin, FixedLegMixin, BaseLeg):
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
       index_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_base=100.0)
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

    # TODO: spread calculations to determine the fixed rate on this leg do not work.
    def __init__(
        self,
        *args,
        index_base: float,
        index_fixings: float | Series | NoInput = NoInput(0),
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        fixed_rate: float | NoInput = NoInput(0),
        **kwargs,
    ) -> None:
        self._fixed_rate = fixed_rate
        self.index_lag = defaults.index_lag if index_lag is NoInput.blank else index_lag
        self.index_method = (
            defaults.index_method if index_method is NoInput.blank else index_method.lower()
        )
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super().__init__(*args, **kwargs)
        self.index_fixings = index_fixings  # set index fixings after periods init
        self.index_base = index_base  # set after periods initialised

    def _set_periods(self) -> None:
        self.periods = []

        # initial exchange
        if self.initial_exchange:
            raise NotImplementedError(
                "Cannot construct `IndexFixedLeg` with `initial_exchange` "
                "due to not implemented `index_fixings` input argument applicable to "
                "the indexing-up the initial exchange."
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

        # regular periods
        regular_periods = [
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
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                roll=self.schedule.roll,
                calendar=self.schedule.calendar,
                index_base=self.index_base,
                index_method=self.index_method,
                index_fixings=self.index_fixings,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        if self.final_exchange and self.amortization != 0:
            amortization = [
                IndexCashflow(
                    notional=self.amortization,
                    payment=self.schedule.pschedule[1 + i],
                    currency=self.currency,
                    stub_type="Amortization",
                    rate=NoInput(0),
                    index_base=self.index_base,
                    index_fixings=self.index_fixings,
                    index_method=self.index_method,
                )
                for i in range(self.schedule.n_periods - 1)
            ]
            interleaved_periods = [
                val for pair in zip(regular_periods, amortization) for val in pair
            ]
            interleaved_periods.append(regular_periods[-1])  # add last regular period
        else:
            interleaved_periods = regular_periods
        self.periods.extend(interleaved_periods)

        # final cashflow
        if self.final_exchange:
            self.periods.append(
                IndexCashflow(
                    notional=self.notional - self.amortization * (self.schedule.n_periods - 1),
                    payment=self.schedule.calendar.lag(
                        self.schedule.aschedule[-1], self.payment_lag_exchange, True
                    ),
                    currency=self.currency,
                    stub_type="Exchange",
                    rate=NoInput(0),
                    index_base=self.index_base,
                    index_fixings=self.index_fixings,
                    index_method=self.index_method,
                )
            )

    def npv(self, *args, **kwargs):
        return super().npv(*args, **kwargs)

    def cashflows(self, *args, **kwargs):
        return super().cashflows(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs):
        return super().analytic_delta(*args, **kwargs)


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

    _do_not_repeat_set_periods = False
    _is_mtm = True

    def __init__(
        self,
        *args,
        fx_fixings: list | float | Dual | Dual2 | NoInput = NoInput(0),
        alt_currency: str | NoInput = NoInput(0),
        alt_notional: float | NoInput = NoInput(0),
        **kwargs,
    ):
        if alt_currency is NoInput.blank:
            raise ValueError("`alt_currency` and `currency` must be supplied for MtmLeg.")
        self.alt_currency = alt_currency.lower()
        self.alt_notional = defaults.notional if alt_notional is NoInput.blank else alt_notional
        self._delay_set_periods = True
        if "initial_exchange" not in kwargs:
            kwargs["initial_exchange"] = True
        kwargs["final_exchange"] = True
        super().__init__(*args, **kwargs)
        if self.amortization != 0:
            raise ValueError("`amortization` cannot be supplied to a `FixedLegExchangeMtm` type.")
        self.fx_fixings = fx_fixings  # calls the setter

    @property
    def notional(self):
        return self._notional

    @notional.setter
    def notional(self, value):
        self._notional = value

    @property
    def fx_fixings(self):
        """
        list : FX fixing values used for consecutive periods.
        """
        return self._fx_fixings

    def _get_fx_fixings_from_series(self, ser: Series, ini_period: int = 0):
        last_fixing_date = ser.index[-1]
        fixings_list = []
        for i in range(ini_period, self.schedule.n_periods):
            required_date = add_tenor(
                self.schedule.aschedule[i],
                f"{self.payment_lag_exchange}B",
                NoInput(0),
                self.schedule.calendar,
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
                        f"available within the Series."
                    )
        return fixings_list

    @fx_fixings.setter
    def fx_fixings(self, value):
        if value is NoInput.blank:
            self._fx_fixings = []
        elif isinstance(value, list):
            self._fx_fixings = value
        elif isinstance(value, (float, Dual, Dual2)):
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

    def _get_fx_fixings(self, fx):
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
        fx_fixings = self.fx_fixings.copy()

        if isinstance(fx, FXForwards):
            for i in range(n_given, n_req):
                fx_fixings.append(
                    fx.rate(
                        self.alt_currency + self.currency,
                        self.schedule.calendar.lag(
                            self.schedule.aschedule[i], self.payment_lag_exchange, True
                        ),
                    )
                )
        elif n_req > 0:  # only check if unknown fixings are required
            if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                raise ValueError(
                    "`fx` is required when `fx_fixings` are not pre-set and "
                    "if rateslib option `no_fx_fixings_for_xcs` is set to "
                    "'raise'."
                )
            if n_given == 0:
                if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                    warnings.warn(
                        "Using 1.0 for FX, no `fx` or `fx_fixing` given and "
                        "rateslib option `no_fx_fixings_for_xcs` is set to "
                        "'warn'.",
                        UserWarning,
                    )
                fx_fixings = [1.0] * n_req
            else:
                if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                    warnings.warn(
                        "Using final FX fixing given for missing periods, "
                        "rateslib option `no_fx_fixings_for_xcs` is set to "
                        "'warn'.",
                        UserWarning,
                    )
                fx_fixings.extend([fx_fixings[-1]] * (n_req - n_given))
        return fx_fixings

    def _set_periods(self, fx):
        fx_fixings = self._get_fx_fixings(fx)
        self.notional = fx_fixings[0] * self.alt_notional
        notionals = [self.alt_notional * fx_fixings[i] for i in range(len(fx_fixings))]

        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[0], self.payment_lag_exchange, True
                    ),
                    self.currency,
                    "Exchange",
                    fx_fixings[0],
                )
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            self._regular_period(
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                notional=notionals[i],
                iterator=i,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        mtm_flows = [
            Cashflow(
                -notionals[i + 1] + notionals[i],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[i + 1], self.payment_lag_exchange, True
                ),
                self.currency,
                "Mtm",
                fx_fixings[i + 1],
            )
            for i in range(len(fx_fixings) - 1)
        ]
        interleaved_periods = [val for pair in zip(regular_periods, mtm_flows) for val in pair]
        interleaved_periods.append(regular_periods[-1])
        self.periods.extend(interleaved_periods)

        # final cashflow
        self.periods.append(
            Cashflow(
                notionals[-1],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[-1], self.payment_lag_exchange, True
                ),
                self.currency,
                "Exchange",
                fx_fixings[-1],
            )
        )

    def npv(
        self,
        curve: Curve,
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().npv(curve, disc_curve, fx, base, local)
        # self._is_set_periods_fx = False
        return ret

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().cashflows(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().analytic_delta(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret


class FixedLegMtm(BaseLegMtm, FixedLegMixin):
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
        *args,
        fixed_rate: float | NoInput = NoInput(0),
        **kwargs,
    ):
        self._fixed_rate = fixed_rate
        super().__init__(
            *args,
            **kwargs,
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatLegMtm(BaseLegMtm, FloatLegMixin):
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
        *args,
        float_spread: float | NoInput = NoInput(0),
        fixings: float | list | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        **kwargs,
    ):
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

       fp1 = FixedPeriod(dt(2021,1,1), dt(2021,7,1), dt(2021,7,2), "Q", 1e6, "Act365F", fixed_rate=2.10)
       fp2 = FixedPeriod(dt(2021,3,7), dt(2021,9,7), dt(2021,9,8), "Q", -5e6, "Act365F", fixed_rate=3.10)
       custom_leg = CustomLeg(periods=[fp1, fp2])
       custom_leg.cashflows(curve)

    """  # noqa: E501

    def __init__(self, periods):
        if not all(isinstance(p, (FixedPeriod, FloatPeriod, Cashflow)) for p in periods):
            raise ValueError(
                "Each object in `periods` must be of type {FixedPeriod, FloatPeriod, " "Cashflow}."
            )
        self._set_periods(periods)

    def _set_periods(self, periods):
        self.periods = periods

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *CustomLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def cashflows(self, *args, **kwargs):
        """
        Return the properties of the *CustomLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *CustomLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
