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

from abc import abstractmethod, ABCMeta
from datetime import datetime
from typing import Optional, Union
import abc
import warnings

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.calendars import add_tenor
from rateslib.scheduling import Schedule
from rateslib.curves import Curve, IndexCurve
from rateslib.periods import (
    IndexFixedPeriod,
    FixedPeriod,
    FloatPeriod,
    Cashflow,
    IndexCashflow,
    IndexMixin,
    _validate_float_args,
    _get_fx_and_base,
)
from rateslib.dual import Dual, Dual2, set_order, DualTypes
from rateslib.fx import FXForwards, FXRates


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
        static data.
    payment_lag : int, optional
        The number of business days to lag payments by.
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

    Notes
    -----
    See also :class:`~rateslib.scheduling.Schedule` for a more thorough description
    of some of these arguments.

    Attributes
    ----------
    schedule : Schedule
    notional : float
    currency : str
    amortization : float
    convention : str
    periods : list

    See Also
    --------
    FixedLeg : Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.
    FloatLeg : Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.
    BaseLegExchange : Abstract base class for ``Legs`` with notional exchanges.
    CustomLeg : Create a leg composed of user specified periods.
    """

    @abc.abstractmethod
    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str],
        frequency: str,
        stub: Optional[str] = None,
        front_stub: Optional[datetime] = None,
        back_stub: Optional[datetime] = None,
        roll: Optional[Union[str, int]] = None,
        eom: Optional[bool] = None,
        modifier: Optional[str] = False,
        calendar: Optional[Union[CustomBusinessDay, str]] = None,
        payment_lag: Optional[int] = None,
        notional: Optional[float] = None,
        currency: Optional[str] = None,
        amortization: Optional[float] = None,
        convention: Optional[str] = None,
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
        self.convention = defaults.convention if convention is None else convention
        self.currency = defaults.base_currency if currency is None else currency.lower()
        if getattr(self, "_no_base_notional", False):
            self._notional = defaults.notional if notional is None else notional
        else:
            self.notional = defaults.notional if notional is None else notional
        if getattr(self, "_no_base_notional", False):
            self._amortization = 0 if amortization is None else amortization
        else:
            self.amortization = 0 if amortization is None else amortization
        self.periods = []

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the leg object via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        sum = 0
        for period in self.periods:
            sum += period.analytic_delta(*args, **kwargs)
        return sum

    def cashflows(self, *args, **kwargs):
        """
        Return the properties of the leg used in calculating cashflows.

        Parameters
        ----------
        args :
            Positional arguments supplied to :meth:`~rateslib.periods.BasePeriod.cashflows`.
        kwargs :
            Keyword arguments supplied to :meth:`~rateslib.periods.BasePeriod.cashflows`.

        Returns
        -------
        DataFrame
        """
        seq = [period.cashflows(*args, **kwargs) for period in self.periods]
        return DataFrame.from_records(seq)

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the leg object via summing all periods.

        Calculates the cashflow for the all periods and multiplies them by the
        DF associated with each payment date.

        Parameters
        ----------
        args :
            Positional arguments supplied to :meth:`~rateslib.periods.BasePeriod.npv`.
        kwargs :
            Keyword arguments supplied to :meth:`~rateslib.periods.BasePeriod.npv`.

        Returns
        -------
        Dual
        """
        sum = 0
        _is_local = (len(args) == 5 and args[4]) or kwargs.get("local", False)
        if _is_local:
            for period in self.periods:
                sum += period.npv(*args, **kwargs)[self.currency]
            return {self.currency: sum}
        else:
            for period in self.periods:
                sum += period.npv(*args, **kwargs)
            return sum

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
            if (
                "rfr" in self.fixing_method
                and self.spread_compound_method != "none_simple"
            ):
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
                a_, b_ = period._get_analytic_delta_quadratic_coeffs(
                    fore_curve, disc_curve
                )
                a += a_
                b += b_
            except AttributeError:
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
        self, target_npv, fore_curve, disc_curve, fx=None
    ):  # pragma: no cover
        # This method is unused and untested, superseded by _spread_isda_approx_rate

        # This method creates a dual2 variable for float spread and obtains derivatives automatically
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
        b = npv.gradient("spread_z", order=1)[0]
        a = 0.5 * npv.gradient("spread_z", order=2)[0][0]
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
                "Divide by zero encountered and the spread is approximated to "
                "first order.",
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

    def _spread(self, target_npv, fore_curve, disc_curve, fx=None):
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
        float or None : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        self._fixed_rate = value
        # if value is not None:
        for period in self.periods:
            if isinstance(period, FixedPeriod):
                period.fixed_rate = value


class FixedLeg(BaseLeg, FixedLegMixin):
    """
    Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The rate applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market rate for all periods has been calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.
    """

    def __init__(self, *args, fixed_rate: Optional[float] = None, **kwargs):
        self._fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)
        self.periods = [
            FixedPeriod(
                fixed_rate=self.fixed_rate,
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                notional=self.notional - self.amortization * i,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                frequency=self.schedule.frequency,
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatLegMixin:
    """
    Add the functionality to add and retrieve ``float_spread`` on
    :class:`~rateslib.periods.FloatPeriod` s and a :meth:`fixings_table`.
    """

    def _set_fixings(
        self,
        fixings,
    ):
        """
        Re-organises the fixings input to list structure for each period.
        Requires a ``schedule`` object and ``float_args``.
        """
        if fixings is None:
            fixings = []
        elif isinstance(fixings, Series):
            last_fixing = fixings.index[-1]
            if self.fixing_method in ["rfr_payment_delay", "rfr_lockout"]:
                adj_days = 0
            else:
                # fixing_method in ["rfr_lookback", "rfr_observation_shift", "ibor"]:
                adj_days = self.method_param
            first_required_day = [
                add_tenor(
                    self.schedule.aschedule[i],
                    f"-{adj_days}B",
                    None,
                    self.schedule.calendar,
                )
                for i in range(self.schedule.n_periods)
            ]
            fixings = [
                fixings if last_fixing >= day else None for day in first_required_day
            ]
        elif not isinstance(fixings, list):
            fixings = [fixings]

        self.fixings = fixings + [None] * (self.schedule.n_periods - len(fixings))
        return None

    @property
    def float_spread(self):
        """
        float or None : If set will also set the ``float_spread`` of contained
            :class:`~rateslib.periods.FloatPeriod` s.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value):
        self._float_spread = value
        # if value is not None:
        for period in self.periods:
            if isinstance(period, FloatPeriod):
                period.float_spread = value if value is not None else 0.0

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

    def fixings_table(self, curve: Curve):
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        See :meth:`~rateslib.periods.FloatPeriod.fixings_table` for more info.

        Parameters
        ----------
        curve : Curve
            The forecast :class:`~rateslib.curves.Curve` or
            :class:`~rateslib.curves.LineCurve` needed to calculate rates which
            affect compounding and dependent notional exposure.

        Returns
        -------
        DataFrame
        """
        df, _ = None, 0
        while df is None:
            if type(self.periods[_]) is FloatPeriod:
                df = self.periods[_].fixings_table(curve)
            _ += 1

        n = len(self.periods)
        for _ in range(_, n):
            if type(self.periods[_]) is FloatPeriod:
                df = pd.concat([df, self.periods[_].fixings_table(curve)])
        return df


class FloatLeg(BaseLeg, FloatLegMixin):
    """
    Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.

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
    ... warn::

        When floating rates are determined from historical fixings the forecast
        ``Curve`` ``calendar`` will be used to determine fixing dates.
        If this calendar does not align with the leg ``calendar`` then
        spurious results or errors may be generated. Including the curve calendar in
        the leg is acceptable, i.e. a leg calendar of *"nyc,ldn,tgt"* and a curve
        calendar of *"ldn"* is valid, whereas only *"nyc,tgt"* may give errors.

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

    Set the initial RFR fixings in the first period of an RFR leg (notice the sublist
    and the implied -10% year end turn).

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2022, 12, 28),
           termination="2M",
           frequency="M",
           fixings=[[1.19, 1.19, -8.81]],
           currency="SEK",
       )
       float_leg.cashflows(curve)
       float_leg.fixings_table(curve)[dt(2022,12,28):dt(2023,1,4)]
    """

    def __init__(
        self,
        *args,
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list, Series]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)

        self._set_fixings(fixings)
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
                fixing_method=self.fixing_method,
                fixings=self.fixings[i],
                method_param=self.method_param,
                spread_compound_method=self.spread_compound_method,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]

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
    index_fixings = None

    def _set_index_fixings(
        self,
        index_fixings,
    ):
        """
        Re-organises the fixings input to list structure for each period.
        Requires a ``schedule`` object and ``float_args``.
        """
        if index_fixings is None:
            _ = []
        elif isinstance(index_fixings, Series):
            last_fixing = index_fixings.index[-1]
            if self.index_method == "daily":
                first_req = [
                    self.schedule.aschedule[i + 1]
                    for i in range(self.schedule.n_periods)
                ]
            else:  # index_method == "monthly":
                first_req = [
                    datetime(
                        self.schedule.aschedule[i + 1].year,
                        self.schedule.aschedule[i + 1].month,
                        1,
                    )
                    for i in range(self.schedule.n_periods)
                ]
            _ = [index_fixings if last_fixing >= day else None for day in first_req]
        elif not isinstance(index_fixings, list):
            _ = [index_fixings]
        else:
            _ = index_fixings

        self.index_fixings = _ + [None] * (self.schedule.n_periods - len(_))
        return None


class IndexFixedLeg(IndexLegMixin, FixedLeg):
    """
    Create a fixed leg composed of :class:`~rateslib.periods.IndexFixedPeriod` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The rate applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market rate for all periods has been calculated.
    index_base : float or None, optional
        The base index applied to all periods.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the first
        period.
        If a list of *n* fixings will be used as the index fixings for the first *n*
        periods.
        If a datetime indexed ``Series`` will use the fixings that are available in
        that object, and derive the rest from the ``curve``.
    index_method : str, optional
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.
    """

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        index_base: float,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: Optional[str] = None,
        index_lag: Optional[int] = None,
        **kwargs,
    ):

        super().__init__(*args, fixed_rate=fixed_rate, **kwargs)
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = defaults.index_method if index_method is None else index_method.lower()
        self.index_lag = defaults.index_lag if index_lag is None else index_lag
        self._set_index_fixings(index_fixings)
        self.periods = [
            IndexFixedPeriod(
                fixed_rate=self.fixed_rate,
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                notional=self.notional - self.amortization * i,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                frequency=self.schedule.frequency,
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                index_base=index_base,
                index_fixings=self.index_fixings[i],
                index_method=index_method,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]


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
    .. warning::

       When floating rates are determined from historical fixings the forecast
       ``Curve`` ``calendar`` will be used to determine fixing dates.
       If this calendar does not align with the leg ``calendar`` then
       spurious results or errors may be generated. Including the curve calendar in
       the leg is acceptable, i.e. a leg calendar of *"nyc,ldn,tgt"* and a curve
       calendar of *"ldn"* is valid, whereas only *"nyc,tgt"* may give errors.

    Examples
    --------
    TODO
    """

    def __init__(
        self,
        *args,
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list, Series]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)
        if abs(float(self.amortization)) > 1e-2:
            raise NotImplementedError("`ZeroFloatLeg` cannot accept `amortization`.")

        self._set_fixings(fixings)
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
                fixing_method=self.fixing_method,
                fixings=self.fixings[i],
                method_param=self.method_param,
                spread_compound_method=self.spread_compound_method,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]

    @property
    def dcf(self):
        _ = 0.0
        for period in self.periods:
            _ += period.dcf
        return _

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
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = (
            self.rate(curve)
            / 100
            * self.dcf
            * disc_curve[self.schedule.pschedule[-1]]
            * -self.notional
        )
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def fixings_table(self, curve: Curve):  # pragma: no cover
        # TODO: fixing table for ZeroFloatLeg
        raise NotImplementedError("fixings table on ZeroFloatLeg.")

    def analytic_delta(self, *args, **kwargs):  # pragma: no cover
        # TODO: a delta for ZeroFloatLeg
        raise NotImplementedError("analytic delta on ZeroFloatLeg.")

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        """
        Return the properties of the leg used in calculating cashflows.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults to
            ``fx.base``.

        Returns
        -------
        DataFrame
        """
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)
        rate = None if curve is None else float(self.rate(curve))
        cashflow = (
            None if rate is None else -float(self.notional * self.dcf * rate / 100)
        )
        if disc_curve is None or rate is None:
            npv, npv_fx = None, None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)
        spread = 0.0 if self.float_spread is None else float(self.float_spread)
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
                defaults.headers["df"]: None
                if disc_curve is None
                else float(disc_curve[self.schedule.pschedule[-1]]),
                defaults.headers["rate"]: rate,
                defaults.headers["spread"]: spread,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: float(fx),
                defaults.headers["npv_fx"]: npv_fx,
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

    The rate in this calculation is not a period rate but an IRR defining the cashflow
    as follows,

    .. math::

       C = -N ( (1 + \\frac{R}{f}) ^ {df} - 1)

    Examples
    --------
    TODO
    """

    def __init__(self, *args, fixed_rate: Optional[float] = None, **kwargs):

        super().__init__(*args, **kwargs)
        self.periods = [
            FixedPeriod(
                fixed_rate=None,
                start=self.schedule.effective,
                end=self.schedule.termination,
                payment=self.schedule.pschedule[-1],
                notional=self.notional,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                frequency=self.schedule.frequency,
                stub=False,
            )
        ]
        self.fixed_rate = fixed_rate

    @property
    def fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of
            contained :class:`FixedPeriod` s.
        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        # overload the setter for a zero coupon to convert from irr to period rate
        self._fixed_rate = value
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        if value is not None:
            period_rate = 100 * (1 / self.dcf) * ((1 + value / (100 * f)) ** (self.dcf * f) - 1)
        else:
            period_rate = None

        for period in self.periods:
            if isinstance(period, FixedPeriod):
                period.fixed_rate = period_rate

    @property
    def dcf(self):
        _ = 0.0
        for period in self.periods:
            _ += period.dcf
        return _

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        """
        Return the properties of the leg used in calculating cashflows.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults to
            ``fx.base``.

        Returns
        -------
        DataFrame
        """
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)
        rate = self.fixed_rate
        cashflow = self.periods[0].cashflow
        if disc_curve is None or rate is None:
            npv, npv_fx = None, None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)
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
                defaults.headers["df"]: None
                if disc_curve is None
                else float(disc_curve[self.schedule.pschedule[-1]]),
                defaults.headers["rate"]: self.fixed_rate,
                defaults.headers["spread"]: None,
                defaults.headers["cashflow"]: cashflow,
                defaults.headers["npv"]: npv,
                defaults.headers["fx"]: float(fx),
                defaults.headers["npv_fx"]: npv_fx,
            }
        ]
        return DataFrame.from_records(seq)

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ) -> DualTypes:
        """
        Return the analytic delta of the leg object.

        Parameters
        ----------
        curve : Curve
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.

        Returns
        -------
        float, Dual, Dual2, None

        Notes
        -----
        For a :class:`ZeroFixedLeg` this gives the sensitivity to the IRR fixed rate.
        This is a non-linear quantity and depends on the fixed rate itself.
        This value is *not* used when determining mid-market prices.

        .. math::

           \\frac{\\partial P}{\\partial R} = N v d \\left ( 1 + \\frac{R^{irr}}{f} \\right ) ^{df-1}

        If the ``fixed_rate`` is undetermined and set to *None* will return *None*.

        """
        disc_curve_: Curve = disc_curve or curve
        if self.fixed_rate is None:
            return None

        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _ = self.notional * self.dcf * disc_curve_[self.periods[0].payment]
        _ *= (1 + self.fixed_rate / (100*f)) ** (self.dcf * f - 1)
        return _ / 10000

    def _analytic_delta(self, *args, **kwargs) -> DualTypes:
        """
        Analytic delta based on period rate and not IRR.
        """
        _ = 0.0
        for period in self.periods:
            _ += period.analytic_delta(*args, **kwargs)
        return _

    def _spread(self, target_npv, fore_curve, disc_curve, fx=None):
        """
        Overload the _spread calc to use analytic delta based on period rate
        """
        a_delta = self._analytic_delta(fore_curve, disc_curve, fx, self.currency)
        period_rate = -target_npv / (a_delta * 100)
        f = 12 / defaults.frequency_months[self.schedule.frequency]
        _ = f * ((1 + period_rate * self.dcf/ 100)**(1/(self.dcf*f)) - 1)
        return _ * 10000


class ZeroIndexLeg(BaseLeg):
    """
    Create a zero coupon index leg.

    This leg is composed of an :class:`~rateslib.perods.IndexFixedPeriod` set to 100%
    for the indexing up of the notional and an offsetting
    :class:`~rateslib.periods.Cashflow` to negate the notional.

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

    """

    def __init__(
        self,
        *args,
        index_base: Optional[Union[float, Series]] = None,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: Optional[str] = None,
        index_lag: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = defaults.index_method if index_method is None else index_method.lower()
        self.index_lag = defaults.index_lag if index_lag is None else index_lag
        # The first period indexes up the complete notional amount.
        # The second period deducts the un-indexed notional amount.
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
                index_base=self.index_base,
                index_fixings=self.index_fixings,
                index_lag=self.index_lag,
                index_method=self.index_method,
            ),
            Cashflow(
                notional=-self.notional,
                payment=self.schedule.pschedule[-1],
                currency=self.currency,
                stub_type=None,
                rate=None,
            )
        ]

    def cashflow(self, curve: Optional[IndexCurve] = None):
        _ = self.periods[0].cashflow(curve) + self.periods[1].cashflow
        return _

    def cashflows(self, *args, **kwargs):
        cfs = super().cashflows(*args, **kwargs)
        _ = cfs.iloc[[0]].copy()
        for attr in ["Cashflow", "NPV", "NPV Ccy"]:
            _[attr] += cfs.iloc[1][attr]
        _["Type"] = "ZeroIndexLeg"
        _["Period"] = None
        return _


# class ZeroIndexLeg2(BaseLeg):
#     """
#     Create a zero coupon index leg
#
#
#
#     """
#
#     def __init__(
#         self,
#         *args,
#         index_base: Optional[Union[float, Series]] = None,
#         index_fixings: Optional[Union[float, Series]] = None,
#         index_method: str = "daily",
#         index_lag: Optional[int] = None,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.index_base = index_base
#         self.index_fixings = index_fixings
#         self.index_method = defaults.index_method if index_method is None else index_method.lower()
#         self.index_lag = defaults.index_lag if index_lag is None else index_lag
#
#     def cashflow(self, curve: Optional[IndexCurve] = None):
#         base_value = IndexMixin._index_value(
#             i_fixings=self.index_base,
#             i_date=self.schedule.effective,
#             i_curve=curve,
#             i_lag=self.index_lag,
#             i_method=self.index_method,
#         )
#         end_value = IndexMixin._index_value(
#             i_fixings=self.index_fixings,
#             i_date=self.schedule.termination,
#             i_curve=curve,
#             i_lag=self.index_lag,
#             i_method=self.index_method,
#         )
#         _ = -self.notional * (end_value / base_value - 1)
#         return _
#
#     def npv(
#         self,
#         curve: IndexCurve,
#         disc_curve: Optional[Curve] = None,
#         fx: Optional[Union[float, FXRates, FXForwards]] = None,
#         base: Optional[str] = None,
#         local: bool = False,
#     ) -> DualTypes:
#         """
#         Return the NPV of the leg object.
#
#         Calculates the cashflow for the period and multiplies it by the DF associated
#         with the payment date.
#
#         Parameters
#         ----------
#         curve : IndexCurve, optional
#             The forecasting curve object. Not used unless it is set equal to
#             ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
#         disc_curve : Curve, optional
#             The discounting curve object used in calculations.
#             Set equal to ``curve`` if not given.
#         fx : float, FXRates, FXForwards, optional
#             The immediate settlement FX rate that will be used to convert values
#             into another currency. A given `float` is used directly. If giving a
#             :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
#             object, converts from local currency into ``base``.
#         base : str, optional
#             The base currency to convert cashflows into (3-digit code), set by default.
#             Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
#             :class:`~rateslib.fx.FXForwards` object.
#         local : bool, optional
#             If `True` will ignore the ``base`` request and return a dict identifying
#             local currency NPV.
#
#         Returns
#         -------
#         float, Dual, Dual2, or dict of such
#         """
#         disc_curve = disc_curve or curve
#         if disc_curve is None or curve is None:
#             raise TypeError(
#                 "`curves` have not been supplied correctly. NoneType has been detected."
#             )
#         if self.schedule.pschedule[-1] < disc_curve.node_dates[0]:
#             return 0.0  # payment date is in the past avoid issues with fixings or rates
#         fx, base = _get_fx_and_base(self.currency, fx, base)
#         value = self.cashflow(curve) * disc_curve[self.payment]
#         if local:
#             return {self.currency: value}
#         else:
#             return fx * value


class BaseLegExchange(BaseLeg):
    """
    Abstract base class with common parameters for all ``LegExchange`` subclasses.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    initial_exchange : bool
        Whether to also include an initial notional exchange.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    See Also
    --------
    FixedLegExchange : Create a fixed leg with additional notional exchanges.
    FloatLegExchange : Create a floating leg with additional notional exchanges.
    BaseLegExchangeMtm : Base class for legs with additional MTM notional exchanges.
    """

    _is_mtm = False

    def __init__(
        self,
        *args,
        initial_exchange: bool = True,
        payment_lag_exchange: Optional[int] = None,
        **kwargs,
    ):
        self._no_base_notional = True
        self.payment_lag_exchange = (
            defaults.payment_lag_exchange
            if payment_lag_exchange is None
            else payment_lag_exchange
        )
        self.initial_exchange = initial_exchange
        super().__init__(*args, **kwargs)

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
    def _set_periods(self):
        pass  # pragma: no cover


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FixedLegExchange(FixedLegMixin, BaseLegExchange):
    """
    Create a leg of :class:`~rateslib.periods.FixedPeriod` s and initial and final
    :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLegExchange`.

    Notes
    -----
    The initial cashflow notional is set as the negative of the notional. The payment
    date is set equal to the accrual start date adjusted by
    the ``payment_lag_exchange``.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accrual date adjusted by ``payment_lag_exchange``.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods. For similar examples see
    :class:`~rateslib.legs.FloatLegExchange`.
    """

    def __init__(self, *args, fixed_rate: Optional[float] = None, **kwargs) -> None:
        self._fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)
        self._set_periods()

    def _set_periods(self) -> None:
        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    add_tenor(
                        self.schedule.aschedule[0],
                        f"{self.payment_lag_exchange}B",
                        None,
                        self.schedule.calendar,
                    ),
                    self.currency,
                    "Exchange",
                )
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            FixedPeriod(
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
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        if self.amortization != 0:
            amortization = [
                Cashflow(
                    self.amortization,
                    self.schedule.pschedule[1 + i],
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
        self.periods.append(
            Cashflow(
                self.notional - self.amortization * (self.schedule.n_periods - 1),
                add_tenor(
                    self.schedule.aschedule[-1],
                    f"{self.payment_lag_exchange}B",
                    None,
                    self.schedule.calendar,
                ),
                self.currency,
                "Exchange",
            )
        )


class IndexFixedLegExchange(IndexLegMixin, FixedLegMixin, BaseLegExchange):
    """
    Create a leg of :class:`~rateslib.periods.IndexFixedPeriod` s and initial and
    final :class:`~rateslib.periods.IndexCashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
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
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLegExchange`.

    Notes
    -----
    An initial exchange is not currently implement for this leg.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accrual date adjusted by ``payment_lag_exchange``.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods. For similar examples see
    :class:`~rateslib.legs.FloatLegExchange`.
    """

    # TODO: spread calculations to determine the fixed rate on this leg do not work.
    def __init__(
        self,
        *args,
        index_base: float,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: Optional[str] = None,
        index_lag: Optional[int] = None,
        fixed_rate: Optional[float] = None,
        **kwargs,
    ) -> None:
        self._fixed_rate = fixed_rate
        self.index_base = index_base
        self.index_lag = defaults.index_lag if index_lag is None else index_lag
        self.index_method = defaults.index_method if index_method is None else index_method.lower()
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super().__init__(*args, **kwargs)
        if self.initial_exchange:
            raise NotImplementedError(
                "Cannot construct `IndexFixedLegExchange` with `initial_exchange` "
                "due to not implemented `index_fixings` input argument applicable to "
                "the indexing-up the initial exchange."
            )

        self._set_index_fixings(index_fixings)
        self._set_periods()

    def _set_periods(self) -> None:
        # initial exchange
        self.periods = (
            [
                IndexCashflow(
                    notional=-self.notional,
                    payment=add_tenor(
                        self.schedule.aschedule[0],
                        f"{self.payment_lag_exchange}B",
                        None,
                        self.schedule.calendar,
                    ),
                    currency=self.currency,
                    stub_type="Exchange",
                    rate=None,
                    index_base=self.index_base,
                    index_fixings=self.index_fixings[0],
                    index_method=self.index_method,
                )
            ]
            if self.initial_exchange
            else []
        )

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
                index_base=self.index_base,
                index_method=self.index_method,
                index_fixings=self.index_fixings[i],
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        if self.amortization != 0:
            amortization = [
                IndexCashflow(
                    notional=self.amortization,
                    payment=self.schedule.pschedule[1 + i],
                    currency=self.currency,
                    stub_type="Amortization",
                    rate=None,
                    index_base=self.index_base,
                    index_fixings=self.index_fixings[1 + i],
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
        self.periods.append(
            IndexCashflow(
                notional=self.notional
                - self.amortization * (self.schedule.n_periods - 1),
                payment=add_tenor(
                    self.schedule.aschedule[-1],
                    f"{self.payment_lag_exchange}B",
                    None,
                    self.schedule.calendar,
                ),
                currency=self.currency,
                stub_type="Exchange",
                rate=None,
                index_base=self.index_base,
                index_fixings=self.index_fixings[-1],
                index_method=self.index_method,
            )
        )


class FloatLegExchange(BaseLegExchange, FloatLegMixin):
    """
    Create a leg of :class:`~rateslib.periods.FloatPeriod` s and initial and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
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
    kwargs : dict
        Required keyword arguments to :class:`BaseLegExchange`.

    Notes
    -----
    For more details of floating rate options see :class:`rateslib.periods.FloatPeriod`.

    The initial cashflow notional is set as the negative of the notional. The payment
    date is set equal to the accrual date adjusted by the ``payment_lag_exchange``
    days.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accraul date adjusted by the ``payment_lag_exchange``
    days.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods.

    Examples
    --------

    .. ipython:: python

       float_leg_exch = FloatLegExchange(dt(2022, 1, 1), "9M", "Q", notional=1000000)
       float_leg_exch.cashflows(curve)

    .. ipython:: python

       float_leg_exch = FloatLegExchange(dt(2022, 1, 1), "9M", "Q", notional=1000000, amortization=200000)
       float_leg_exch.cashflows(curve)
    """

    def __init__(
        self,
        *args,
        float_spread: Union[float, None] = None,
        fixings: Optional[Union[float, list, Series]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)

        self._set_fixings(fixings)
        self._set_periods()

    def _set_periods(self):
        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    add_tenor(
                        self.schedule.aschedule[0],
                        f"{self.payment_lag_exchange}B",
                        None,
                        self.schedule.calendar,
                    ),
                    self.currency,
                    "Exchange",
                )
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            FloatPeriod(
                float_spread=self.float_spread,
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                frequency=self.schedule.frequency,
                notional=self.notional - self.amortization * i,
                currency=self.currency,
                convention=self.convention,
                termination=self.schedule.termination,
                stub=True if period[defaults.headers["stub_type"]] == "Stub" else False,
                fixings=self.fixings[i],
                fixing_method=self.fixing_method,
                method_param=self.method_param,
                spread_compound_method=self.spread_compound_method,
            )
            for i, period in self.schedule.table.to_dict(orient="index").items()
        ]
        if self.amortization != 0:
            amortization = [
                Cashflow(
                    self.amortization,
                    self.schedule.pschedule[1 + i],
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
        self.periods.append(
            Cashflow(
                self.notional - self.amortization * (self.schedule.n_periods - 1),
                add_tenor(
                    self.schedule.aschedule[-1],
                    f"{self.payment_lag_exchange}B",
                    None,
                    self.schedule.calendar,
                ),
                self.currency,
                "Exchange",
            )
        )


class BaseLegExchangeMtm(BaseLegExchange, metaclass=ABCMeta):
    _do_not_repeat_set_periods = False
    _is_mtm = True

    """
    Abstract base class with common parameters for all ``LegExchangeMtm``
    subclasses.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
    fx_fixings : float, Dual, Dual2 or list of such
        Define the known FX fixings for each period which affects the mark-the-market
        (MTM) notional exchanges after each period. If not given, or only some
        FX fixings are given, the remaining unknown fixings will be forecast
        by a provided :class:`~rateslib.fx.FXForwards` object later.
    alt_currency : str
        The alternative reference currency against which FX fixings are measured
        for MTM notional exchanges (3-digit code).
    alt_notional : float
        The notional expressed in the alternative currency which will be used to
        determine the notional for this leg using the ``fx_fixings`` as FX rates.
    kwargs : dict
        Required keyword args to :class:`BaseLegExchange`.

    See Also
    --------
    FixedLegExchangeMtm: Create a fixed leg with notional and Mtm exchanges.
    FloatLegExchangeMtm : Create a floating leg with notional and Mtm exchanges.
    """

    def __init__(
        self,
        *args,
        fx_fixings: Optional[Union[list, float, Dual, Dual2]] = None,
        alt_currency: str = None,
        alt_notional: Optional[float] = None,
        **kwargs,
    ):
        self.alt_currency = alt_currency.lower()
        self.alt_notional = defaults.notional if alt_notional is None else alt_notional
        super().__init__(*args, **kwargs)
        if self.amortization != 0:
            raise ValueError(
                "`amortization` cannot be supplied to a `FixedLegExchangeMtm` type."
            )
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

    @fx_fixings.setter
    def fx_fixings(self, value):
        if value is None:
            self._fx_fixings = []
        elif isinstance(value, list):
            self._fx_fixings = value
        elif isinstance(value, (float, Dual, Dual2)):
            self._fx_fixings = [value]
        else:
            raise TypeError("`fx_fixings` should be scalar value or list of such")

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
                        add_tenor(
                            self.schedule.aschedule[i],
                            f"{self.payment_lag_exchange}B",
                            None,
                            self.schedule.calendar,
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

    @abc.abstractmethod
    def _regular_period(self, *args, **kwargs):
        pass  # pragma: no cover

    def _set_periods(self, fx):
        fx_fixings = self._get_fx_fixings(fx)
        self.notional = fx_fixings[0] * self.alt_notional
        notionals = [self.alt_notional * fx_fixings[i] for i in range(len(fx_fixings))]

        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    add_tenor(
                        self.schedule.aschedule[0],
                        f"{self.payment_lag_exchange}B",
                        None,
                        self.schedule.calendar,
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
                add_tenor(
                    self.schedule.aschedule[i + 1],
                    f"{self.payment_lag_exchange}B",
                    None,
                    self.schedule.calendar,
                ),
                self.currency,
                "Mtm",
                fx_fixings[i + 1],
            )
            for i in range(len(fx_fixings) - 1)
        ]
        interleaved_periods = [
            val for pair in zip(regular_periods, mtm_flows) for val in pair
        ]
        interleaved_periods.append(regular_periods[-1])
        self.periods.extend(interleaved_periods)

        # final cashflow
        self.periods.append(
            Cashflow(
                notionals[-1],
                add_tenor(
                    self.schedule.aschedule[-1],
                    f"{self.payment_lag_exchange}B",
                    None,
                    self.schedule.calendar,
                ),
                self.currency,
                "Exchange",
                fx_fixings[-1],
            )
        )

    def npv(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the leg object via summing all periods.

        Calculates the cashflow for the all periods and multiplies them by the
        DF associated with each payment date.

        Parameters
        ----------
        args :
            Positional arguments supplied to :meth:`BasePeriod.npv`.
        kwargs :
            Keyword arguments supplied to :meth:`BasePeriod.npv`.

        Returns
        -------
        float, Dual, Dual2 or dict of such
        """
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().npv(curve, disc_curve, fx, base, local)
        # self._is_set_periods_fx = False
        return ret

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().cashflows(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        if not self._do_not_repeat_set_periods:
            self._set_periods(fx)
        ret = super().analytic_delta(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret


class FixedLegExchangeMtm(BaseLegExchangeMtm, FixedLegMixin):
    """
    Create a leg of :class:`~rateslib.periods.FixedPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
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
        Required keyword arguments to :class:`BaseLegExchange`.

    Notes
    -----

    .. warning::

       ``amortization`` is **not** permitted for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegExchangeMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

    The initial cashflow notional is set as the negative of the notional.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final period payment date (i.e. the end accrual date plus payment lag).

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods.
    """

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        fx_fixings: Optional[Union[list, float, Dual, Dual2]] = None,
        alt_currency: Optional[str] = None,
        alt_notional: Optional[float] = None,
        **kwargs,
    ):
        self._fixed_rate = fixed_rate
        # self._initialised = False
        super().__init__(
            *args,
            fx_fixings=fx_fixings,
            alt_currency=alt_currency,
            alt_notional=alt_notional,
            **kwargs,
        )
        # self._initialised = True

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

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
        )


class FloatLegExchangeMtm(BaseLegExchangeMtm, FloatLegMixin):
    """
    Create a leg of :class:`~rateslib.periods.FloatPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLegExchange`.
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
        Required keyword arguments to :class:`BaseLegExchange`.

    Notes
    -----

    .. warning::

       ``amortization`` is **not** permitted for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegExchangeMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

    .. note::

       ``currency`` and ``alt_currency`` are required in order to determine FX fixings
       from an :class:`~rateslib.fx.FXForwards` object at pricing time.

    The initial cashflow notional is set as the negative of the notional.

    The final cashflow notional is set as the end notional. The payment date is set
    equal to the final period payment date (i.e. the end accrual date plus
    ``payment_lag_exchange``).

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods.

    Examples
    --------
    .. ipython:: python

       usd_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
       eur_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
       fxr = FXRates({"usdeur": 0.98}, settlement=dt(2022, 1, 1))
       fxf = FXForwards(
           fxr, {"usdusd": usd_curve, "eureur": eur_curve, "eurusd": eur_curve}
       )

    .. ipython:: python

       float_leg_exch_mtm = FloatLegExchangeMtm(
           effective=dt(2022, 1, 1),
           termination="9M",
           frequency="Q",
           currency="eur",
           alt_currency="usd",
           alt_notional=1000000,
       )
       float_leg_exch_mtm.cashflows(curve, None, fxf)

    """

    def __init__(
        self,
        *args,
        float_spread: Union[float, None] = None,
        fixings: Optional[Union[float, list]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        fx_fixings: Optional[Union[list, float, Dual, Dual2]] = None,
        alt_currency: Optional[str] = None,
        alt_notional: Optional[float] = None,
        **kwargs,
    ):
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)
        # self._initialised = False  # flag for not calling fx_fixings in super()

        super().__init__(
            *args,
            fx_fixings=fx_fixings,
            alt_currency=alt_currency,
            alt_notional=alt_notional,
            **kwargs,
        )

        self._set_fixings(fixings)
        self.fx_fixings = fx_fixings  # sets fx_fixings and periods after initialising
        # self._initialised = True  # once rates fixings are set can set periods

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
            fixings=self.fixings[iterator],
            fixing_method=self.fixing_method,
            method_param=self.method_param,
            spread_compound_method=self.spread_compound_method,
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

       fp1 = FixedPeriod(dt(2021,1,1), dt(2021,7,1), dt(2021,7,2), "Q", 1e6, "Act365F", fixed_rate=2.10)
       fp2 = FixedPeriod(dt(2021,3,7), dt(2021,9,7), dt(2021,9,8), "Q", -5e6, "Act365F", fixed_rate=3.10)
       custom_leg = CustomLeg(periods=[fp1, fp2])
       custom_leg.cashflows(curve)

    """

    def __init__(self, periods):
        if not all(
            isinstance(p, (FixedPeriod, FloatPeriod, Cashflow)) for p in periods
        ):
            raise ValueError(
                "Each object in `periods` must be of type {FixedPeriod, FloatPeriod, "
                "Cashflow}."
            )
        self.periods = periods


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
