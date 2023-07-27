# -*- coding: utf-8 -*-

# Sphinx substitutions
"""
.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from rateslib.curves import Curve
   from datetime import datetime as dt
   from pandas import Series, date_range
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
from math import sqrt

import numpy as np

# from scipy.optimize import brentq
from pandas.tseries.offsets import CustomBusinessDay
from pandas import DataFrame, concat, date_range, Series

from rateslib import defaults
from rateslib.calendars import add_tenor, _add_days, get_calendar, dcf
from rateslib.scheduling import Schedule
from rateslib.curves import Curve, index_left, LineCurve, CompositeCurve, IndexCurve
from rateslib.solver import Solver
from rateslib.periods import Cashflow, FixedPeriod, FloatPeriod, _get_fx_and_base, IndexMixin
from rateslib.legs import (
    FixedLeg,
    FixedLegExchange,
    FloatLeg,
    FloatLegExchange,
    FloatLegExchangeMtm,
    FixedLegExchangeMtm,
    ZeroFloatLeg,
    ZeroFixedLeg,
    ZeroIndexLeg,
    IndexFixedLeg,
    IndexFixedLegExchange,
    CustomLeg,
)
from rateslib.dual import Dual, Dual2, set_order, DualTypes
from rateslib.fx import FXForwards, FXRates


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_curve_from_solver(curve, solver):
    if getattr(curve, "_is_proxy", False):
        # proxy curves exist outside of solvers but still have Dual variables associated
        # with curves inside the solver, so can still generate risks to calibrating
        # instruments
        return curve
    else:
        if isinstance(curve, str):
            return solver.pre_curves[curve]
        else:
            try:
                # it is a safeguard to load curves from solvers when a solver is
                # provided and multiple curves might have the same id
                return solver.pre_curves[curve.id]
            except KeyError:
                if defaults.curve_not_in_solver == "ignore":
                    return curve
                elif defaults.curve_not_in_solver == "warn":
                    warnings.warn("`curve` not found in `solver`.", UserWarning)
                    return curve
                else:
                    raise ValueError("`curve` must be in `solver`.")


def _get_curves_and_fx_maybe_from_solver(
    curves_attr: Optional[Union[Curve, str, list]],
    solver: Optional[Solver],
    curves: Optional[Union[Curve, str, list]],
    fx: Optional[Union[float, FXRates, FXForwards]],
):
    """
    Parses the ``solver``, ``curves`` and ``fx`` arguments in combination.

    Parameters
    ----------
    curves_attr
        The curves attribute attached to the class.
    solver
        The solver argument passed in the outer method.
    curves
        The curves argument passed in the outer method.
    fx
        The fx argument agrument passed in the outer method.

    Returns
    -------
    tuple : (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting), fx

    Notes
    -----
    If only one curve is given this is used as all four curves.

    If two curves are given the forecasting curve is used as the forecasting
    curve on both legs and the discounting curve is used as the discounting
    curve for both legs.

    If three curves are given the single discounting curve is used as the
    discounting curve for both legs.
    """
    if fx is None:
        if solver is None:
            fx_ = None
            # fx_ = 1.0
        elif solver is not None:
            if solver.fx is None:
                fx_ = None
                # fx_ = 1.0
            else:
                fx_ = solver.fx
    else:
        fx_ = fx

    if curves is None and curves_attr is None:
        return (None, None, None, None), fx_
    elif curves is None:
        curves = curves_attr

    if isinstance(curves, (Curve, str, CompositeCurve)):
        curves = [curves]
    if solver is None:

        def check_curve(curve):
            if isinstance(curve, str):
                raise ValueError(
                    "`curves` must contain Curve, not str, if `solver` not given."
                )
            return curve

        curves_ = tuple(check_curve(curve) for curve in curves)
    else:
        try:
            curves_ = tuple(_get_curve_from_solver(curve, solver) for curve in curves)
        except KeyError:
            raise ValueError(
                "`curves` must contain str curve `id` s existing in `solver` "
                "(or its associated `pre_solvers`)"
            )

    if len(curves_) == 1:
        curves_ *= 4
    elif len(curves_) == 2:
        curves_ *= 2
    elif len(curves_) == 3:
        curves_ += (curves_[1],)
    elif len(curves_) > 4:
        raise ValueError("Can only supply a maximum of 4 `curves`.")

    return curves_, fx_


# def _get_curves_and_fx_maybe_from_solver(
#     solver: Optional[Solver],
#     curves: Union[Curve, str, list],
#     fx: Optional[Union[float, FXRates, FXForwards]],
# ):
#     """
#     Parses the ``solver``, ``curves`` and ``fx`` arguments in combination.
#
#     Returns
#     -------
#     tuple : (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting), fx
#
#     Notes
#     -----
#     If only one curve is given this is used as all four curves.
#
#     If two curves are given the forecasting curve is used as the forecasting
#     curve on both legs and the discounting curve is used as the discounting
#     curve for both legs.
#
#     If three curves are given the single discounting curve is used as the
#     discounting curve for both legs.
#     """
#
#     if fx is None:
#         if solver is None:
#             fx_ = None
#             # fx_ = 1.0
#         elif solver is not None:
#             if solver.fx is None:
#                 fx_ = None
#                 # fx_ = 1.0
#             else:
#                 fx_ = solver.fx
#     else:
#         fx_ = fx
#
#     if curves is None:
#         return (None, None, None, None), fx_
#
#     if isinstance(curves, (Curve, str)):
#         curves = [curves]
#     if solver is None:
#         def check_curve(curve):
#             if isinstance(curve, str):
#                 raise ValueError(
#                     "`curves` must contain Curve, not str, if `solver` not given."
#                 )
#             return curve
#         curves_ = tuple(check_curve(curve) for curve in curves)
#     else:
#         try:
#             curves_ = tuple(_get_curve_from_solver(curve, solver) for curve in curves)
#         except KeyError:
#             raise ValueError(
#                 "`curves` must contain str curve `id` s existing in `solver` "
#                 "(or its associated `pre_solvers`)"
#             )
#
#     if len(curves_) == 1:
#         curves_ *= 4
#     elif len(curves_) == 2:
#         curves_ *= 2
#     elif len(curves_) == 3:
#         curves_ += (curves_[1],)
#     elif len(curves_) > 4:
#         raise ValueError("Can only supply a maximum of 4 `curves`.")
#
#     return curves_, fx_


class Sensitivities:
    """
    Base class to add risk sensitivity calculations to an object with an ``npv()``
    method.
    """

    def delta(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Calculate delta risk against the calibrating instruments of the
        :class:`~rateslib.curves.Curve`.

        Parameters
        ----------
        curves : Curve, str or list of such, optional
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.

        Returns
        -------
        DataFrame
        """
        if solver is None:
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv = self.npv(curves, solver, fx, base, local=True)
        _, fx = _get_curves_and_fx_maybe_from_solver(None, solver, None, fx)
        return solver.delta(npv, base, fx)

    def gamma(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Calculate cross-gamma risk against the calibrating instruments of the
        :class:`~rateslib.curves.Curve`.

        Parameters
        ----------
        curves : Curve, str or list of such, optional
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.

        Returns
        -------
        DataFrame
        """
        if solver is None:
            raise ValueError("`solver` is required for delta/gamma methods.")
        _, fx_ = _get_curves_and_fx_maybe_from_solver(None, solver, None, fx)

        # store original order
        if fx_ is not None:
            _ad2 = fx_._ad
            fx_._set_ad_order(2)

        _ad1 = solver._ad
        solver._set_ad_order(2)

        npv = self.npv(curves, solver, fx_, base, local=True)
        grad_s_sT_P = solver.gamma(npv, base, fx_)

        # reset original order
        if fx_ is not None:
            fx_._set_ad_order(_ad2)
        solver._set_ad_order(_ad1)

        return grad_s_sT_P


class BaseMixin:
    _fixed_rate_mixin = False
    _float_spread_mixin = False
    _leg2_fixed_rate_mixin = False
    _leg2_float_spread_mixin = False
    _index_base_mixin = False
    _leg2_index_base_mixin = False
    _rate_scalar = 1.0

    @property
    def fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of the contained
        leg1.

        .. note::
           ``fixed_rate``, ``float_spread``, ``leg2_fixed_rate`` and
           ``leg2_float_spread`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        if not self._fixed_rate_mixin:
            raise AttributeError("Cannot set `fixed_rate` for this Instrument.")
        self._fixed_rate = value
        self.leg1.fixed_rate = value

    @property
    def leg2_fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of the contained
        leg2.
        """
        return self._leg2_fixed_rate

    @leg2_fixed_rate.setter
    def leg2_fixed_rate(self, value):
        if not self._leg2_fixed_rate_mixin:
            raise AttributeError("Cannot set `leg2_fixed_rate` for this Instrument.")
        self._leg2_fixed_rate = value
        self.leg2.fixed_rate = value

    @property
    def float_spread(self):
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg1.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value):
        if not self._float_spread_mixin:
            raise AttributeError("Cannot set `float_spread` for this Instrument.")
        self._float_spread = value
        self.leg1.float_spread = value
        # if getattr(self, "_float_mixin_leg", None) is None:
        #     self.leg1.float_spread = value
        # else:
        #     # allows fixed_rate and float_rate to exist simultaneously for diff legs.
        #     leg = getattr(self, "_float_mixin_leg", None)
        #     getattr(self, f"leg{leg}").float_spread = value

    @property
    def leg2_float_spread(self):
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg2.
        """
        return self._leg2_float_spread

    @leg2_float_spread.setter
    def leg2_float_spread(self, value):
        if not self._leg2_float_spread_mixin:
            raise AttributeError("Cannot set `leg2_float_spread` for this Instrument.")
        self._leg2_float_spread = value
        self.leg2.float_spread = value

    @property
    def index_base(self):
        """
        float or None : If set will also set the ``index_base`` of the contained
        leg1.

        .. note::
           ``fixed_rate``, ``float_spread``, ``leg2_fixed_rate`` and
           ``leg2_float_spread`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        # TODO: re-write these docstrings to include index base mixin
        return self._index_base

    @index_base.setter
    def index_base(self, value):
        if not self._index_base_mixin:
            raise AttributeError("Cannot set `index_base` for this Instrument.")
        self._index_base = value
        self.leg1.index_base = value

    @property
    def leg2_index_base(self):
        """
        float or None : If set will also set the ``index_base`` of the contained
        leg1.

        .. note::
           ``fixed_rate``, ``float_spread``, ``leg2_fixed_rate`` and
           ``leg2_float_spread`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        # TODO: re-write these docstrings to include index base mixin
        return self._leg2_index_base

    @leg2_index_base.setter
    def leg2_index_base(self, value):
        if not self._leg2_index_base_mixin:
            raise AttributeError("Cannot set `leg2_index_base` for this Instrument.")
        self._leg2_index_base = value
        self.leg2.index_base = value


class Value(BaseMixin):
    """
    A null instrument which can be used within a :class:`~rateslib.solver.Solver`
    to directly parametrise a node.

    Parameters
    ----------
    effective : datetime
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    curves : Curve, LineCurve, str or list of such, optional
        A single :class:`~rateslib.curves.Curve`,
        :class:`~rateslib.curves.LineCurve` or id or a
        list of such. A list defines the following curves in the order:

        - Forecasting :class:`~rateslib.curves.Curve` or
          :class:`~rateslib.curves.LineCurve` for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
        - Forecasting :class:`~rateslib.curves.Curve` or
          :class:`~rateslib.curves.LineCurve` for ``leg2``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.

    Examples
    --------
    The below :class:`~rateslib.curves.Curve` is solved directly
    from a calibrating DF value on 1st Nov 2022.

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="v")
       instruments = [(Value(dt(2022, 11, 1)), (curve,), {})]
       solver = Solver([curve], instruments, [0.99])
       curve[dt(2022, 1, 1)]
       curve[dt(2022, 11, 1)]
       curve[dt(2023, 1, 1)]
    """

    def __init__(
        self,
        effective: datetime,
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.effective = effective
        self.curves = curves

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the forecasting :class:`~rateslib.curves.Curve` or
        :class:`~rateslib.curves.LineCurve` value on the ``effective`` date of the
        instrument.
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, None
        )
        return curves[0][self.effective]


### Securities


class BondMixin:
    def _set_base_index_if_none(self, curve: IndexCurve):
        if self._index_base_mixin and self.index_base is None:
            self.leg1.index_base = curve.index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )

    def ex_div(self, settlement: datetime):
        """
        Return a boolean whether the security is ex-div on the settlement.

        Parameters
        ----------
        settlement : datetime
             The settlement date to test.

        Returns
        -------
        bool
        """
        prev_a_idx = index_left(
            self.leg1.schedule.aschedule,
            len(self.leg1.schedule.aschedule),
            settlement,
        )
        ex_div_date = add_tenor(
            self.leg1.schedule.aschedule[prev_a_idx + 1],
            f"{-self.ex_div_days}B",
            None,  # modifier not required for business day tenor
            self.leg1.schedule.calendar,
        )
        return True if settlement >= ex_div_date else False

    def _accrued_frac(self, settlement: datetime):
        """
        Return the accrual fraction of period between last coupon and settlement and
        coupon period left index
        """
        acc_idx = index_left(
            self.leg1.schedule.aschedule,
            len(self.leg1.schedule.aschedule),
            settlement,
        )
        _ = (settlement - self.leg1.schedule.aschedule[acc_idx]) / (
            self.leg1.schedule.aschedule[acc_idx + 1]
            - self.leg1.schedule.aschedule[acc_idx]
        )
        return _, acc_idx

    def _npv_local(
        self,
        curve: Union[Curve, LineCurve],
        disc_curve: Curve,
        fx: Optional[Union[float, FXRates, FXForwards]],
        base: Optional[str],
        settlement: datetime,
        projection: datetime,
    ):
        """
        Return the NPV (local) of the security by summing cashflow valuations.

        Parameters
        ----------
        curve : Curve or LineCurve
            A curve used for projecting cashflows of floating rates.
        disc_curve : Curve, str or list of such
            A single :class:`Curve` for discounting cashflows.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        settlement : datetime
            The date of settlement of the bond which declares which cashflows are
            unpaid and therefore valid for the calculation.
        projection : datetime, optional
           Curves discount cashflows to the initial node of the Curve. This parameter
           allows the NPV to be projected forward to a future date under the appropriate
           discounting mechanism. If *None* is not projected forward.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        The cashflows for determination (excluding an ``ex_div`` cashflow) are
        evaluated by ``settlement``.

        The date for which the PV is returned is by ``projection``, and not the
        initial node date of the ``disc_curve``.
        """
        self._set_base_index_if_none(curve)
        npv = self.leg1.npv(curve, disc_curve, fx, base)

        # now must systematically deduct any cashflow between the initial node date
        # and the settlement date, including the cashflow after settlement if ex_div.
        initial_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            disc_curve.node_dates[0],
        )
        settle_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )

        for period_idx in range(initial_idx, settle_idx):
            # deduct coupon period
            npv -= self.leg1.periods[period_idx].npv(curve, disc_curve, fx, base)

        if self.ex_div(settlement):
            # deduct coupon after settlement which is also unpaid
            npv -= self.leg1.periods[settle_idx].npv(curve, disc_curve, fx, base)

        if projection is None:
            return npv
        else:
            return npv / disc_curve[projection]

    def _price_from_ytm(self, ytm: float, settlement: datetime, dirty: bool = False):
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        # TODO note this formula does not account for back stubs
        # this is also mentioned in Coding IRs

        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        v = 1 / (1 + ytm / (100 * f))

        acc_frac, acc_idx = self._accrued_frac(settlement)
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            fd0 = 1 - acc_frac

        d = 0
        for i, p_idx in enumerate(
                range(acc_idx, len(self.leg1.schedule.aschedule) - 1)
        ):
            if i == 0 and self.ex_div(settlement):
                continue
            else:
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v ** i
                # d += self.leg1.periods[p_idx].cashflow * v ** i
        d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v ** i
        p = v ** fd0 * d / -self.leg1.notional * 100
        return p if dirty else p - self.accrued(settlement)

    def price(self, ytm: float, settlement: datetime, dirty: bool = False):
        """
        Calculate the price of the security per nominal value of 100, given
        yield-to-maturity.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity against which to determine the price.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will include the
            :meth:`rateslib.instruments.FixedRateBond.accrued` in the price.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        This example is taken from the UK debt management office website.
        The result should be `141.070132` and the bond is ex-div.

        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ex_div(dt(1999, 5, 27))
           gilt.price(
               ytm=4.445,
               settlement=dt(1999, 5, 27),
               dirty=True
           )

        This example is taken from the Swedish national debt office website.
        The result of accrued should, apparently, be `0.210417` and the clean
        price should be `99.334778`.

        .. ipython:: python

           bond = FixedRateBond(
               effective=dt(2017, 5, 12),
               termination=dt(2028, 5, 12),
               frequency="A",
               calendar="stk",
               currency="sek",
               convention="ActActICMA",
               ex_div=5,
               fixed_rate=0.75
           )
           bond.ex_div(dt(2017, 8, 23))
           bond.accrued(dt(2017, 8, 23))
           bond.price(
               ytm=0.815,
               settlement=dt(2017, 8, 23),
               dirty=False
           )

        """
        return self._price_from_ytm(ytm, settlement, dirty)

    def duration(self, ytm: float, settlement: datetime, metric: str = "risk"):
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.
        metric : str
            The specific duration calculation to return. See notes.

        Returns
        -------
        float

        Notes
        -----
        The available metrics are:

        - *"risk"*: the derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = - \\frac{\partial P }{\partial y}

        - *"modified"*: the modified duration which is *risk* divided by price.

          .. math::

             mduration = \\frac{risk}{P} = - \\frac{1}{P} \\frac{\partial P }{\partial y}

        - *"duration"*: the duration which is modified duration reverse modified.

          .. math::

             duration = mduration \\times (1 + y / f)

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.duration(4.445, dt(1999, 5, 27), "risk")
           gilt.duration(4.445, dt(1999, 5, 27), "modified")
           gilt.duration(4.445, dt(1999, 5, 27), "duration")

        This result is interpreted as cents. If the yield is increased by 1bp the price
        will fall by 14.65 cents.

        .. ipython:: python

           gilt.price(4.445, dt(1999, 5, 27))
           gilt.price(4.455, dt(1999, 5, 27))
        """
        if metric == "risk":
            _ = -self.price(Dual(float(ytm), "y"), settlement).gradient("y")[0]
        elif metric == "modified":
            price = -self.price(Dual(float(ytm), "y"), settlement, dirty=True)
            _ = -price.gradient("y")[0] / float(price) * 100
        elif metric == "duration":
            price = -self.price(Dual(float(ytm), "y"), settlement, dirty=True)
            f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
            v = 1 + float(ytm) / (100 * f)
            _ = -price.gradient("y")[0] / float(price) * v * 100
        return _

    def convexity(self, ytm: float, settlement: datetime):
        """
        Return the second derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.

        Returns
        -------
        float

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.convexity(4.445, dt(1999, 5, 27))

        This number is interpreted as hundredths of a cent. For a 1bp increase in
        yield the duration will decrease by 2 hundredths of a cent.

        .. ipython:: python

           gilt.duration(4.445, dt(1999, 5, 27))
           gilt.duration(4.455, dt(1999, 5, 27))
        """
        return self.price(Dual2(float(ytm), "y"), settlement).gradient("y", 2)[0][0]

    def ytm(self, price: float, settlement: datetime, dirty: bool = False):
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ytm(
               price=141.0701315,
               settlement=dt(1999,5,27),
               dirty=True
           )
           gilt.ytm(Dual(141.0701315, ["price", "a", "b"], [1, -0.5, 2]), dt(1999, 5, 27), True)
           gilt.ytm(Dual2(141.0701315, ["price", "a", "b"], [1, -0.5, 2]), dt(1999, 5, 27), True)

        """

        def root(y):
            # we set this to work in float arithmetic for efficiency. Dual is added
            # back below, see PR GH3
            return self._price_from_ytm(y, settlement, dirty) - float(price)

        # x = brentq(root, -99, 10000)  # remove dependence to scipy.optimize.brentq
        # x, iters = _brents(root, -99, 10000)  # use own local brents code
        x = _ytm_quadratic_converger2(root, -3.0, 2.0, 12.0)  # use special quad interp

        if isinstance(price, Dual):
            # use the inverse function theorem to express x as a Dual
            p = self._price_from_ytm(Dual(x, "y"), settlement, dirty)
            return Dual(x, price.vars, 1 / p.gradient("y")[0] * price.dual)
        elif isinstance(price, Dual2):
            # use the IFT in 2nd order to express x as a Dual2
            p = self._price_from_ytm(Dual2(x, "y"), settlement, dirty)
            dydP = 1 / p.gradient("y")[0]
            d2ydP2 = -p.gradient("y", order=2)[0][0] * p.gradient("y")[0] ** -3
            return Dual2(
                x,
                price.vars,
                dydP * price.dual,
                0.5
                * (
                        dydP * price.gradient(price.vars, order=2)
                        + d2ydP2 * np.matmul(price.dual[:, None], price.dual[None, :])
                ),
            )
        else:
            return x

    def fwd_from_repo(
            self,
            price: Union[float, Dual, Dual2],
            settlement: datetime,
            forward_settlement: datetime,
            repo_rate: Union[float, Dual, Dual2],
            convention: Optional[str] = None,
            dirty: bool = False,
    ):
        """
        Return a forward price implied by a given repo rate.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        repo_rate : float, Dual or Dual2
            The rate which is used to calculate values.
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention = defaults.convention if convention is None else convention
        dcf_ = dcf(settlement, forward_settlement, convention)
        if not dirty:
            d_price = price + self.accrued(settlement)
        else:
            d_price = price
        if self.leg1.amortization != 0:
            raise NotImplementedError(  # pragma: no cover
                "method for forward price not available with amortization"
            )
        total_rtn = d_price * (1 + repo_rate * dcf_ / 100) * -self.leg1.notional / 100

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            dcf_ = dcf(self.leg1.periods[p_idx].payment, forward_settlement, convention)
            accrued_coup = self.leg1.periods[p_idx].cashflow * (
                    1 + dcf_ * repo_rate / 100
            )
            total_rtn -= accrued_coup

        forward_price = total_rtn / -self.leg1.notional * 100
        if dirty:
            return forward_price
        else:
            return forward_price - self.accrued(forward_settlement)

    def repo_from_fwd(
            self,
            price: Union[float, Dual, Dual2],
            settlement: datetime,
            forward_settlement: datetime,
            forward_price: Union[float, Dual, Dual2],
            convention: Optional[str] = None,
            dirty: bool = False,
    ):
        """
        Return an implied repo rate from a forward price.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        forward_price : float, Dual or Dual2
            The forward price which iplies the repo rate
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention = defaults.convention if convention is None else convention
        # forward price from repo is linear in repo_rate so reverse calculate with AD
        if not dirty:
            p_t = forward_price + self.accrued(forward_settlement)
            p_0 = price + self.accrued(settlement)
        else:
            p_t, p_0 = forward_price, price

        dcf_ = dcf(settlement, forward_settlement, convention)
        numerator = p_t - p_0
        denominator = p_0 * dcf_

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            dcf_ = dcf(self.leg1.periods[p_idx].payment, forward_settlement, convention)
            numerator += 100 * self.leg1.periods[p_idx].cashflow / -self.leg1.notional
            denominator -= (
                    100 * dcf_ * self.leg1.periods[p_idx].cashflow / -self.leg1.notional
            )

        return numerator / denominator * 100

    def accrued(self, settlement: datetime):
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.

        Notes
        -----
        Fractionally apportions the coupon payment based on calendar days.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        """
        # TODO validate against effective and termination?
        frac, acc_idx = self._accrued_frac(settlement)
        if self.ex_div(settlement):
            frac = frac - 1  # accrued is negative in ex-div period
        _ = getattr(self.leg1.periods[acc_idx], self._ytm_attribute)
        return frac * _ / -self.leg1.notional * 100

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the security by summing cashflow valuations.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        local : bool, optional
            If `True` will ignore the ``base`` request and return a dict identifying
            local currency NPV.

        Returns
        -------
        float, Dual, Dual2 or dict of such

        Notes
        -----
        The ``settlement`` date of the bond is inferred from the objects ``settle``
        days parameter and the initial date of the supplied ``curves``.
        The NPV returned is for immediate settlement.

        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        settlement = add_tenor(
            curves[1].node_dates[0],
            f"{self.settle}B",
            None,
            self.leg1.schedule.calendar,
        )
        base = self.leg1.currency if local else base
        npv = self._npv_local(curves[0], curves[1], fx, base, settlement, None)
        if local:
            return {self.leg1.currency: npv}
        else:
            return npv

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        """
        Return the analytic delta of the security via summing all periods.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve = disc_curve or curve
        settlement = add_tenor(
            disc_curve.node_dates[0],
            f"{self.settle}B",
            None,
            self.leg1.schedule.calendar,
        )
        a_delta = self.leg1.analytic_delta(curve, disc_curve, fx, base)
        if self.ex_div(settlement):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement,
            )
            a_delta -= self.leg1.periods[current_period].analytic_delta(
                curve, disc_curve, fx, base
            )
        return a_delta

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        settlement: datetime = None,
    ):
        """
        Return the properties of the security used in calculating cashflows.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an ``FXRates`` or ``FXForwards`` object.
        settlement : datetime, optional
            The settlement date of the security. If *None* adds the regular ``settle``
            time to the initial node date of the given discount ``curves``.

        Returns
        -------
        DataFrame
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        self._set_base_index_if_none(curves[0])

        if settlement is None:
            settlement = add_tenor(
                curves[1].node_dates[0],
                f"{self.settle}B",
                None,
                self.leg1.schedule.calendar,
            )
        cashflows = self.leg1.cashflows(curves[0], curves[1], fx, base)
        if self.ex_div(settlement):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement,
            )
            cashflows.loc[current_period, defaults.headers["npv"]] = 0
            cashflows.loc[current_period, defaults.headers["npv_fx"]] = 0
        return cashflows


class FixedRateBond(Sensitivities, BondMixin, BaseMixin):
    # TODO ensure calculations work for amortizing bonds.
    """
    Create a fixed rate bond security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is not permitted.
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
    fixed_rate : float, optional
        The **coupon** rate applied to determine cashflows. Can be set
        to `None` and designated
        later, perhaps after a mid-market rate for all periods has been calculated.
    ex_div : int
        The number of days prior to a cashflow during which the bond is considered
        ex-dividend.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.

    Attributes
    ----------
    ex_div_days : int
    settle : int
    curves : str, list, CurveType
    leg1 : FixedLegExchange

    Examples
    --------
    This example is taken from the UK debt management office (DMO) website. A copy of
    which is available :download:`here<_static/ukdmoyldconv.pdf>`.

    We demonstrate the use of **analogue methods** which do not need *Curves* or
    *Solvers*,
    :meth:`~rateslib.instruments.FixedRateBond.price`,
    :meth:`~rateslib.instruments.FixedRateBond.ytm`,
    :meth:`~rateslib.instruments.FixedRateBond.ex_div`,
    :meth:`~rateslib.instruments.FixedRateBond.accrued`,
    :meth:`~rateslib.instruments.FixedRateBond.repo_from_fwd`
    :meth:`~rateslib.instruments.FixedRateBond.fwd_from_repo`
    :meth:`~rateslib.instruments.FixedRateBond.duration`,
    :meth:`~rateslib.instruments.FixedRateBond.convexity`.

    .. ipython:: python

       gilt = FixedRateBond(
           effective=dt(1998, 12, 7),
           termination=dt(2015, 12, 7),
           frequency="S",
           calendar="ldn",
           currency="gbp",
           convention="ActActICMA",
           ex_div=7,
           settle=1,
           fixed_rate=8.0,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bond
           curves="gilt_curve",
       )
       gilt.ex_div(dt(1999, 5, 27))
       gilt.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=True)
       gilt.ytm(price=141.070132, settlement=dt(1999, 5, 27), dirty=True)
       gilt.accrued(dt(1999, 5, 27))
       gilt.fwd_from_repo(
           price=141.070132,
           settlement=dt(1999, 5, 27),
           forward_settlement=dt(2000, 2, 27),
           repo_rate=4.5,
           convention="Act365F",
           dirty=True,
       )
       gilt.repo_from_fwd(
           price=141.070132,
           settlement=dt(1999, 5, 27),
           forward_settlement=dt(2000, 2, 27),
           forward_price=141.829943,
           convention="Act365F",
           dirty=True,
       )
       gilt.duration(settlement=dt(1999, 5, 27), ytm=4.445, metric="risk")
       gilt.duration(settlement=dt(1999, 5, 27), ytm=4.445, metric="modified")
       gilt.convexity(settlement=dt(1999, 5, 27), ytm=4.445)


    The following **digital methods** consistent with the library's ecosystem are
    also available,
    :meth:`~rateslib.instruments.FixedRateBond.analytic_delta`,
    :meth:`~rateslib.instruments.FixedRateBond.rate`,
    :meth:`~rateslib.instruments.FixedRateBond.npv`,
    :meth:`~rateslib.instruments.FixedRateBond.cashflows`,
    :meth:`~rateslib.instruments.FixedRateBond.delta`,
    :meth:`~rateslib.instruments.FixedRateBond.gamma`.

    .. ipython:: python

       gilt_curve = Curve({dt(1999, 5, 26): 1.0, dt(2019, 5, 26): 1.0}, id="gilt_curve")
       instruments = [
           (gilt, (), {"metric": "ytm"}),
       ]
       solver = Solver(
           curves=[gilt_curve],
           instruments=instruments,
           s=[4.445],
           instrument_labels=["8% Dec15"],
           id="gilt_solver",
       )
       gilt.npv(solver=solver)
       gilt.analytic_delta(disc_curve=gilt_curve)
       gilt.rate(solver=solver, metric="clean_price")

    The sensitivities are also available. In this case the *Solver* is calibrated
    with *instruments* priced in yield terms so sensitivities are measured in basis
    points (bps).

    .. ipython:: python

       gilt.delta(solver=solver)
       gilt.gamma(solver=solver)

    The DataFrame of cashflows.

    .. ipython:: python

       gilt.cashflows(solver=solver)

    """
    _fixed_rate_mixin = True
    _ytm_attribute = "cashflow"  # nominal bonds use cashflows in YTM calculation

    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str] = None,
        frequency: str = None,
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
        fixed_rate: Optional[float] = None,
        ex_div: int = 0,
        settle: int = 1,
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.curves = curves
        if frequency.lower() == "z":
            raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")
        if payment_lag is None:
            payment_lag = defaults.payment_lag_specific[type(self).__name__]
        self._fixed_rate = fixed_rate
        self.ex_div_days = ex_div
        self.settle = settle
        self.leg1 = FixedLegExchange(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=False,
        )
        if self.leg1.amortization != 0:
            # Note if amortization is added to FixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FixedRateBond must be zero.")

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    # Digital Methods

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        metric: str = "clean_price",
        forward_settlement: Optional[datetime] = None,
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "ytm", "fwd_clean_price", "fwd_dirty_price"}
        forward_settlement : datetime
            The forward settlement date, required if the metric is in
            {"fwd_clean_price", "fwd_dirty_price"}.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "ytm"]:
            settlement = add_tenor(
                curves[1].node_dates[0],
                f"{self.settle}B",
                None,
                self.leg1.schedule.calendar,
            )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, settlement, settlement
            )
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "ytm":
                return self.ytm(dirty_price, settlement, True)

        elif metric in ["fwd_clean_price", "fwd_dirty_price"]:
            if forward_settlement is None:
                raise ValueError(
                    "`forward_settlement` needed to determine forward price."
                )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, forward_settlement, forward_settlement
            )
            dirty_price = npv / -self.leg1.notional * 100
            if metric == "fwd_dirty_price":
                return dirty_price
            elif metric == "fwd_clean_price":
                return dirty_price - self.accrued(forward_settlement)

        raise ValueError(
            "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
            "'fwd_clean_price', 'fwd_dirty_price'}."
        )

    # def par_spread(self, *args, price, settlement, dirty, **kwargs):
    #     """
    #     The spread to the fixed rate added to value the security at par valued from
    #     the given :class:`~rateslib.curves.Curve` s.
    #
    #     Parameters
    #     ----------
    #     args: tuple
    #         Positional arguments to :meth:`~rateslib.periods.BasePeriod.npv`.
    #     price: float
    #         The price of the security.
    #     settlement : datetime
    #         The settlement date.
    #     dirty : bool
    #         Whether the price given includes accrued interest.
    #     kwargs : dict
    #         Keyword arguments to :meth:`~rateslib.periods.BasePeriod.npv`.
    #
    #     Returns
    #     -------
    #     float, Dual, Dual2
    #     """
    #     TODO: calculate this par_spread formula.
    #     return (self.notional - self.npv(*args, **kwargs)) / self.analytic_delta(*args, **kwargs)


class IndexFixedRateBond(Sensitivities, BondMixin, BaseMixin):
    _fixed_rate_mixin = True
    _ytm_attribute = "real_cashflow"  # index linked bonds use real cashflows
    _index_base_mixin = True

    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str] = None,
        frequency: str = None,
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
        fixed_rate: Optional[float] = None,
        index_base: Optional[Union[float, Series]] = None,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: Optional[str] = None,
        index_lag: Optional[int] = None,
        ex_div: int = 0,
        settle: int = 1,
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.curves = curves
        if frequency.lower() == "z":
            raise ValueError(
                "IndexFixedRateBond `frequency` must be in {M, B, Q, T, S, A}."
            )
        if payment_lag is None:
            payment_lag = defaults.payment_lag_specific[type(self).__name__]
        self._fixed_rate = fixed_rate
        self._index_base = index_base
        self.ex_div_days = ex_div
        self.settle = settle
        self.leg1 = IndexFixedLegExchange(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=False,
            index_base=index_base,
            index_method=index_method,
            index_lag=index_lag,
            index_fixings=index_fixings,
        )
        if self.leg1.amortization != 0:
            # Note if amortization is added to FixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FixedRateBond must be zero.")

    def index_ratio(self, settlement: datetime, curve: Optional[IndexCurve]):
        if self.leg1.index_fixings is not None \
                and not isinstance(self.leg1.index_fixings, Series):
            raise ValueError(
                "Must provide `index_fixings` as a Series for inter-period settlement."
            )
        # TODO: this indexing of periods assumes no amortization
        index_val = IndexMixin._index_value(
            i_fixings=self.leg1.index_fixings,
            i_curve=curve,
            i_lag=self.leg1.index_lag,
            i_method=self.leg1.index_method,
            i_date=settlement,
        )
        index_base = IndexMixin._index_value(
            i_fixings=self.index_base,
            i_date=self.leg1.schedule.effective,
            i_lag=self.leg1.index_lag,
            i_method=self.leg1.index_method,
            i_curve=curve
        )
        return index_val / index_base

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        metric: str = "clean_price",
        forward_settlement: Optional[datetime] = None,
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "ytm", "fwd_clean_price", "fwd_dirty_price"}
        forward_settlement : datetime
            The forward settlement date, required if the metric is in
            {"fwd_clean_price", "fwd_dirty_price"}.

        Returns
        -------
        float, Dual, Dual2
        """

        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )

        metric = metric.lower()
        if metric in [
            "clean_price",
            "dirty_price",
            "index_clean_price",
            "ytm",
            "index_dirty_price",
        ]:
            settlement = add_tenor(
                curves[1].node_dates[0],
                f"{self.settle}B",
                None,
                self.leg1.schedule.calendar,
            )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, settlement, settlement
            )
            # scale price to par 100 (npv is already projected forward to settlement)
            index_dirty_price = npv * 100 / -self.leg1.notional
            index_ratio = self.index_ratio(settlement, curves[0])
            dirty_price = index_dirty_price / index_ratio

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "ytm":
                return self.ytm(dirty_price, settlement, True)
            elif metric == "index_dirty_price":
                return index_dirty_price
            elif metric == "index_clean_price":
                return index_dirty_price - self.accrued(settlement) * index_ratio

        elif metric in [
            "fwd_clean_price",
            "fwd_dirty_price",
            "fwd_index_clean_price",
            "fwd_index_dirty_price",
        ]:
            if forward_settlement is None:
                raise ValueError(
                    "`forward_settlement` needed to determine forward price."
                )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, forward_settlement, forward_settlement
            )
            index_dirty_price = npv / -self.leg1.notional * 100
            index_ratio = self.index_ratio(forward_settlement, curves[0])
            dirty_price = index_dirty_price / index_ratio
            if metric == "fwd_dirty_price":
                return dirty_price
            elif metric == "fwd_clean_price":
                return dirty_price - self.accrued(forward_settlement)
            elif metric == "fwd_index_dirty_price":
                return index_dirty_price
            elif metric == "fwd_index_clean_price":
                return (
                    index_dirty_price - self.accrued(forward_settlement) * index_ratio
                )

        raise ValueError(
            "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
            "'fwd_clean_price', 'fwd_dirty_price', 'index_dirty_price', "
            "'index_clean_price', 'fwd_index_dirty_price', 'fwd_index_clean_price'}."
        )


class Bill(FixedRateBond):
    """
    Create a discount security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is not permitted.
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
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.

    Examples
    --------
    This example is taken from the US Treasury Federal website. A copy of
    which is available :download:`here<_static/ofcalc6decTbill.pdf>`.

    We demonstrate the use of **analogue methods** which do not need *Curves* or
    *Solvers*,
    :meth:`~rateslib.instruments.Bill.price`,
    :meth:`~rateslib.instruments.Bill.simple_rate`,
    :meth:`~rateslib.instruments.Bill.discount_rate`,
    :meth:`~rateslib.instruments.FixedRateBond.ytm`,
    :meth:`~rateslib.instruments.FixedRateBond.ex_div`,
    :meth:`~rateslib.instruments.FixedRateBond.accrued`,
    :meth:`~rateslib.instruments.FixedRateBond.repo_from_fwd`
    :meth:`~rateslib.instruments.FixedRateBond.fwd_from_repo`
    :meth:`~rateslib.instruments.FixedRateBond.duration`,
    :meth:`~rateslib.instruments.FixedRateBond.convexity`.

    .. ipython:: python

       bill = Bill(
           effective=dt(2004, 1, 22),
           termination=dt(2004, 2, 19),
           frequency="M",
           calendar="nyc",
           modifier="MF",
           currency="usd",
           convention="Act360",
           settle=1,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bill
           curves="bill_curve",
       )
       bill.ex_div(dt(2004, 1, 22))
       bill.price(discount_rate=0.80, settlement=dt(2004, 1, 22))
       bill.simple_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.discount_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.ytm(price=99.937778, settlement=dt(2004, 1, 22))
       bill.accrued(dt(2004, 1, 22))
       bill.fwd_from_repo(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           repo_rate=0.8005,
           convention="Act360",
       )
       bill.repo_from_fwd(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           forward_price=100.00,
           convention="Act360",
       )
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="risk")
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="modified")
       bill.convexity(settlement=dt(2004, 1, 22), ytm=0.8005)


    The following **digital methods** consistent with the library's ecosystem are
    also available,
    :meth:`~rateslib.instruments.Bill.rate`,
    :meth:`~rateslib.instruments.FixedRateBond.npv`,
    :meth:`~rateslib.instruments.FixedRateBond.analytic_delta`,
    :meth:`~rateslib.instruments.FixedRateBond.cashflows`,
    :meth:`~rateslib.instruments.FixedRateBond.delta`,
    :meth:`~rateslib.instruments.FixedRateBond.gamma`,

    .. ipython:: python

       bill_curve = Curve({dt(2004, 1, 21): 1.0, dt(2004, 3, 21): 1.0}, id="bill_curve")
       instruments = [
           (bill, (), {"metric": "ytm"}),
       ]
       solver = Solver(
           curves=[bill_curve],
           instruments=instruments,
           s=[0.8005],
           instrument_labels=["Feb04 Tbill"],
           id="bill_solver",
       )
       bill.npv(solver=solver)
       bill.analytic_delta(disc_curve=bill_curve)
       bill.rate(solver=solver, metric="price")

    The sensitivities are also available. In this case the *Solver* is calibrated
    with *instruments* priced in yield terms so sensitivities are measured in basis
    points (bps).

    .. ipython:: python

       bill.delta(solver=solver)
       bill.gamma(solver=solver)

    The DataFrame of cashflows.

    .. ipython:: python

       bill.cashflows(solver=solver)

    """

    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str] = None,
        frequency: str = None,
        modifier: Optional[str] = False,
        calendar: Optional[Union[CustomBusinessDay, str]] = None,
        payment_lag: Optional[int] = None,
        notional: Optional[float] = None,
        currency: Optional[str] = None,
        convention: Optional[str] = None,
        settle: int = 1,
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        if payment_lag is None:
            payment_lag = defaults.payment_lag_specific[type(self).__name__]
        super().__init__(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=None,
            front_stub=None,
            back_stub=None,
            roll=None,
            eom=None,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=None,
            convention=convention,
            fixed_rate=0,
            ex_div=0,
            settle=settle,
            curves=curves,
        )

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        metric="price",
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str in {"price", "discount_rate", "ytm", "simple_rate"}
            Metric returned by the method.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        settlement = add_tenor(
            curves[1].node_dates[0],
            f"{self.settle}B",
            None,
            self.leg1.schedule.calendar,
        )
        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves, solver, fx, base)
            * 100
            / (-self.leg1.notional * curves[1][settlement])
        )
        if metric in ["price", "clean_price"]:
            return price
        elif metric == "discount_rate":
            return self.discount_rate(price, settlement)
        elif metric == "simple_rate":
            return self.simple_rate(price, settlement)
        elif metric == "ytm":
            return self.ytm(price, settlement, False)
        raise ValueError(
            "`metric` must be in {'price', 'discount_rate', 'ytm', 'simple_rate'}"
        )

    def simple_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the simple rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        dcf = (1 - self._accrued_frac(settlement)[0]) * self.leg1.periods[0].dcf
        return ((100 / price - 1) / dcf) * 100

    def discount_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the discount rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        dcf = (1 - self._accrued_frac(settlement)[0]) * self.leg1.periods[0].dcf
        rate = ((1 - price / 100) / dcf) * 100
        return rate

    def price(
        self, discount_rate: DualTypes, settlement: datetime, dirty: bool = False
    ) -> DualTypes:
        """
        Return the price of the bill given the ``discount_rate``.

        Parameters
        ----------
        discount_rate : float
            The rate used by the pricing formula.
        settlement : datetime
            The settlement date.
        dirty : bool, not required
            Discount securities have no coupon, the concept of clean or dirty is not
            relevant. Argument is included for signature consistency with
            :meth:`FixedRateBond.price<rateslib.instruments.FixedRateBond.price>`.

        Returns
        -------
        float, Dual, Dual2
        """
        dcf = (1 - self._accrued_frac(settlement)[0]) * self.leg1.periods[0].dcf
        return 100 - discount_rate * dcf


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatRateBond(Sensitivities, BondMixin, BaseMixin):
    """
    Create a floating rate bond security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is not permitted.
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
    float_spread : float, optional
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    ex_div : int
        The number of days prior to a cashflow during which the bond is considered
        ex-dividend.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.

    Notes
    -----
    .. warning::

       FRNs based on RFR rates which have ex-div days must ensure that fixings are
       available to define the entire period. This means that `ex_div` days must be less
       than the `fixing_method` `method_param` lag minus the time to settlement time.

        That is, a bond with a `method_param` of 5 and a settlement time of 2 days
        can have an `ex_div` period of at maximum 3.

        A bond with a `method_param` of 2 and a settlement time of 1 day cnan have an
        `ex_div` period of at maximum 1.

    Attributes
    ----------
    ex_div_days : int
    leg1 : FloatLegExchange
    """

    _float_spread_mixin = True

    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str] = None,
        frequency: str = None,
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
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        ex_div: int = 0,
        settle: int = 1,
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.curves = curves
        if frequency.lower() == "z":
            raise ValueError("FloatRateBond `frequency` must be in {M, B, Q, T, S, A}.")
        if payment_lag is None:
            payment_lag = defaults.payment_lag_specific[type(self).__name__]
        self._float_spread = float_spread
        self.leg1 = FloatLegExchange(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            initial_exchange=False,
        )
        self.ex_div_days = ex_div
        self.settle = settle
        if "rfr" in self.leg1.fixing_method:
            if self.ex_div_days > self.leg1.method_param:
                raise ValueError(
                    "For RFR FRNs `ex_div` must be less than or equal to `method_param`"
                    " otherwise negative accrued payments cannot be explicitly "
                    "determined due to unknown fixings."
                )

    def accrued(
        self,
        settlement: datetime,
        forecast: bool = False,
        curve: Curve = None,
    ):
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.
        forecast : bool, optional
            Whether to use a curve to forecast future fixings.
        curve : Curve, optional
            If ``forecast`` is *True* and fixings are future based then must provide
            a forecast curve.

        Notes
        -----
        The settlement of an FRN will always be a definite amount. The
        ``fixing_method``, ``method_param`` and ``ex_div`` will contain a
        valid combination of parameters such that when payments need to be
        cleared these definitive amounts can be calculated
        via previously published fixings.

        If the coupon is IBOR based then the accrued
        fractionally apportions the coupon payment based on calendar days, including
        negative accrued during ex div periods. This rarely poses a problem since
        IBOR is fixed well in advance of settlement.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        With RFR rates, however, and since ``settlement`` typically occurs
        in the future, e.g. T+2, it may be
        possible, particularly if the bond is *ex-div* that some fixings are not known
        today, but they will be known by ``settlement``. This is also true if we
        wish to calculate the forward dirty price of a bond and need to forecast
        the accrued amount (and also for a forecast IBOR period).

        Thus, there are two options:

        - In the analogue mode where very few fixings might be missing, and we require
          these values to calculate negative accrued in an ex-div period we do not
          require a ``curve`` but repeat the last historic fixing.
        - In the digital mode where the ``settlement`` may be well in the future we
          use a ``curve`` to forecast rates,

        Examples
        --------
        An RFR based FRN where the fixings are known up to the end of period.

        .. ipython:: python

           fixings = Series(2.0, index=date_range(dt(1999, 12, 1), dt(2000, 6, 2)))
           frn = FloatRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="ActActICMA",
               ex_div=3,
               fixings=fixings,
               fixing_method="rfr_observation_shift",
               method_param=5,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))


        An IBOR based FRN where the coupon is known in advance.

        .. ipython:: python

           fixings = Series(2.0, index=[dt(1999, 12, 5)])
           frn = FloatRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixings=fixings,
               fixing_method="ibor",
               method_param=2,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))
        """
        if self.leg1.fixing_method == "ibor":
            frac, acc_idx = self._accrued_frac(settlement)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period

            if forecast:
                curve = curve
            else:
                curve = Curve(
                    {  # create a dummy curve. rate() will return the fixing
                        self.leg1.periods[acc_idx].start: 1.0,
                        self.leg1.periods[acc_idx].end: 1.0,
                    }
                )
            rate = self.leg1.periods[acc_idx].rate(curve)

            cashflow = (
                -self.leg1.periods[acc_idx].notional
                * self.leg1.periods[acc_idx].dcf
                * rate
                / 100
            )
            return frac * cashflow / -self.leg1.notional * 100
        else:  # is "rfr"
            acc_idx = index_left(
                self.leg1.schedule.aschedule,
                len(self.leg1.schedule.aschedule),
                settlement,
            )
            p = FloatPeriod(
                start=self.leg1.schedule.aschedule[acc_idx],
                end=settlement,
                payment=settlement,
                frequency=self.leg1.schedule.frequency,
                notional=-100,
                currency=self.leg1.currency,
                convention=self.leg1.convention,
                termination=self.leg1.schedule.aschedule[acc_idx + 1],
                stub=True,
                float_spread=self.float_spread,
                fixing_method=self.leg1.fixing_method,
                fixings=self.leg1.fixings[acc_idx],
                method_param=self.leg1.method_param,
                spread_compound_method=self.leg1.spread_compound_method,
            )

            if forecast:
                curve = curve
            else:
                try:
                    last_fixing = p.fixings[-1]
                    # For negative accr in ex-div we need to forecast unpublished rates.
                    # Build a curve which replicates the last fixing value from fixings.
                except TypeError:
                    # then rfr fixing cannot be fetched from attribute

                    # if acc_idx == 0 and p.start == self.leg1.schedule.aschedule[0]:
                    #     # bond settles on issue date of bond, fixing may not be available.
                    #     accrued_to_settle = 0.

                    if p.dcf < 1e-10:
                        # then settlement is same as period.start so no rate necessary
                        # create a dummy curve
                        last_fixing = 0.0
                    else:
                        raise TypeError(
                            "`fixings` are not available for RFR float period. Must be a "
                            f"Series or list, {p.fixings} was given."
                        )
                curve = LineCurve(
                    {
                        self.leg1.periods[acc_idx].start: last_fixing,
                        self.leg1.periods[acc_idx].end: last_fixing,
                    }
                )

            # Otherwise rate to settle is determined fully by known fixings.
            if p.dcf < 1e-10:
                rate_to_settle = 0.0  # there are no fixings in the period.
            else:
                rate_to_settle = float(p.rate(curve))
            accrued_to_settle = 100 * p.dcf * rate_to_settle / 100

            if self.ex_div(settlement):
                rate_to_end = self.leg1.periods[acc_idx].rate(curve)
                accrued_to_end = (
                    100 * self.leg1.periods[acc_idx].dcf * rate_to_end / 100
                )
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle

    def rate(
        self,
        curves: Union[Curve, str, list],
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        metric="clean_price",
        forward_settlement: Optional[datetime] = None,
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "spread", "fwd_clean_price", "fwd_dirty_price"}
        forward_settlement : datetime
            The forward settlement date, required if the metric is in
            {"fwd_clean_price", "fwd_dirty_price"}.

        Returns
        -------
        float, Dual, Dual2

        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "spread"]:
            settlement = add_tenor(
                curves[1].node_dates[0],
                f"{self.settle}B",
                None,
                self.leg1.schedule.calendar,
            )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, settlement, settlement
            )
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "spread":
                _ = self.leg1._spread(-(npv + self.leg1.notional), curves[0], curves[1])
                z = 0.0 if self.float_spread is None else self.float_spread
                return _ + z

        elif metric in ["fwd_clean_price", "fwd_dirty_price"]:
            if forward_settlement is None:
                raise ValueError(
                    "`forward_settlement` needed to determine forward price."
                )
            npv = self._npv_local(
                curves[0], curves[1], fx, base, forward_settlement, forward_settlement
            )
            dirty_price = npv / -self.leg1.notional * 100
            if metric == "fwd_dirty_price":
                return dirty_price
            elif metric == "fwd_clean_price":
                return dirty_price - self.accrued(forward_settlement, True, curves[0])

        raise ValueError(
            "`metric` must be in {'dirty_price', 'clean_price', 'spread'}."
        )


### Single currency derivatives


class BondFuture(Sensitivities):
    """
    Create a bond future derivative.

    Parameters
    ----------
    coupon: float
        The nominal coupon rate set on the contract specifications.
    delivery: datetime or 2-tuple of datetimes
        The delivery window first and last delivery day, or a single delivery day.
    basket: tuple of FixedRateBond
        The bonds that are available as deliverables.
    nominal: float, optional
        The nominal amount of the contract.
    contracts: int, optional
        The number of contracts owned or short.
    calendar: str, optional
        The calendar to define delivery days within the delivery window.
    currency: str, optional
        The currency (3-digit code) of the settlement contract.

    Examples
    --------
    The :meth:`~rateslib.instruments.BondFuture.dlv` method is a summary method which
    displays many attributes simultaneously in a DataFrame.
    This example replicates the Bloomberg screen print in the publication
    *The Futures Bond Basis: Second Edition (p77)* by Moorad Choudhry. To replicate
    that publication exactly no calendar has been provided. A more modern
    Bloomberg would probably consider the London business day calendar and
    this would affect the metrics of the third bond to a small degree (i.e.
    set `calendar="ldn"`)

    .. ipython:: python

       kws = dict(
           frequency="S",
           ex_div=7,
           convention="ActActICMA",
           calendar=None,
           currency="gbp",
           settle=1,
           curves="gilt_curve"
       )
       bonds = [
           FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
           FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
       ]
       prices=[102.732, 131.461, 107.877, 134.455]
       ytms=[bond.ytm(price, dt(2000, 3, 16)) for bond, price in zip(bonds, prices)]
       future = BondFuture(
           delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
           coupon=7.0,
           basket=bonds,
           nominal=100000,
           contracts=10,
           currency="gbp",
       )
       future.dlv(
           future_price=112.98,
           prices=[102.732, 131.461, 107.877, 134.455],
           repo_rate=6.24,
           settlement=dt(2000, 3, 16),
           convention="Act365f",
       )

    Various other metrics can be extracted in isolation including,
    ``notional``, and conversion factors (``cfs``),
    :meth:`~rateslib.instruments.BondFuture.gross_basis`,
    :meth:`~rateslib.instruments.BondFuture.net_basis`,
    :meth:`~rateslib.instruments.BondFuture.implied_repo`,
    :meth:`~rateslib.instruments.BondFuture.ytm`,
    :meth:`~rateslib.instruments.BondFuture.duration`,
    :meth:`~rateslib.instruments.BondFuture.convexity`,
    :meth:`~rateslib.instruments.BondFuture.ctd_index`,

    .. ipython:: python

        future.cfs
        future.notional
        future.gross_basis(
            future_price=112.98,
            prices=prices,
        )
        future.net_basis(
            future_price=112.98,
            prices=prices,
            repo_rate=6.24,
            settlement=dt(2000, 3, 16),
            delivery=dt(2000, 6, 30),
            convention="Act365f"
        )
        future.implied_repo(
            future_price=112.98,
            prices=prices,
            settlement=dt(2000, 3, 16)
        )
        future.ytm(future_price=112.98)
        future.duration(future_price=112.98)
        future.convexity(future_price=112.98)
        future.ctd_index(
            future_price=112.98,
            prices=prices,
            settlement=dt(2000, 3, 16)
        )

    As opposed to the **analogue methods** above, we can also use
    the **digital methods**,
    :meth:`~rateslib.instruments.BondFuture.npv`,
    :meth:`~rateslib.instruments.BondFuture.rate`,
    but we need to create *Curves* and a *Solver* in the usual way.

    .. ipython:: python

       gilt_curve = Curve(
           nodes={
               dt(2000, 3, 15): 1.0,
               dt(2009, 12, 7): 1.0,
               dt(2010, 11, 25): 1.0,
               dt(2011, 7, 12): 1.0,
               dt(2012, 8, 6): 1.0,
           },
           id="gilt_curve",
       )
       solver = Solver(
           curves=[gilt_curve],
           instruments=[(b, (), {"metric": "ytm"}) for b in bonds],
           s=ytms,
           id="gilt_solver",
           instrument_labels=["5.75% '09", "9% '11", "6.25% '10", "9% '12"],
       )

    Sensitivities are also available;
    :meth:`~rateslib.instruments.BondFuture.delta`
    :meth:`~rateslib.instruments.BondFuture.gamma`.

    .. ipython:: python

       future.delta(solver=solver)

    The delta of a *BondFuture* is individually assigned to the CTD. If the CTD changes
    the delta is reassigned.

    .. ipython:: python

       solver.s = [5.3842, 5.2732, 5.2755, 5.52]
       solver.iterate()
       future.delta(solver=solver)
       future.gamma(solver=solver)

    """

    def __init__(
        self,
        coupon: float,
        delivery: Union[datetime, tuple[datetime, datetime]],
        basket: tuple[FixedRateBond],
        # last_trading: Optional[int] = None,
        nominal: Optional[float] = None,
        contracts: Optional[int] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
    ):
        self.currency = defaults.base_currency if currency is None else currency.lower()
        self.coupon = coupon
        if isinstance(delivery, datetime):
            self.delivery = (delivery, delivery)
        else:
            self.delivery = tuple(delivery)
        self.basket = tuple(basket)
        self.calendar = get_calendar(calendar)
        # self.last_trading = delivery[1] if last_trading is None else
        self.nominal = defaults.notional if nominal is None else nominal
        self.contracts = 1 if contracts is None else contracts
        self._cfs = None

    @property
    def notional(self):
        """
        Return the notional as number of contracts multiplied by contract nominal.

        Returns
        -------
        float
        """
        return self.nominal * self.contracts * -1  # long positions is negative notn

    @property
    def cfs(self):
        """
        Return the conversion factors for each bond in the ordered ``basket``.

        Returns
        -------
        tuple

        Notes
        -----
        This method uses the traditional calculation of obtaining a clean price
        for each bond on the **first delivery date** assuming the **yield-to-maturity**
        is set as the nominal coupon of the bond future, and scaled to 100.

        .. warning::

           Some exchanges, such as EUREX, specify their own conversion factors' formula
           which differs slightly in the definition of yield-to-maturity than the
           implementation offered by *rateslib*. This results in small differences and
           is *potentially* explained in the way dates, holidays and DCFs are handled
           by each calculator.

        For ICE-LIFFE and gilt futures the methods between the exchange and *rateslib*
        align which results in accurate values. Official values can be validated
        against the document
        :download:`ICE-LIFFE Jun23 Long Gilt<_static/long_gilt_initial_jun23.pdf>`.

        For an equivalent comparison with values which do not exactly align see
        :download:`EUREX Jun23 Bond Futures<_static/eurex_bond_conversion_factors.csv>`.

        Examples
        --------

        .. ipython:: python

           kws = dict(
               stub="ShortFront",
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               settle=1,
           )
           bonds = [
               FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
               FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
           ]
           future = BondFuture(
               delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
           )
           future.cfs

        """
        if self._cfs is None:
            self._cfs = self._conversion_factors()
        return self._cfs

    def _conversion_factors(self):
        return tuple(
            bond.price(self.coupon, self.delivery[0]) / 100 for bond in self.basket
        )

    def dlv(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        repo_rate: Union[float, Dual, Dual2, list, tuple],
        settlement: datetime,
        delivery: Optional[datetime] = None,
        convention: Optional[str] = None,
        dirty: bool = False,
    ):
        """
        Return an aggregated DataFrame of metrics similar to the Bloomberg DLV function.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        repo_rate: float, Dual, Dual2 or list/tuple of such
            The repo rates of the bonds to delivery.
        settlement: datetime
            The settlement date of the bonds, required only if ``dirty`` is *True*.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention applied to the repo rates.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        DataFrame
        """
        if not isinstance(repo_rate, (tuple, list)):
            r_ = (repo_rate,) * len(self.basket)
        else:
            r_ = tuple(repo_rate)

        df = DataFrame(
            columns=[
                "Bond",
                "Price",
                "YTM",
                "C.Factor",
                "Gross Basis",
                "Implied Repo",
                "Actual Repo",
                "Net Basis",
            ],
            index=range(len(self.basket)),
        )
        df["Price"] = prices
        df["YTM"] = [
            bond.ytm(prices[i], settlement, dirty=dirty)
            for i, bond in enumerate(self.basket)
        ]
        df["C.Factor"] = self.cfs
        df["Gross Basis"] = self.gross_basis(
            future_price, prices, settlement, dirty=dirty
        )
        df["Implied Repo"] = self.implied_repo(
            future_price, prices, settlement, delivery, convention, dirty=dirty
        )
        df["Actual Repo"] = r_
        df["Net Basis"] = self.net_basis(
            future_price, prices, r_, settlement, delivery, convention, dirty=dirty
        )
        df["Bond"] = [
            f"{bond.fixed_rate:,.3f}% "
            f"{bond.leg1.schedule.termination.strftime('%d-%m-%Y')}"
            for bond in self.basket
        ]
        return df

    def gross_basis(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        settlement: datetime = None,
        dirty: bool = False,
    ):
        """
        Calculate the gross basis of each bond in the basket.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds, required only if ``dirty`` is *True*.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if dirty:
            prices_ = tuple(
                prices[i] - bond.accrued(settlement)
                for i, bond in enumerate(self.basket)
            )
        else:
            prices_ = prices
        return tuple(
            prices_[i] - self.cfs[i] * future_price for i in range(len(self.basket))
        )

    def net_basis(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        repo_rate: Union[float, Dual, Dual2, list, tuple],
        settlement: datetime,
        delivery: Optional[datetime] = None,
        convention: Optional[str] = None,
        dirty: bool = False,
    ):
        """
        Calculate the net basis of each bond in the basket via the proceeds
        method of repo.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        repo_rate: float, Dual, Dual2 or list/tuple of such
            The repo rates of the bonds to delivery.
        settlement: datetime
            The settlement date of the bonds, required only if ``dirty`` is *True*.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention applied to the repo rates.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if delivery is None:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        if not isinstance(repo_rate, (list, tuple)):
            r_ = (repo_rate,) * len(self.basket)
        else:
            r_ = repo_rate

        net_basis_ = tuple(
            bond.fwd_from_repo(
                prices[i], settlement, f_settlement, r_[i], convention, dirty=dirty
            )
            - self.cfs[i] * future_price
            for i, bond in enumerate(self.basket)
        )
        return net_basis_

    def implied_repo(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        settlement: datetime,
        delivery: Optional[datetime] = None,
        convention: Optional[str] = None,
        dirty: bool = False,
    ):
        """
        Calculate the implied repo of each bond in the basket using the proceeds
        method.

        Parameters
        ----------
        future_price: float, Dual, Dual2
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention used in the rate.
        dirty: bool
            Whether the bond prices are given including accrued interest.

        Returns
        -------
        tuple
        """
        if delivery is None:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        implied_repos = tuple()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            implied_repos += (
                bond.repo_from_fwd(
                    price=prices[i],
                    settlement=settlement,
                    forward_settlement=f_settlement,
                    forward_price=invoice_price,
                    convention=convention,
                    dirty=dirty,
                ),
            )
        return implied_repos

    def ytm(
        self,
        future_price: Union[float, Dual, Dual2],
        delivery: Optional[datetime] = None,
    ):
        """
        Calculate the yield-to-maturity of the bond future.

        Parameters
        ----------
        future_price : float, Dual, Dual2
            The price of the future.
        delivery : datetime, optional
            The future delivery day on which to calculate the yield. If not given aligns
            with the last delivery day specified on the future.

        Returns
        -------
        tuple
        """
        if delivery is None:
            settlement = self.delivery[1]
        else:
            settlement = delivery
        adjusted_prices = [future_price * cf for cf in self.cfs]
        yields = tuple(
            bond.ytm(adjusted_prices[i], settlement)
            for i, bond in enumerate(self.basket)
        )
        return yields

    def duration(
        self,
        future_price: float,
        metric: str = "risk",
        delivery: Optional[datetime] = None,
    ):
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm`` .

        Parameters
        ----------
        future_price : float
            The price of the future.
        metric : str
            The specific duration calculation to return. See notes.
        delivery : datetime, optional
            The delivery date of the contract.

        Returns
        -------
        float

        See Also
        --------
        FixedRateBond.duration: Calculation the risk of a FixedRateBond.

        Example
        -------
        .. ipython:: python

           risk = future.duration(112.98)
           risk

        The difference in yield is shown to be 1bp for the CTD (index: 0)
        when the futures price is adjusted by the risk amount.

        .. ipython:: python

           future.ytm(112.98)
           future.ytm(112.98 + risk[0] / 100)
        """
        if delivery is None:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        _ = ()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            ytm = bond.ytm(invoice_price, f_settlement)
            if metric == "risk":
                _ += (bond.duration(ytm, f_settlement, "risk") / self.cfs[i],)
            else:
                _ += (bond.duration(ytm, f_settlement, metric),)
        return _

    def convexity(
        self,
        future_price: float,
        delivery: Optional[datetime] = None,
    ):
        """
        Return the second derivative of ``price`` w.r.t. ``ytm`` .

        Parameters
        ----------
        future_price : float
            The price of the future.
        delivery : datetime, optional
            The delivery date of the contract. If not given uses the last delivery day
            in the delivery window.

        Returns
        -------
        float

        See Also
        --------
        FixedRateBond.convexity: Calculate the convexity of a FixedRateBond.

        Example
        -------
        .. ipython:: python

           risk = future.duration(112.98)
           convx = future.convexity(112.98)
           convx

        Observe the change in risk duration when the prices is increased by 1bp.

        .. ipython:: python

           future.duration(112.98)
           future.duration(112.98 + risk[0] / 100)
        """
        if delivery is None:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        _ = ()
        for i, bond in enumerate(self.basket):
            invoice_price = future_price * self.cfs[i]
            ytm = bond.ytm(invoice_price, f_settlement)
            _ += (bond.convexity(ytm, f_settlement) / self.cfs[i],)
        return _

    def ctd_index(
        self,
        future_price: float,
        prices: Union[list, tuple],
        settlement: datetime,
        delivery: Optional[datetime] = None,
        dirty: bool = False,
    ):
        """
        Determine the index of the CTD in the basket from implied repo rate.

        Parameters
        ----------
        future_price : float
            The price of the future.
        prices : list or tuple of float, Dual, Dual2, optional
            The prices of the bonds to determine the CTD. Not used is ``ctd_index``
            is given.
        settlement : datetime
            The settlement date of the bonds' ``prices``. Only required if ``prices``
            are given.
        delivery : datetime, optional
            The delivery date of the contract.
        dirty : bool, optional
            Whether the ``prices`` given include accrued interest or not.

        Returns
        -------
        int
        """
        implied_repo = self.implied_repo(
            future_price, prices, settlement, delivery, "Act365F", dirty
        )
        ctd_index_ = implied_repo.index(max(implied_repo))
        return ctd_index_

    # Digital Methods

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        metric: str = "future_price",
        delivery: Optional[datetime] = None,
    ):
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str in {"future_price", "ytm"}, optional
            Metric returned by the method.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        This method determines the *'futures_price'* and *'ytm'*  by assuming a net
        basis of zero and pricing from the cheapest to delivery (CTD).
        """
        metric = metric.lower()
        if metric not in ["future_price", "ytm"]:
            raise ValueError("`metric` must be in {'future_price', 'ytm'}.")

        if delivery is None:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery
        prices_ = [
            bond.rate(curves, solver, fx, base, "fwd_clean_price", f_settlement)
            for bond in self.basket
        ]
        future_prices_ = [price / self.cfs[i] for i, price in enumerate(prices_)]
        future_price = min(future_prices_)
        ctd_index = future_prices_.index(min(future_prices_))

        if metric == "future_price":
            return future_price
        elif metric == "ytm":
            return self.basket[ctd_index].ytm(
                future_price * self.cfs[ctd_index], f_settlement
            )

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Determine the monetary value of the bond future position.

        This method is mainly included to calculate risk sensitivities. The
        monetary value of bond futures is not usually a metric worth considering.
        The profit or loss of a position based on entry level is a more common
        metric, however the initial value of the position does not affect the risk.

        See :meth:`BaseDerivative.npv`.
        """
        future_price = self.rate(curves, solver, fx, base, "future_price")
        fx, base = _get_fx_and_base(self.currency, fx, base)
        npv_ = future_price / 100 * -self.notional
        if local:
            return {self.currency: npv_}
        else:
            return npv_ * fx


class BaseDerivative(Sensitivities, BaseMixin, metaclass=ABCMeta):
    """
    Abstract base class with common parameters for many ``Derivative`` subclasses.

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
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    leg2_kwargs: Any
        All ``leg2`` arguments can be similarly input as above, e.g. ``leg2_frequency``.
        If **not** given, any ``leg2``
        argument inherits its value from the ``leg1`` arguments, except in the case of
        ``notional`` and ``amortization`` where ``leg2`` inherits the negated value.
    curves : Curve, LineCurve, str or list of such, optional
        A single :class:`~rateslib.curves.Curve`,
        :class:`~rateslib.curves.LineCurve` or id or a
        list of such. A list defines the following curves in the order:

        - Forecasting :class:`~rateslib.curves.Curve` or
          :class:`~rateslib.curves.LineCurve` for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
        - Forecasting :class:`~rateslib.curves.Curve` or
          :class:`~rateslib.curves.LineCurve` for ``leg2``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.

    Attributes
    ----------
    effective : datetime
    termination : datetime
    frequency : str
    stub : str
    front_stub : datetime
    back_stub : datetime
    roll : str, int
    eom : bool
    modifier : str
    calendar : Calendar
    payment_lag : int
    notional : float
    amortization : float
    convention : str
    leg2_effective : datetime
    leg2_termination : datetime
    leg2_frequency : str
    leg2_stub : str
    leg2_front_stub : datetime
    leg2_back_stub : datetime
    leg2_roll : str, int
    leg2_eom : bool
    leg2_modifier : str
    leg2_calendar : Calendar
    leg2_payment_lag : int
    leg2_notional : float
    leg2_amortization : float
    leg2_convention : str
    """

    @abc.abstractmethod
    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str] = None,
        frequency: Optional[int] = None,
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
        leg2_effective: Optional[datetime] = "inherit",
        leg2_termination: Optional[Union[datetime, str]] = "inherit",
        leg2_frequency: Optional[int] = "inherit",
        leg2_stub: Optional[str] = "inherit",
        leg2_front_stub: Optional[datetime] = "inherit",
        leg2_back_stub: Optional[datetime] = "inherit",
        leg2_roll: Optional[Union[str, int]] = "inherit",
        leg2_eom: Optional[bool] = "inherit",
        leg2_modifier: Optional[str] = "inherit",
        leg2_calendar: Optional[Union[CustomBusinessDay, str]] = "inherit",
        leg2_payment_lag: Optional[int] = "inherit",
        leg2_notional: Optional[float] = "inherit_negate",
        leg2_currency: Optional[str] = "inherit",
        leg2_amortization: Optional[float] = "inherit_negate",
        leg2_convention: Optional[str] = "inherit",
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.curves = curves
        notional = defaults.notional if notional is None else notional
        if payment_lag is None:
            payment_lag = defaults.payment_lag_specific[type(self).__name__]
        for attribute in [
            "effective",
            "termination",
            "frequency",
            "stub",
            "front_stub",
            "back_stub",
            "roll",
            "eom",
            "modifier",
            "calendar",
            "payment_lag",
            "convention",
            "notional",
            "amortization",
            "currency",
        ]:
            leg2_val, val = vars()[f"leg2_{attribute}"], vars()[attribute]
            if leg2_val == "inherit":
                _ = val
            elif leg2_val == "inherit_negate":
                _ = None if val is None else val * -1
            else:
                _ = leg2_val
            setattr(self, attribute, val)
            setattr(self, f"leg2_{attribute}", _)

    def analytic_delta(self, *args, leg=1, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        Parameters
        ----------
        args :
            Required positional arguments supplied to
            :meth:`BaseLeg.analytic_delta<rateslib.legs.BaseLeg.analytic_delta>`.
        leg : int in [1, 2]
            The leg identifier of which to take the analytic delta.
        kwargs :
            Required Keyword arguments supplied to
            :meth:`BaseLeg.analytic_delta()<rateslib.legs.BaseLeg.analytic_delta>`.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        .. ipython:: python

           curve = Curve({dt(2021,1,1): 1.00, dt(2025,1,1): 0.83}, id="SONIA")
           fxr = FXRates({"gbpusd": 1.25}, base="usd")

        .. ipython:: python

           irs = IRS(
               effective=dt(2022, 1, 1),
               termination="6M",
               frequency="Q",
               currency="gbp",
               notional=1e9,
               fixed_rate=5.0,
           )
           irs.analytic_delta(curve, curve)
           irs.analytic_delta(curve, curve, fxr)
           irs.analytic_delta(curve, curve, fxr, "gbp")
        """
        return getattr(self, f"leg{leg}").analytic_delta(*args, **kwargs)

    @abstractmethod
    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        Parameters
        ----------
        curves : CurveType, str or list of such, optional
            A single :class:`~rateslib.curves.Curve`,
            :class:`~rateslib.curves.LineCurve` or id or a
            list of such. A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults
            to ``fx.base``.

        Returns
        -------
        DataFrame

        Notes
        -----
        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.

        If **three curves** are given the single discounting curve is used as the
        discounting curve for both legs.

        Examples
        --------
        .. ipython:: python

           irs.cashflows([curve], None, fxr)
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        return concat(
            [
                self.leg1.cashflows(curves[0], curves[1], fx, base),
                self.leg2.cashflows(curves[2], curves[3], fx, base),
            ],
            keys=["leg1", "leg2"],
        )

    @abc.abstractmethod
    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the derivative object by summing legs.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            A single :class:`~rateslib.curves.Curve`,
            :class:`~rateslib.curves.LineCurve` or id or a
            list of such. A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults
            to ``fx.base``.
        local : bool, optional
            If `True` will return a dict identifying NPV by local currencies on each
            leg. Useful for multi-currency derivatives and for ensuring risk
            sensitivities are allocated to local currencies without conversion.

        Returns
        -------
        float, Dual or Dual2, or dict of such.

        Notes
        -----
        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.

        If **three curves** are given the single discounting curve is used as the
        discounting curve for both legs.

        Examples
        --------
        .. ipython:: python

           irs.npv(curve)
           irs.npv([curve], None, fxr)
           irs.npv([curve], None, fxr, "gbp")
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        leg1_npv = self.leg1.npv(curves[0], curves[1], fx, base, local)
        leg2_npv = self.leg2.npv(curves[2], curves[3], fx, base, local)
        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)
                for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    @abc.abstractmethod
    def rate(self, *args, **kwargs):
        """
        Return the `rate` or typical `price` for a derivative instrument.

        Returns
        -------
        Dual

        Notes
        -----
        This method must be implemented for instruments to function effectively in
        :class:`Solver` iterations.
        """
        pass  # pragma: no cover

    # def delta(
    #     self,
    #     curves: Union[Curve, str, list],
    #     solver: Solver,
    #     fx: Optional[Union[float, FXRates, FXForwards]] = None,
    #     base: Optional[str] = None,
    # ):
    #     npv = self.npv(curves, solver, fx, base)
    #     return solver.delta(npv)
    #
    # def gamma(
    #     self,
    #     curves: Union[Curve, str, list],
    #     solver: Solver,
    #     fx: Optional[Union[float, FXRates, FXForwards]] = None,
    #     base: Optional[str] = None,
    # ):
    #     _ = solver._ad  # store original order
    #     solver._set_ad_order(2)
    #     npv = self.npv(curves, solver, fx, base)
    #     grad_s_sT_P = solver.gamma(npv)
    #     solver._set_ad_order(_)  # reset original order
    #     return grad_s_sT_P


class IRS(BaseDerivative):
    """
    Create an interest rate swap composing a :class:`~rateslib.legs.FixedLeg`
    and a :class:`~rateslib.legs.FloatLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="usd"
       )

    Create the IRS, and demonstrate the :meth:`~rateslib.instruments.IRS.rate`,
    :meth:`~rateslib.instruments.IRS.npv`,
    :meth:`~rateslib.instruments.IRS.analytic_delta`, and
    :meth:`~rateslib.instruments.IRS.spread`.

    .. ipython:: python

       irs = IRS(
           effective=dt(2022, 1, 1),
           termination="18M",
           frequency="A",
           calendar="nyc",
           currency="usd",
           fixed_rate=3.269,
           convention="Act360",
           notional=100e6,
           curves=["usd"],
       )
       irs.rate(curves=usd)
       irs.npv(curves=usd)
       irs.analytic_delta(curve=usd)
       irs.spread(curves=usd)

    A DataFrame of :meth:`~rateslib.instruments.IRS.cashflows`.

    .. ipython:: python

       irs.cashflows(curves=usd)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.IRS.delta`
    and :meth:`~rateslib.instruments.IRS.gamma`, construct a curve model.

    .. ipython:: python

       sofr_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="Act360",
           calendar="nyc",
           currency="usd",
           curves=["usd"]
       )
       instruments = [
           IRS(termination="1Y", **sofr_kws),
           IRS(termination="2Y", **sofr_kws),
       ]
       solver = Solver(
           curves=[usd],
           instruments=instruments,
           s=[3.65, 3.20],
           instrument_labels=["1Y", "2Y"],
           id="sofr",
       )
       irs.delta(solver=solver)
       irs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_fixings: Optional[Union[float, list, Series]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FixedLeg(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLeg(
            float_spread=leg2_float_spread,
            spread_compound_method=leg2_spread_compound_method,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )

    def _set_pricing_mid(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
    ):
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = float(mid_market_rate)

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return super().analytic_delta(*args, **kwargs)

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        if self.fixed_rate is None:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market rate of the IRS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        leg2_npv = self.leg2.npv(curves[2], curves[3])
        return self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100
        # leg1_analytic_delta = self.leg1.analytic_delta(curves[0], curves[1])
        # return leg2_npv / (leg1_analytic_delta * 100)

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        return super().cashflows(curves, solver, fx, base)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def spread(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market float spread (bps) required to equate to the fixed rate.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        If the :class:`IRS` is specified without a ``fixed_rate`` this should always
        return the current ``leg2_float_spread`` value or zero since the fixed rate used
        for calculation is the implied rate including the current ``leg2_float_spread``
        parameter.

        Examples
        --------
        For the most common parameters this method will be exact.

        .. ipython:: python

           irs.spread(curves=usd)
           irs.leg2_float_spread = -6.948753
           irs.npv(curves=usd)

        When a non-linear spread compound method is used for float RFR legs this is
        an approximation, via second order Taylor expansion.

        .. ipython:: python

           irs = IRS(
               effective=dt(2022, 2, 15),
               termination=dt(2022, 8, 15),
               frequency="Q",
               convention="30e360",
               leg2_convention="Act360",
               leg2_fixing_method="rfr_payment_delay",
               leg2_spread_compound_method="isda_compounding",
               payment_lag=2,
               fixed_rate=2.50,
               leg2_float_spread=0,
               notional=50000000,
               currency="usd",
           )
           irs.spread(curves=usd)
           irs.leg2_float_spread = -111.060143
           irs.npv(curves=usd)
           irs.spread(curves=usd)

        The ``leg2_float_spread`` is determined through NPV differences. If the difference
        is small since the defined spread is already quite close to the solution the
        approximation is much more accurate. This is shown above where the second call
        to ``irs.spread`` is different to the previous call, albeit the difference
        is 1/10000th of a basis point.
        """
        irs_npv = self.npv(curves, solver)
        specified_spd = 0 if self.leg2.float_spread is None else self.leg2.float_spread
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        return self.leg2._spread(-irs_npv, curves[2], curves[3]) + specified_spd
        # leg2_analytic_delta = self.leg2.analytic_delta(curves[2], curves[3])
        # return irs_npv / leg2_analytic_delta + specified_spd


class Swap(IRS):
    """
    Alias for :class:`~rateslib.instruments.IRS`.
    """


class IIRS(BaseDerivative):
    """
    Create an indexed interest rate swap (IIRS) composing an
    :class:`~rateslib.legs.IndexFixedLeg` and a :class:`~rateslib.legs.FloatLeg`.

    If ``notional_exchange``, the legs are :class:`~rateslib.legs.IndexFixedLegExchange`
    and :class:`~rateslib.legs.FloatLegExchange`.

    Parameters
    ----------
    args : dict
       Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
       The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
       will be set to mid-market when curves are provided.
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
    notional_exchange : bool, optional
       Whether the legs include final notional exchanges and interim
       amortization notional exchanges.
    kwargs : dict
       Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

      usd = Curve(
          nodes={
              dt(2022, 1, 1): 1.0,
              dt(2027, 1, 1): 0.85,
              dt(2032, 1, 1): 0.65,
          },
          id="usd",
      )
      us_cpi = IndexCurve(
          nodes={
              dt(2022, 1, 1): 1.0,
              dt(2027, 1, 1): 0.85,
              dt(2032, 1, 1): 0.70,
          },
          id="us_cpi",
          index_base=100,
          index_lag=3,
      )

    Create the ZCIS, and demonstrate the :meth:`~rateslib.instruments.ZCIS.rate`,
    :meth:`~rateslib.instruments.ZCIS.npv`,
    :meth:`~rateslib.instruments.ZCIS.analytic_delta`, and

    .. ipython:: python

      iirs = IIRS(
          effective=dt(2022, 1, 1),
          termination="4Y",
          frequency="A",
          calendar="nyc",
          currency="usd",
          fixed_rate=2.05,
          convention="1+",
          notional=100e6,
          index_base=100.0,
          index_method="monthly",
          index_lag=3,
          index_fixings=None,
          notional_exchange=True,
          leg2_convention="Act360",
          curves=["us_cpi", "usd", "usd", "usd"],
      )
      iirs.rate(curves=[us_cpi, usd, usd, usd])
      iirs.npv(curves=[us_cpi, usd, usd, usd])

    A DataFrame of :meth:`~rateslib.instruments.IIRS.cashflows`.

    .. ipython:: python

      iirs.cashflows(curves=[us_cpi, usd, usd, usd])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.IIRS.delta`
    and :meth:`~rateslib.instruments.IIRS.gamma`, construct a curve model.

    .. ipython:: python

      sofr_kws = dict(
          effective=dt(2022, 1, 1),
          frequency="A",
          convention="Act360",
          calendar="nyc",
          currency="usd",
          curves=["usd"]
      )
      cpi_kws = dict(
          effective=dt(2022, 1, 1),
          frequency="A",
          convention="1+",
          calendar="nyc",
          leg2_index_method="monthly",
          currency="usd",
          curves=["usd", "usd", "us_cpi", "usd"]
      )
      instruments = [
          IRS(termination="5Y", **sofr_kws),
          IRS(termination="10Y", **sofr_kws),
          ZCIS(termination="5Y", **cpi_kws),
          ZCIS(termination="10Y", **cpi_kws),
      ]
      solver = Solver(
          curves=[usd, us_cpi],
          instruments=instruments,
          s=[3.40, 3.60, 2.2, 2.05],
          instrument_labels=["5Y", "10Y", "5Yi", "10Yi"],
          id="us",
      )
      iirs.delta(solver=solver)
      iirs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _index_base_mixin = True
    _leg2_float_spread_mixin = True

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        index_base: Optional[Union[float, Series]] = None,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: Optional[str] = None,
        index_lag: Optional[int] = None,
        notional_exchange: Optional[bool] = False,
        payment_lag_exchange: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_fixings: Optional[Union[float, list]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._index_base = index_base
        self._fixed_rate = fixed_rate
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange

        self.notional_exchange = notional_exchange
        if not notional_exchange:
            L1, L2 = IndexFixedLeg, FloatLeg
            l1_args, l2_args = {}, {}
        else:
            L1, L2 = IndexFixedLegExchange, FloatLegExchange
            l1_args = dict(
                payment_lag_exchange=payment_lag_exchange,
                initial_exchange=False,
            )
            l2_args = dict(
                payment_lag_exchange=leg2_payment_lag_exchange,
                initial_exchange=False,
            )

        self.leg1 = L1(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
            index_base=index_base,
            index_method=index_method,
            index_lag=index_lag,
            index_fixings=index_fixings,
            **l1_args,
        )
        self.leg2 = L2(
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
            float_spread=leg2_float_spread,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            spread_compound_method=leg2_spread_compound_method,
            **l2_args,
        )

    def _set_pricing_mid(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
    ):
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = float(mid_market_rate)

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        if self.index_base is None:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        if self.fixed_rate is None:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        if self.index_base is None:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        return super().cashflows(curves, solver, fx, base)

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market rate of the IRS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        if self.index_base is None:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        leg2_npv = self.leg2.npv(curves[2], curves[3])

        if self.fixed_rate is None:
            self.leg1.fixed_rate = 0.0
        _existing = self.leg1.fixed_rate
        leg1_npv = self.leg1.npv(curves[0], curves[1])

        _ = self.leg1._spread(-leg2_npv - leg1_npv, curves[0], curves[1]) / 100
        return _ + _existing

    def spread(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market float spread (bps) required to equate to the fixed rate.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        If the :class:`IRS` is specified without a ``fixed_rate`` this should always
        return the current ``leg2_float_spread`` value or zero since the fixed rate used
        for calculation is the implied rate including the current ``leg2_float_spread``
        parameter.

        Examples
        --------
        For the most common parameters this method will be exact.

        .. ipython:: python

           irs.spread(curves=usd)
           irs.leg2_float_spread = -6.948753
           irs.npv(curves=usd)

        When a non-linear spread compound method is used for float RFR legs this is
        an approximation, via second order Taylor expansion.

        .. ipython:: python

           irs = IRS(
               effective=dt(2022, 2, 15),
               termination=dt(2022, 8, 15),
               frequency="Q",
               convention="30e360",
               leg2_convention="Act360",
               leg2_fixing_method="rfr_payment_delay",
               leg2_spread_compound_method="isda_compounding",
               payment_lag=2,
               fixed_rate=2.50,
               leg2_float_spread=0,
               notional=50000000,
               currency="usd",
           )
           irs.spread(curves=usd)
           irs.leg2_float_spread = -111.060143
           irs.npv(curves=usd)
           irs.spread(curves=usd)

        The ``leg2_float_spread`` is determined through NPV differences. If the difference
        is small since the defined spread is already quite close to the solution the
        approximation is much more accurate. This is shown above where the second call
        to ``irs.spread`` is different to the previous call, albeit the difference
        is 1/10000th of a basis point.
        """
        irs_npv = self.npv(curves, solver)
        specified_spd = 0 if self.leg2.float_spread is None else self.leg2.float_spread
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        return self.leg2._spread(-irs_npv, curves[2], curves[3]) + specified_spd


class ZCS(BaseDerivative):
    """
    Create a zero coupon swap (ZCS) composing a :class:`~rateslib.legs.ZeroFixedLeg`
    and a :class:`~rateslib.legs.ZeroFloatLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.70,
           },
           id="usd"
       )

    Create the ZCS, and demonstrate the :meth:`~rateslib.instruments.ZCS.rate`,
    :meth:`~rateslib.instruments.ZCS.npv`,
    :meth:`~rateslib.instruments.ZCS.analytic_delta`, and

    .. ipython:: python

       zcs = ZCS(
           effective=dt(2022, 1, 1),
           termination="10Y",
           frequency="Q",
           calendar="nyc",
           currency="usd",
           fixed_rate=4.0,
           convention="Act360",
           notional=100e6,
           curves=["usd"],
       )
       zcs.rate(curves=usd)
       zcs.npv(curves=usd)
       zcs.analytic_delta(curve=usd)

    A DataFrame of :meth:`~rateslib.instruments.ZCS.cashflows`.

    .. ipython:: python

       zcs.cashflows(curves=usd)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.ZCS.delta`
    and :meth:`~rateslib.instruments.ZCS.gamma`, construct a curve model.

    .. ipython:: python

       sofr_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="Act360",
           calendar="nyc",
           currency="usd",
           curves=["usd"]
       )
       instruments = [
           IRS(termination="5Y", **sofr_kws),
           IRS(termination="10Y", **sofr_kws),
       ]
       solver = Solver(
           curves=[usd],
           instruments=instruments,
           s=[3.40, 3.60],
           instrument_labels=["5Y", "10Y"],
           id="sofr",
       )
       zcs.delta(solver=solver)
       zcs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_fixings: Optional[Union[float, list, Series]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = ZeroFixedLeg(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = ZeroFloatLeg(
            float_spread=leg2_float_spread,
            spread_compound_method=leg2_spread_compound_method,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta<rateslib.instruments.BaseDerivative.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def _set_pricing_mid(self, curves, solver):
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = float(mid_market_rate)

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        if self.fixed_rate is None:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market rate of the ZCS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.

        The *'irr'* ``fixed_rate`` defines a cashflow by:

        .. math::

           -notional * ((1 + irr / f)^{f \\times dcf} - 1)

        where :math:`f` is associated with the compounding frequency.
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        leg2_npv = self.leg2.npv(curves[2], curves[3])
        _ = self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100
        return _

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        return super().cashflows(curves, solver, fx, base)


class ZCIS(BaseDerivative):
    """
    Create a zero coupon index swap (ZCIS) composing an
    :class:`~rateslib.legs.ZeroFixedLeg`
    and a :class:`~rateslib.legs.ZeroIndexLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
        will be set to mid-market when curves are provided.
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
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.65,
           },
           id="usd",
       )
       us_cpi = IndexCurve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.70,
           },
           id="us_cpi",
           index_base=100,
           index_lag=3,
       )

    Create the ZCIS, and demonstrate the :meth:`~rateslib.instruments.ZCIS.rate`,
    :meth:`~rateslib.instruments.ZCIS.npv`,
    :meth:`~rateslib.instruments.ZCIS.analytic_delta`, and

    .. ipython:: python

       zcis = ZCIS(
           effective=dt(2022, 1, 1),
           termination="10Y",
           frequency="A",
           calendar="nyc",
           currency="usd",
           fixed_rate=2.05,
           convention="1+",
           notional=100e6,
           leg2_index_base=100.0,
           leg2_index_method="monthly",
           leg2_index_lag=3,
           leg2_index_fixings=None,
           curves=["usd", "usd", "us_cpi", "usd"],
       )
       zcis.rate(curves=[usd, usd, us_cpi, usd])
       zcis.npv(curves=[usd, usd, us_cpi, usd])
       zcis.analytic_delta(usd, usd)

    A DataFrame of :meth:`~rateslib.instruments.ZCIS.cashflows`.

    .. ipython:: python

       zcis.cashflows(curves=[usd, usd, us_cpi, usd])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.ZCIS.delta`
    and :meth:`~rateslib.instruments.ZCIS.gamma`, construct a curve model.

    .. ipython:: python

       sofr_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="Act360",
           calendar="nyc",
           currency="usd",
           curves=["usd"]
       )
       cpi_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="1+",
           calendar="nyc",
           leg2_index_method="monthly",
           currency="usd",
           curves=["usd", "usd", "us_cpi", "usd"]
       )
       instruments = [
           IRS(termination="5Y", **sofr_kws),
           IRS(termination="10Y", **sofr_kws),
           ZCIS(termination="5Y", **cpi_kws),
           ZCIS(termination="10Y", **cpi_kws),
       ]
       solver = Solver(
           curves=[usd, us_cpi],
           instruments=instruments,
           s=[3.40, 3.60, 2.2, 2.05],
           instrument_labels=["5Y", "10Y", "5Yi", "10Yi"],
           id="us",
       )
       zcis.delta(solver=solver)
       zcis.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_index_base_mixin = True

    def __init__(
        self,
        *args,
        fixed_rate: Optional[float] = None,
        leg2_index_base: Optional[Union[float, Series]] = None,
        leg2_index_fixings: Optional[Union[float, Series]] = None,
        leg2_index_method: Optional[str] = None,
        leg2_index_lag: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_index_base = leg2_index_base
        self.leg1 = ZeroFixedLeg(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = ZeroIndexLeg(
            index_base=leg2_index_base,
            index_method=leg2_index_method,
            index_lag=leg2_index_lag,
            index_fixings=leg2_index_fixings,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )

    def _set_pricing_mid(self, curves, solver):
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = float(mid_market_rate)

    def cashflows(self, *args, **kwargs):
        return super().cashflows(*args, **kwargs)

    def npv(
            self,
            curves: Optional[Union[Curve, str, list]] = None,
            solver: Optional[Solver] = None,
            fx: Optional[Union[float, FXRates, FXForwards]] = None,
            base: Optional[str] = None,
            local: bool = False,
    ):
        if self.fixed_rate is None:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the mid-market IRR rate of the ZCIS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        if self.leg2_index_base is None:
            # must forecast for the leg
            self.leg2.index_base = curves[2].index_value(
                self.leg2.schedule.effective, self.leg2.index_method
            )
        leg2_npv = self.leg2.npv(curves[2], curves[3])

        return self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100


class SBS(BaseDerivative):
    """
    Create a single currency basis swap composing two
    :class:`~rateslib.legs.FloatLeg` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseDerivative`.
    float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
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
    leg2_float_spread : float or None
        The floating spread applied in a simple way (after daily compounding) to the
        second :class:`~rateslib.legs.FloatLeg`. If `None` will be set to zero.
        float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct curves to price the example.

    .. ipython:: python

       eur3m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="eur3m",
       )
       eur6m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.962,
               dt(2024, 1, 1): 0.936
           },
           id="eur6m",
       )

    Create the SBS, and demonstrate the :meth:`~rateslib.instruments.SBS.rate`,
    :meth:`~rateslib.instruments.SBS.npv`,
    :meth:`~rateslib.instruments.SBS.analytic_delta`, and
    :meth:`~rateslib.instruments.SBS.spread`.

    .. ipython:: python

       sbs = SBS(
           effective=dt(2022, 1, 1),
           termination="18M",
           frequency="Q",
           leg2_frequency="S",
           calendar="tgt",
           currency="eur",
           fixing_method="ibor",
           method_param=2,
           convention="Act360",
           leg2_float_spread=-22.9,
           notional=100e6,
           curves=["eur3m", "eur3m", "eur6m", "eur3m"],
       )
       sbs.rate(curves=[eur3m, eur3m, eur6m, eur3m])
       sbs.npv(curves=[eur3m, eur3m, eur6m, eur3m])
       sbs.analytic_delta(curve=eur6m, disc_curve=eur3m, leg=2)
       sbs.spread(curves=[eur3m, eur3m, eur6m, eur3m], leg=2)

    A DataFrame of :meth:`~rateslib.instruments.SBS.cashflows`.

    .. ipython:: python

       sbs.cashflows(curves=[eur3m, eur3m, eur6m, eur3m])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.SBS.delta`
    and :meth:`~rateslib.instruments.SBS.gamma`, construct a curve model.

    .. ipython:: python

       irs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           leg2_frequency="Q",
           convention="30E360",
           leg2_convention="Act360",
           leg2_fixing_method="ibor",
           leg2_method_param=2,
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m"],
       )
       sbs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="Q",
           leg2_frequency="S",
           convention="Act360",
           fixing_method="ibor",
           method_param=2,
           leg2_convention="Act360",
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m", "eur6m", "eur3m"]
       )
       instruments = [
           IRS(termination="1Y", **irs_kws),
           IRS(termination="2Y", **irs_kws),
           SBS(termination="1Y", **sbs_kws),
           SBS(termination="2Y", **sbs_kws),
       ]
       solver = Solver(
           curves=[eur3m, eur6m],
           instruments=instruments,
           s=[1.55, 1.6, 5.5, 6.5],
           instrument_labels=["1Y", "2Y", "1Y 3s6s", "2Y 3s6s"],
           id="eur",
       )
       sbs.delta(solver=solver)
       sbs.gamma(solver=solver)

    """

    _float_spread_mixin = True
    _leg2_float_spread_mixin = True
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        float_spread: Optional[float] = None,
        spread_compound_method: Optional[str] = None,
        fixings: Optional[Union[float, list, Series]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_fixings: Optional[Union[float, list, Series]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._float_spread = float_spread
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FloatLeg(
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLeg(
            float_spread=leg2_float_spread,
            spread_compound_method=leg2_spread_compound_method,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )

    def _set_pricing_mid(self, curves, solver):
        rate = self.rate(curves, solver)
        self.leg1.float_spread = float(rate)

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        return super().cashflows(curves, solver, fx, base)

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the derivative object by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        if self.float_spread is None and self.leg2_float_spread is None:
            # set a pricing parameter for the purpose of pricing NPV at zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        leg: int = 1,
    ):
        """
        Return the mid-market float spread on the specified leg of the SBS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg1.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
            - Forecasting :class:`~rateslib.curves.Curve` for floating leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating
            instruments.
        leg: int in [1, 2]
            Specify which leg the spread calculation is applied to.

        Returns
        -------
        float, Dual or Dual2
        """
        core_npv = super().npv(curves, solver)
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        if leg == 1:
            leg_obj, args = self.leg1, (curves[0], curves[1])
        else:
            leg_obj, args = self.leg2, (curves[2], curves[3])

        specified_spd = 0 if leg_obj.float_spread is None else leg_obj.float_spread
        return leg_obj._spread(-core_npv, *args) + specified_spd

        # irs_npv = self.npv(curves, solver)
        # curves, _ = self._get_curves_and_fx_maybe_from_solver(solver, curves, None)
        # if leg == 1:
        #     args = (curves[0], curves[1])
        # else:
        #     args = (curves[2], curves[3])
        # leg_analytic_delta = getattr(self, f"leg{leg}").analytic_delta(*args)
        # adjust = getattr(self, f"leg{leg}").float_spread
        # adjust = 0 if adjust is None else adjust
        # _ = irs_npv / leg_analytic_delta + adjust
        # return _

    def spread(self, *args, **kwargs):
        """
        Return the mid-market float spread on the specified leg of the SBS.

        Alias for :meth:`~rateslib.instruments.SBS.rate`.
        """
        return self.rate(*args, **kwargs)


class FRA(Sensitivities, BaseMixin):
    """
    Create a forward rate agreement composing a :class:`~rateslib.periods.FixedPeriod`
    and :class:`~rateslib.periods.FloatPeriod` valued in a customised manner.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    FRAs are a legacy derivative whose ``fixing_method`` is set to *"ibor"*.

    ``effective`` and ``termination`` are not adjusted prior to initialising
    ``Periods``. Care should be taken to enter these exactly.

    Examples
    --------
    Construct curves to price the example.

    .. ipython:: python

       eur3m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="eur3m",
       )

    Create the FRA, and demonstrate the :meth:`~rateslib.instruments.FRA.rate`,
    :meth:`~rateslib.instruments.FRA.npv`,
    :meth:`~rateslib.instruments.FRA.analytic_delta`.

    .. ipython:: python

       fra = FRA(
           effective=dt(2023, 2, 15),
           termination="3M",
           frequency="Q",
           calendar="tgt",
           currency="eur",
           method_param=2,
           convention="Act360",
           notional=100e6,
           fixed_rate=2.617,
           curves=["eur3m"],
       )
       fra.rate(curves=eur3m)
       fra.npv(curves=eur3m)
       fra.analytic_delta(curve=eur3m)

    A DataFrame of :meth:`~rateslib.instruments.FRA.cashflows`.

    .. ipython:: python

       fra.cashflows(curves=eur3m)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.FRA.delta`
    and :meth:`~rateslib.instruments.FRA.gamma`, construct a curve model.

    .. ipython:: python

       irs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           leg2_frequency="Q",
           convention="30E360",
           leg2_convention="Act360",
           leg2_fixing_method="ibor",
           leg2_method_param=2,
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m"],
       )
       instruments = [
           IRS(termination="1Y", **irs_kws),
           IRS(termination="2Y", **irs_kws),
       ]
       solver = Solver(
           curves=[eur3m],
           instruments=instruments,
           s=[1.55, 1.6],
           instrument_labels=["1Y", "2Y"],
           id="eur",
       )
       fra.delta(solver=solver)
       fra.gamma(solver=solver)

    """

    _fixed_rate_mixin = True

    def __init__(
        self,
        effective: datetime,
        termination: Union[datetime, str],
        frequency: str,
        modifier: Optional[Union[str, bool]] = False,
        calendar: Optional[Union[CustomBusinessDay, str]] = None,
        notional: Optional[float] = None,
        convention: Optional[str] = None,
        method_param: Optional[int] = None,
        payment_lag: Optional[int] = None,
        fixed_rate: Optional[float] = None,
        fixings: Optional[Union[float, Series]] = None,
        currency: Optional[str] = None,
        curves: Optional[Union[str, list, Curve]] = None,
    ) -> None:
        self.curves = curves
        self.currency = defaults.base_currency if currency is None else currency.lower()

        if isinstance(modifier, bool):  # then get default
            modifier_: Optional[str] = defaults.modifier
        else:
            modifier_ = modifier.upper()
        self.modifier = modifier_

        if payment_lag is None:
            self.payment_lag = defaults.payment_lag_specific["FRA"]
        else:
            self.payment_lag = payment_lag
        self.calendar = get_calendar(calendar)
        self.payment = add_tenor(effective, f"{self.payment_lag}B", None, self.calendar)

        if isinstance(termination, str):
            # if termination is string the end date is calculated as unadjusted
            termination = add_tenor(
                effective, termination, self.modifier, self.calendar
            )

        self.notional = defaults.notional if notional is None else notional

        convention = defaults.convention if convention is None else convention

        self._fixed_rate = fixed_rate
        self.leg1 = FixedPeriod(
            start=effective,
            end=termination,
            payment=self.payment,
            convention=convention,
            frequency=frequency,
            stub=False,
            currency=self.currency,
            fixed_rate=fixed_rate,
            notional=notional,
        )

        self.leg2 = FloatPeriod(
            start=effective,
            end=termination,
            payment=termination,
            spread_compound_method="none_simple",
            fixing_method="ibor",
            method_param=method_param,
            fixings=fixings,
            convention=convention,
            frequency=frequency,
            stub=False,
            currency=self.currency,
            notional=-self.notional,
        )  # FloatPeriod is used only to access the rate method for calculations.

    def _set_pricing_mid(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
    ) -> None:
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = mid_market_rate.real

    def analytic_delta(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ) -> DualTypes:
        """
        Return the analytic delta of the FRA.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)
        rate = self.rate([curve])
        _ = self.notional * self.leg1.dcf * disc_curve[self.payment] / 10000
        return fx * _ / (1 + self.leg1.dcf * rate / 100)

    def npv(
        self,
        curves: Optional[Union[str, list, Curve]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ) -> DualTypes:
        """
        Return the NPV of the derivative.

        See :meth:`BaseDerivative.npv`.
        """
        if self.fixed_rate is None:
            self._set_pricing_mid(curves, solver)
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = self.cashflow(curves[0]) * curves[1][self.payment]
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def rate(
        self,
        curves: Optional[Union[str, list, Curve]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ) -> DualTypes:
        """
        Return the mid-market rate of the FRA.

        Only the forecasting curve is required to price an FRA.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for floating leg.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : unused
        base : unused

        Returns
        -------
        float, Dual or Dual2
        """
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        return self.leg2.rate(curves[0])

    def cashflow(self, curve: Union[Curve, LineCurve]):
        """
        Calculate the local currency cashflow on the FRA from current floating rate
        and fixed rate.

        Parameters
        ----------
        curve : Curve or LineCurve,
            The forecasting curve for determining the floating rate.

        Returns
        -------
        float, Dual or Dual2
        """
        cf1 = self.leg1.cashflow
        cf2 = self.leg2.cashflow(curve)
        cf = cf1 + cf2
        rate = None if curve is None else 100 * cf2 / (self.notional * self.leg2.dcf)
        cf /= 1 + self.leg1.dcf * rate / 100

        # if self.fixed_rate is None:
        #     return 0  # set the fixed rate = to floating rate netting to zero
        # rate = self.leg2.rate(curve)
        # cf = self.notional * self.leg1.dcf * (rate - self.fixed_rate) / 100
        # cf /= 1 + self.leg1.dcf * rate / 100
        return cf

    def cashflows(
        self,
        curves: Optional[Union[str, list, Curve]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ) -> DataFrame:
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
        curves, _ = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        fx, base = _get_fx_and_base(self.currency, fx, base)
        cf = float(self.cashflow(curves[0]))
        npv_local = self.cashflow(curves[0]) * curves[1][self.payment]

        _fix = None if self.fixed_rate is None else -float(self.fixed_rate)
        _spd = None if curves[1] is None else -float(self.rate(curves[1])) * 100
        cfs = self.leg1.cashflows(curves[0], curves[1], fx, base)
        cfs[defaults.headers["type"]] = "FRA"
        cfs[defaults.headers["payment"]] = self.payment
        cfs[defaults.headers["cashflow"]] = cf
        cfs[defaults.headers["rate"]] = _fix
        cfs[defaults.headers["spread"]] = _spd
        cfs[defaults.headers["npv"]] = npv_local
        cfs[defaults.headers["fx"]] = float(fx)
        cfs[defaults.headers["npv_fx"]] = npv_local * float(fx)
        return DataFrame.from_records([cfs])


### Multi-currency derivatives


class BaseXCS(BaseDerivative):
    """
    Base class with common methods for multi-currency ``Derivatives``.
    """

    _is_mtm = False

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # TODO set payment_lag_exchange and leg2.. in init here, including inherit and default lookup.
        return super().__init__(*args, **kwargs)

    @property
    def fx_fixings(self):
        return self._fx_fixings

    @fx_fixings.setter
    def fx_fixings(self, value):
        self._fx_fixings = value
        self._set_leg2_notional(value)

    def _initialise_fx_fixings(self, fx_fixings):
        """
        Sets the `fx_fixing` for non-mtm XCS instruments, which require only a single
        value.
        """
        if not self._is_mtm:
            self.pair = self.leg1.currency + self.leg2.currency
            # if self.fx_fixing is None this indicates the swap is unfixed and will be set
            # later. If a fixing is given this means the notional is fixed without any
            # further sensitivity, hence the downcast to a float below.
            if isinstance(fx_fixings, FXForwards):
                self.fx_fixings = float(
                    fx_fixings.rate(self.pair, self.leg2.periods[0].payment)
                )
            elif isinstance(fx_fixings, FXRates):
                self.fx_fixings = float(fx_fixings.rate(self.pair))
            elif isinstance(fx_fixings, (float, Dual, Dual2)):
                self.fx_fixings = float(fx_fixings)
            else:
                self._fx_fixings = None

    def _set_fx_fixings(self, fx):
        """
        Checks the `fx_fixings` and sets them according to given object if null.

        Used by ``rate`` and ``npv`` methods when ``fx_fixings`` are not
        initialised but required for pricing and can be inferred from an FX object.
        """
        if not self._is_mtm:  # then we manage the initial FX from the pricing object.
            if self.fx_fixings is None:
                if fx is None:
                    if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                        raise ValueError(
                            "`fx` is required when `fx_fixing` is not pre-set and "
                            "if rateslib option `no_fx_fixings_for_xcs` is set to "
                            "'raise'."
                        )
                    else:
                        fx_fixing = 1.0
                        if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                            warnings.warn(
                                "Using 1.0 for FX, no `fx` or `fx_fixing` given and "
                                "rateslib option `no_fx_fixings_for_xcs` is set to "
                                "'warn'.",
                                UserWarning,
                            )
                else:
                    fx_fixing = fx.rate(self.pair, self.leg2.periods[0].payment)
                self._set_leg2_notional(fx_fixing)
        else:
            self._set_leg2_notional(fx)

    def _set_leg2_notional(self, fx_arg: Union[float, FXForwards]):
        """
        Update the notional on leg2 (foreign leg) if the initial fx rate is unfixed.

        ----------
        fx_arg : float or FXForwards
            For non-MTM XCSs this input must be a float.
            The FX rate to use as the initial notional fixing.
            Will only update the leg if ``NonMtmXCS.fx_fixings`` has been initially
            set to `None`.

            For MTM XCSs this input must be ``FXForwards``.
            The FX object from which to determine FX rates used as the initial
            notional fixing, and to determine MTM cashflow exchanges.
        """
        if self._is_mtm:
            self.leg2._set_periods(fx_arg)
            self.leg2_notional = self.leg2.notional
        else:
            self.leg2_notional = self.leg1.notional * -fx_arg
            self.leg2.notional = self.leg2_notional
            self.leg2_amortization = self.leg1.amortization * -fx_arg
            self.leg2.amortization = self.leg2_amortization

    @property
    def _is_unpriced(self):
        if getattr(self, "_unpriced", None) is True:
            return True
        if self._fixed_rate_mixin and self._leg2_fixed_rate_mixin:
            # Fixed/Fixed where one leg is unpriced.
            if self.fixed_rate is None or self.leg2_fixed_rate is None:
                return True
            return False
        elif self._fixed_rate_mixin and self.fixed_rate is None:
            # Fixed/Float where fixed leg is unpriced
            return True
        elif self._float_spread_mixin and self.float_spread is None:
            # Float leg1 where leg1 is
            pass  # goto 2)
        else:
            return False

        # 2) leg1 is Float
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is None:
            return True
        elif self._leg2_float_spread_mixin and self.leg2_float_spread is None:
            return True
        else:
            return False

    def _set_pricing_mid(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
    ):
        leg: int = 1
        lookup = {
            1: ["_fixed_rate_mixin", "_float_spread_mixin"],
            2: ["_leg2_fixed_rate_mixin", "_leg2_float_spread_mixin"],
        }
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is None:
            # Fixed/Fixed or Float/Fixed
            leg = 2

        rate = self.rate(curves, solver, fx, leg=leg)
        if getattr(self, lookup[leg][0]):
            getattr(self, f"leg{leg}").fixed_rate = float(rate)
        elif getattr(self, lookup[leg][1]):
            getattr(self, f"leg{leg}").float_spread = float(rate)
        else:
            # this line should not be hit: internal code check
            raise AttributeError(
                "BaseXCS leg1 must be defined fixed or float."
            )  # pragma: no cover

    def npv(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        .. warning::

           If ``fx_fixing`` has not been set for the instrument requires
           ``fx`` as an FXForwards object to dynamically determine this.

        See :meth:`BaseDerivative.npv`.
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        base = self.leg1.currency if base is None else base

        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx)

        self._set_fx_fixings(fx)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().npv(curves, solver, fx, base, local)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset for next calculation
        return ret

    def rate(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
        leg: int = 1,
    ):
        """
        Return the mid-market pricing parameter of the XCS.

        Parameters
        ----------
        curves : list of Curves
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for leg1 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg1.
            - Forecasting :class:`~rateslib.curves.Curve` for leg2 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : FXForwards, optional
            The FX forwards object that is used to determine the initial FX fixing for
            determining ``leg2_notional``, if not specified at initialisation, and for
            determining mark-to-market exchanges on mtm XCSs.
        leg : int in [1, 2]
            The leg whose pricing parameter is to be determined.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Fixed legs have pricing parameter returned in percentage terms, and
        float legs have pricing parameter returned in basis point (bp) terms.

        If the ``XCS`` type is specified without a ``fixed_rate`` on any leg then an
        implied ``float_spread`` will return as its originaly value or zero since
        the fixed rate used
        for calculation is the implied mid-market rate including the
        current ``float_spread`` parameter.

        Examples
        --------
        """
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )

        if leg == 1:
            tgt_fore_curve, tgt_disc_curve = curves[0], curves[1]
            alt_fore_curve, alt_disc_curve = curves[2], curves[3]
        else:
            tgt_fore_curve, tgt_disc_curve = curves[2], curves[3]
            alt_fore_curve, alt_disc_curve = curves[0], curves[1]

        leg2 = 1 if leg == 2 else 2
        tgt_str, alt_str = "" if leg == 1 else "leg2_", "" if leg2 == 1 else "leg2_"
        tgt_leg, alt_leg = getattr(self, f"leg{leg}"), getattr(self, f"leg{leg2}")
        base = tgt_leg.currency

        _is_float_tgt_leg = "Float" in type(tgt_leg).__name__
        _is_float_alt_leg = "Float" in type(alt_leg).__name__
        if not _is_float_alt_leg and getattr(alt_leg, f"fixed_rate") is None:
            raise ValueError(
                "Cannot solve for a `fixed_rate` or `float_spread` where the "
                "`fixed_rate` on the non-solvable leg is None."
            )

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
        # Commercial use of this code, and/or copying and redistribution is prohibited.
        # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

        if not _is_float_tgt_leg and getattr(tgt_leg, f"fixed_rate") is None:
            # set the target fixed leg to a null fixed rate for calculation
            tgt_leg.fixed_rate = 0.0

        self._set_fx_fixings(fx)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        tgt_leg_npv = tgt_leg.npv(tgt_fore_curve, tgt_disc_curve, fx, base)
        alt_leg_npv = alt_leg.npv(alt_fore_curve, alt_disc_curve, fx, base)
        fx_a_delta = 1.0 if not tgt_leg._is_mtm else fx
        _ = tgt_leg._spread(
            -(tgt_leg_npv + alt_leg_npv), tgt_fore_curve, tgt_disc_curve, fx_a_delta
        )

        specified_spd = 0.0
        if _is_float_tgt_leg and not (getattr(tgt_leg, f"float_spread") is None):
            specified_spd = tgt_leg.float_spread
        elif not _is_float_tgt_leg:
            specified_spd = tgt_leg.fixed_rate * 100

        _ += specified_spd

        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc

        return _ if _is_float_tgt_leg else _ * 0.01

    def spread(self, *args, **kwargs):
        """
        Alias for :meth:`~rateslib.instruments.BaseXCS.rate`
        """
        return self.rate(*args, **kwargs)

    def cashflows(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
        base: Optional[str] = None,
    ):
        curves, fx = _get_curves_and_fx_maybe_from_solver(
            self.curves, solver, curves, fx
        )
        self._set_fx_fixings(fx)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().cashflows(curves, solver, fx, base)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc
        return ret


class NonMtmXCS(BaseXCS):
    """
    Create a non-mark-to-market cross currency swap (XCS) derivative composing two
    :class:`~rateslib.legs.FloatLegExchange` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fx_fixing : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for
        `fx_fixing` implies the notional on leg 2 is 110m USD. If `None` determines
        this dynamically later.
    float_spread : float or None
        The float spread applied in a simple way (after daily compounding) to leg 2.
        If `None` will be set to zero.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_float_spread : float or None
        The float spread applied in a simple way (after daily compounding) to leg 2.
        If `None` will be set to zero.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    Non-mtm cross currency swaps create identical yet opposite currency exchanges at
    the effective date and the payment termination date of the swap. There are no
    intermediate currency exchanges.

    .. note::

       Although non-MTM XCSs have an ``fx_fixing`` argument, which consists of a single,
       initial FX fixing, this is internally mapped to the ``fx_fixings`` attribute,
       which, for MTM XCSs, provides all the FX fixings throughout the swap.

    """

    _float_spread_mixin = True
    _leg2_float_spread_mixin = True
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        fx_fixing: Optional[Union[float, FXRates, FXForwards]] = None,
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_fixings: Optional[Union[float, list]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        self._leg2_float_spread = leg2_float_spread
        self._float_spread = float_spread
        self.leg1 = FloatLegExchange(
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLegExchange(
            float_spread=leg2_float_spread,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            spread_compound_method=leg2_spread_compound_method,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )
        self._initialise_fx_fixings(fx_fixing)

    # def _set_pricing_mid(
    #     self,
    #     curves: Optional[Union[Curve, str, list]] = None,
    #     solver: Optional[Solver] = None,
    #     fx: Optional[FXForwards] = None,
    # ):
    #     rate = self.rate(curves, solver, fx, leg=1)
    #     self.leg1.float_spread = float(rate)

    def _rate2(
        self,
        curve_domestic: Curve,
        disc_curve_domestic: Curve,
        curve_foreign: Curve,
        disc_curve_foreign: Curve,
        fx_rate: Union[float, Dual],
        fx_settlement: Optional[datetime] = None,
    ):  # pragma: no cover
        """
        Determine the mid-market floating spread on domestic leg 1, to equate leg 2.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.

        Returns
        -------
        BP Spread to leg 1 : Dual
        """
        npv = self.npv(
            curve_domestic,
            disc_curve_domestic,
            curve_foreign,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        leg1_analytic_delta = f_0 * self.leg1.analytic_delta(
            curve_domestic, disc_curve_domestic
        )
        spread = npv / leg1_analytic_delta
        return spread

    def _npv2(
        self,
        curve_domestic: Curve,
        disc_curve_domestic: Curve,
        curve_foreign: Curve,
        disc_curve_foreign: Curve,
        fx_rate: Union[float, Dual],
        fx_settlement: Optional[datetime] = None,
        base: str = None,
    ):  # pragma: no cover
        """
        Return the NPV of the non-mtm XCS.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.
        base : str, optional
            The base currency to express the NPV, either `"domestic"` or `"foreign"`.
            Set by default.
        """
        base = defaults.fx_swap_base if base is None else base
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        fx = forward_fx(
            self.effective,
            disc_curve_domestic,
            disc_curve_foreign,
            f_0,
        )
        self._set_leg2_notional(fx)
        leg1_npv = self.leg1.npv(curve_domestic)
        leg2_npv = self.leg2.npv(curve_foreign)

        if base == "foreign":
            return leg1_npv * f_0 + leg2_npv
        elif base == "domestic":
            return leg1_npv + leg2_npv / f_0
        else:
            raise ValueError('`base` should be either "domestic" or "foreign".')

    def _cashflows2(
        self,
        curve_domestic: Optional[Curve] = None,
        disc_curve_domestic: Optional[Curve] = None,
        curve_foreign: Optional[Curve] = None,
        disc_curve_foreign: Optional[Curve] = None,
        fx_rate: Optional[Union[float, Dual]] = None,
        fx_settlement: Optional[datetime] = None,
        base: Optional[str] = None,
    ):  # pragma: no cover
        """
        Return the properties of all legs used in calculating cashflows.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.
        base : str, optional
            The base currency to express the NPV, either `"domestic"` or `"foreign"`.
            Set by default.

        Returns
        -------
        DataFrame
        """
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        base = defaults.fx_swap_base if base is None else base
        if base == "foreign":
            d_fx, f_fx = f_0, 1.0
        elif base == "domestic":
            d_fx, f_fx = 1.0, 1.0 / f_0
        else:
            raise ValueError('`base` should be either "domestic" or "foreign".')
        self._set_leg2_notional(f_0)
        return concat(
            [
                self.leg1.cashflows(curve_domestic, disc_curve_domestic, d_fx),
                self.leg2.cashflows(curve_foreign, disc_curve_foreign, f_fx),
            ],
            keys=["leg1", "leg2"],
        )


class NonMtmFixedFloatXCS(BaseXCS):
    """
    Create a non-mark-to-market cross currency swap (XCS) derivative composing a
    :class:`~rateslib.legs.FixedLegExchange` and a
    :class:`~rateslib.legs.FloatLegExchange`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fx_fixing : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for `fx0`
        implies the notional on leg 2 is 110m USD. If `None` determines this
        dynamically.
    fixed_rate : float or None
        The fixed rate applied to leg 1.
        If `None` will be set to mid-market when curves are provided.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_float_spread2 : float or None
        The float spread applied in a simple way (after daily compounding) to leg 2.
        If `None` will be set to zero.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    Non-mtm cross currency swaps create identical yet opposite currency exchanges at
    the effective date and the payment termination date of the swap. There are no
    intermediate currency exchanges.
    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    def __init__(
        self,
        *args,
        fx_fixing: Optional[Union[float, FXRates, FXForwards]] = None,
        fixed_rate: Optional[float] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_fixings: Optional[Union[float, list]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        self._leg2_float_spread = leg2_float_spread
        self._fixed_rate = fixed_rate
        self.leg1 = FixedLegExchange(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLegExchange(
            float_spread=leg2_float_spread,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            spread_compound_method=leg2_spread_compound_method,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )
        self._initialise_fx_fixings(fx_fixing)


class NonMtmFixedFixedXCS(BaseXCS):
    """
    Create a non-mark-to-market cross currency swap (XCS) derivative composing two
    :class:`~rateslib.legs.FixedLegExchange` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fx_fixing : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for `fx0`
        implies the notional on leg 2 is 110m USD. If `None` determines this
        dynamically.
    fixed_rate : float or None
        The fixed rate applied to leg 1.
        If `None` will be set to mid-market when curves are provided.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_fixed_rate : float or None
        The fixed rate applied to leg 2.
        If `None` will be set to mid-market when curves are provided.
        Must set the ``fixed_rate`` on at least one leg.
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    Non-mtm cross currency swaps create identical yet opposite currency exchanges at
    the effective date and the payment termination date of the swap. There are no
    intermediate currency exchanges.
    """

    _fixed_rate_mixin = True
    _leg2_fixed_rate_mixin = True

    def __init__(
        self,
        *args,
        fx_fixing: Optional[Union[float, FXRates, FXForwards]] = None,
        fixed_rate: Optional[float] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_fixed_rate: Optional[float] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        self._leg2_fixed_rate = leg2_fixed_rate
        self._fixed_rate = fixed_rate
        self.leg1 = FixedLegExchange(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FixedLegExchange(
            fixed_rate=leg2_fixed_rate,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )
        self._initialise_fx_fixings(fx_fixing)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class XCS(BaseXCS):
    """
    Create a mark-to-market cross currency swap (XCS) derivative instrument.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fx_fixings : float, Dual, Dual2, list of such
        Specify a known initial FX fixing or a list of such for historical legs,
        where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for
        `fx_fixings` implies the notional on leg 2 is 110m USD.
        Fixings that are not specified will be calculated at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    float_spread : float or None
        The float spread applied in a simple way (after daily compounding) to leg 2.
        If `None` will be set to zero.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_float_spread : float or None
        The float spread applied in a simple way (after daily compounding) to leg 2.
        If `None` will be set to zero.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    Mtm cross currency swaps create notional exchanges on the foreign leg throughout
    the life of the derivative and adjust the notional on which interest is accrued.

    .. warning::

       ``Amortization`` is not used as an argument by ``XCS``.
    """

    _float_spread_mixin = True
    _leg2_float_spread_mixin = True
    _is_mtm = True
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        fx_fixings: Union[list, float, Dual, Dual2] = [],
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_fixings: Optional[Union[float, list]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        if fx_fixings is None:
            raise ValueError(
                "`fx_fixings` for MTM XCS should be entered as an empty list, not None."
            )
        self._fx_fixings = fx_fixings
        self._leg2_float_spread = leg2_float_spread
        self._float_spread = float_spread
        self.leg1 = FloatLegExchange(
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLegExchangeMtm(
            float_spread=leg2_float_spread,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            spread_compound_method=leg2_spread_compound_method,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            currency=self.leg2_currency,
            alt_currency=self.currency,
            alt_notional=-self.notional,
            fx_fixings=fx_fixings,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )

    # def _set_pricing_mid(
    #     self,
    #     curves: Optional[Union[Curve, str, list]] = None,
    #     solver: Optional[Solver] = None,
    #     fx: Optional[FXForwards] = None,
    # ):
    #     rate = self.rate(curves, solver, fx, leg=1)
    #     self.leg1.float_spread = float(rate)

    def _npv2(
        self,
        curve_domestic: Curve,
        disc_curve_domestic: Curve,
        curve_foreign: Curve,
        disc_curve_foreign: Curve,
        fx_rate: Union[float, Dual],
        fx_settlement: Optional[datetime] = None,
        base: str = None,
    ):  # pragma: no cover
        """
        Return the NPV of the non-mtm XCS.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.
        base : str, optional
            The base currency to express the NPV, either `"domestic"` or `"foreign"`.
            Set by default.
        """
        base = defaults.fx_swap_base if base is None else base
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        fx = forward_fx(
            self.effective,
            disc_curve_domestic,
            disc_curve_foreign,
            f_0,
        )
        self._set_leg2_notional(fx)
        leg1_npv = self.leg1.npv(curve_domestic)
        leg2_npv = self.leg2.npv(curve_foreign)

        if base == "foreign":
            return leg1_npv * f_0 + leg2_npv
        elif base == "domestic":
            return leg1_npv + leg2_npv / f_0
        else:
            raise ValueError('`base` should be either "domestic" or "foreign".')

    def _rate2(
        self,
        curve_domestic: Curve,
        disc_curve_domestic: Curve,
        curve_foreign: Curve,
        disc_curve_foreign: Curve,
        fx_rate: Union[float, Dual],
        fx_settlement: Optional[datetime] = None,
    ):  # pragma: no cover
        """
        Determine the mid-market floating spread on domestic leg 1, to equate leg 2.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.

        Returns
        -------
        BP Spread to leg 1 : Dual
        """
        npv = self.npv(
            curve_domestic,
            disc_curve_domestic,
            curve_foreign,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        leg1_analytic_delta = f_0 * self.leg1.analytic_delta(
            curve_domestic, disc_curve_domestic
        )
        spread = npv / leg1_analytic_delta
        return spread

    def _cashflows2(
        self,
        curve_domestic: Optional[Curve] = None,
        disc_curve_domestic: Optional[Curve] = None,
        curve_foreign: Optional[Curve] = None,
        disc_curve_foreign: Optional[Curve] = None,
        fx_rate: Optional[Union[float, Dual]] = None,
        fx_settlement: Optional[datetime] = None,
        base: Optional[str] = None,
    ):  # pragma: no cover
        """
        Return the properties of all legs used in calculating cashflows.

        Parameters
        ----------
        curve_domestic : Curve
            The forecast :class:`Curve` for domestic currency cashflows.
        disc_curve_domestic : Curve
            The discount :class:`Curve` for domestic currency cashflows.
        curve_foreign : Curve
            The forecasting :class:`Curve` for foreign currency cashflows.
        disc_curve_foreign : Curve
            The discounting :class:`Curve` for foreign currency cashflows.
        fx_rate : float, optional
            The FX rate for valuing cashflows.
        fx_settlement : datetime, optional
            The date for settlement of ``fx_rate``. If spot then should be input as T+2.
            If `None`, is assumed to be immediate settlement.
        base : str, optional
            The base currency to express the NPV, either `"domestic"` or `"foreign"`.
            Set by default.

        Returns
        -------
        DataFrame
        """
        f_0 = forward_fx(
            disc_curve_domestic.node_dates[0],
            disc_curve_domestic,
            disc_curve_foreign,
            fx_rate,
            fx_settlement,
        )
        base = defaults.fx_swap_base if base is None else base
        if base == "foreign":
            d_fx, f_fx = f_0, 1.0
        elif base == "domestic":
            d_fx, f_fx = 1.0, 1.0 / f_0
        else:
            raise ValueError('`base` should be either "domestic" or "foreign".')
        self._set_leg2_notional(f_0)
        return concat(
            [
                self.leg1.cashflows(curve_domestic, disc_curve_domestic, d_fx),
                self.leg2.cashflows(curve_foreign, disc_curve_foreign, f_fx),
            ],
            keys=["leg1", "leg2"],
        )


class FixedFloatXCS(BaseXCS):
    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True
    _is_mtm = True

    def __init__(
        self,
        *args,
        fx_fixings: Union[list, float, Dual, Dual2] = [],
        fixed_rate: Optional[float] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_float_spread: Optional[float] = None,
        leg2_fixings: Optional[Union[float, list]] = None,
        leg2_fixing_method: Optional[str] = None,
        leg2_method_param: Optional[int] = None,
        leg2_spread_compound_method: Optional[str] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        if fx_fixings is None:
            raise ValueError(
                "`fx_fixings` for MTM XCS should be entered as an empty list, not None."
            )
        self._fx_fixings = fx_fixings
        self._leg2_float_spread = leg2_float_spread
        self._fixed_rate = fixed_rate
        self.leg1 = FixedLegExchange(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FloatLegExchangeMtm(
            float_spread=leg2_float_spread,
            fixings=leg2_fixings,
            fixing_method=leg2_fixing_method,
            method_param=leg2_method_param,
            spread_compound_method=leg2_spread_compound_method,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            currency=self.leg2_currency,
            alt_currency=self.currency,
            alt_notional=-self.notional,
            fx_fixings=fx_fixings,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )


class FixedFixedXCS(BaseXCS):
    _fixed_rate_mixin = True
    _leg2_fixed_rate_mixin = True
    _is_mtm = True

    def __init__(
        self,
        *args,
        fx_fixings: Union[list, float, Dual, Dual2] = [],
        fixed_rate: Optional[float] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_fixed_rate: Optional[float] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        if fx_fixings is None:
            raise ValueError(
                "`fx_fixings` for MTM XCS should be entered as an empty list, not None."
            )
        self._fx_fixings = fx_fixings
        self._leg2_fixed_rate = leg2_fixed_rate
        self._fixed_rate = fixed_rate
        self.leg1 = FixedLegExchange(
            fixed_rate=fixed_rate,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FixedLegExchangeMtm(
            fixed_rate=leg2_fixed_rate,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            currency=self.leg2_currency,
            alt_currency=self.currency,
            alt_notional=-self.notional,
            fx_fixings=fx_fixings,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )


class FloatFixedXCS(BaseXCS):
    _float_spread_mixin = True
    _leg2_fixed_rate_mixin = True
    _is_mtm = True
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        fx_fixings: Union[list, float, Dual, Dual2] = [],
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_fixed_rate: Optional[float] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        if fx_fixings is None:
            raise ValueError(
                "`fx_fixings` for MTM XCS should be entered as an empty list, not None."
            )
        self._fx_fixings = fx_fixings
        self._leg2_fixed_rate = leg2_fixed_rate
        self._float_spread = float_spread
        self.leg1 = FloatLegExchange(
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            effective=self.effective,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            front_stub=self.front_stub,
            back_stub=self.back_stub,
            roll=self.roll,
            eom=self.eom,
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=self.payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            amortization=self.amortization,
            convention=self.convention,
        )
        self.leg2 = FixedLegExchangeMtm(
            fixed_rate=leg2_fixed_rate,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency=self.leg2_frequency,
            stub=self.leg2_stub,
            front_stub=self.leg2_front_stub,
            back_stub=self.leg2_back_stub,
            roll=self.leg2_roll,
            eom=self.leg2_eom,
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=self.leg2_payment_lag,
            payment_lag_exchange=leg2_payment_lag_exchange,
            currency=self.leg2_currency,
            alt_currency=self.currency,
            alt_notional=-self.notional,
            fx_fixings=fx_fixings,
            amortization=self.leg2_amortization,
            convention=self.leg2_convention,
        )


class FXSwap(BaseXCS):
    """
    Create an FX swap simulated via a :class:`NonMtmFixedFixedXCS`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fx_fixing : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for `fx0`
        implies the notional on leg 2 is 110m USD. If `None` determines this
        dynamically.
    points : float, optional
        The pricing parameter for the FX Swap, which will determine the implicit
        fixed rate on leg2.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule. Defaults to 0 for *FXSwaps*.
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule. Defaults to 0 for *FXSwaps*.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    ``leg2_notional`` is determined by the ``fx_fixing`` either initialised or at price
    time and the value of ``notional``. The argument value of ``leg2_notional`` does
    not impact calculations.

    .. note::

       *FXSwaps* can be initialised either *priced* or *unpriced*. Priced derivatives
       represent traded contracts with defined ``fx_fixing`` and ``points`` values.
       This is usual for valuing *npv* against current market conditions. Unpriced
       derivatives do not have a set ``fx_fixing`` nor ``points`` values. Any *rate*
       calculation should return the mid-market rate and an *npv* of zero.

    Examples
    --------
    To value the *FXSwap* we create *Curves* and :class:`~rateslib.fx.FXForwards`
    objects.

    .. ipython:: python

       usd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95}, id="usd")
       eur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97}, id="eur")
       eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.971}, id="eurusd")
       fxr = FXRates({"eurusd": 1.10}, settlement=dt(2022, 1, 3))
       fxf = FXForwards(
           fx_rates=fxr,
           fx_curves={"usdusd": usd, "eureur": eur, "eurusd": eurusd},
       )

    Then we define the *FXSwap*. This in an unpriced instrument.

    .. ipython:: python

       fxs = FXSwap(
           effective=dt(2022, 1, 17),
           termination=dt(2022, 4, 19),
           calendar="nyc",
           currency="usd",
           notional=1000000,
           leg2_currency="eur",
           curves=["usd", "usd", "eur", "eurusd"],
       )

    Now demonstrate the :meth:`~rateslib.instruments.FXSwap.npv` and
    :meth:`~rateslib.instruments.FXSwap.rate` methods:

    .. ipython:: python

       fxs.npv(curves=[usd, usd, eur, eurusd], fx=fxf)
       fxs.rate(curves=[usd, usd, eur, eurusd], fx=fxf)

    In the case of *FXSwaps*, whose mid-market price is the difference between two
    forward FX rates we can also derive this quantity using the independent
    :meth:`FXForwards.swap<rateslib.fx.FXForwards.swap>` method. In this example
    the numerical differences are caused by different calculation methods. The
    difference here equates to a tolerance of 1e-8, or $1 per $100mm.

    .. ipython:: python

       fxf.swap("usdeur", [dt(2022, 1, 17), dt(2022, 4, 19)])

    """

    _fixed_rate_mixin = True
    _leg2_fixed_rate_mixin = True
    _unpriced = True

    def __init__(
        self,
        *args,
        fx_fixing: Optional[Union[float, FXRates, FXForwards]] = None,
        points: Optional[float] = None,
        payment_lag_exchange: Optional[int] = None,
        leg2_payment_lag_exchange: Optional[int] = "inherit",
        **kwargs,
    ):
        if fx_fixing is None and points is not None:
            raise ValueError(
                "Cannot set `points` on FXSwap without giving an `fx_fixing`."
            )
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange == "inherit":
            leg2_payment_lag_exchange = payment_lag_exchange
        self._fixed_rate = 0.0
        self.leg1 = FixedLegExchange(
            fixed_rate=0.0,
            effective=self.effective,
            termination=self.termination,
            frequency="Z",
            modifier=self.modifier,
            calendar=self.calendar,
            payment_lag=payment_lag_exchange,
            payment_lag_exchange=payment_lag_exchange,
            notional=self.notional,
            currency=self.currency,
            convention=self.convention,
        )
        self.leg2 = FixedLegExchange(
            fixed_rate=None,
            effective=self.leg2_effective,
            termination=self.leg2_termination,
            frequency="Z",
            modifier=self.leg2_modifier,
            calendar=self.leg2_calendar,
            payment_lag=leg2_payment_lag_exchange,
            payment_lag_exchange=leg2_payment_lag_exchange,
            notional=self.leg2_notional,
            currency=self.leg2_currency,
            convention=self.leg2_convention,
        )
        self._initialise_fx_fixings(fx_fixing)
        self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._unpriced = False
        self._points = value
        self._leg2_fixed_rate = None
        if value is not None:
            fixed_rate = (
                value
                * -self.notional
                / (self.leg2.periods[1].dcf * 100 * self.leg2.periods[1].notional)
            )
            self.leg2_fixed_rate = fixed_rate

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International

    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _set_pricing_mid(
        self,
        curves: Optional[Union[Curve, str, list]] = None,
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
    ):
        points = self.rate(curves, solver, fx)
        self.points = points
        self._unpriced = True  # setting pricing mid does not define a priced instrument

    def rate(
        self,
        curves: Union[Curve, str, list],
        solver: Optional[Solver] = None,
        fx: Optional[FXForwards] = None,
        fixed_rate: bool = False,
    ):
        """
        Return the mid-market pricing parameter of the FXSwap.

        Parameters
        ----------
        curves : list of Curves
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for leg1 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg1.
            - Forecasting :class:`~rateslib.curves.Curve` for leg2 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : FXForwards, optional
            The FX forwards object that is used to determine the initial FX fixing for
            determining ``leg2_notional``, if not specified at initialisation, and for
            determining mark-to-market exchanges on mtm XCSs.
        fixed_rate : bool
            Whether to return the fixed rate for the leg or the FX swap points price.

        Returns
        -------
        float, Dual or Dual2
        """
        leg2_fixed_rate = super().rate(curves, solver, fx, leg=2)
        if fixed_rate:
            return leg2_fixed_rate
        cf = self.leg2.notional * leg2_fixed_rate * 0.01 * self.leg2.periods[1].dcf
        # fwd_fx = (cf + self.leg2.notional) / -self.leg1.notional
        # ini_fx = self.leg2.notional / -self.leg1.notional
        ## TODO decide how to price mid-market rates when ini fx is struck but
        ## there is no fixed points, i,e the FXswap is semi-determined, which is
        ## not a real instrument.
        return (cf / -self.leg1.notional) * 10000


### Generic Instruments


class Spread(Sensitivities):
    """
    A spread instrument defined as the difference in rate between two ``Instruments``.

    The ``Instruments`` used must share common pricing arguments. See notes.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s.
    instrument2 : Instrument
        The second instrument, usually the longest tenor, e.g. 10Y in 5s10s.

    Notes
    -----
    When using :class:`Spread` both ``Instruments`` must be of the same type
    with shared pricing arguments for their methods. If this is not true
    consider using the :class:`SpreadX`, cross spread ``Instrument``.

    Examples
    --------
    Creating a dynamic :class:`Spread` where the instruments are dynamically priced,
    and each share the pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q")
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q")
       spread = Spread(irs1, irs2)
       spread.npv(curve1)
       spread.rate(curve1)
       spread.cashflows(curve1)

    Creating an assigned :class:`Spread`, where each ``Instrument`` has its own
    assigned pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.99, dt(2022, 7, 1):0.98})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q", curves=curve1)
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q", curves=curve2)
       spread = Spread(irs1, irs2)
       spread.npv()
       spread.rate()
       spread.cashflows()
    """

    _rate_scalar = 1.0

    def __init__(self, instrument1, instrument2):
        self.instrument1 = instrument1
        self.instrument2 = instrument2

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----

        If the argument ``local`` is added to return a dict of currencies, ensure
        that this is added as a **keyword** argument and not a positional argument.
        I.e. use `local=True`.
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)
                for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    # def npv(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument1.npv(*args1) + self.instrument2.npv(*args2)

    def rate(self, *args, **kwargs):
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        return leg2_rate - leg1_rate

    # def rate(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument2.rate(*args2) - self.instrument1.rate(*args1)

    def cashflows(self, *args, **kwargs):
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2"],
        )


# class SpreadX:
#     pass


class Fly(Sensitivities):
    """
    A butterfly instrument which is, mechanically, the spread of two spread instruments.

    The ``Instruments`` used must share common dynamic pricing arguments
    or be statically created. See notes XXXX link o pricingmechanisms.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s15s.
    instrument2 : Instrument
        The second instrument, usually the mid-length tenor, e.g. 10Y in 5s10s15s.
    instrument3 : Instrument
        The third instrument, usually the longest tenor, e.g. 15Y in 5s10s15s.

    Notes
    -----
    When using :class:`Spread` both ``Instruments`` must be of the same type
    with shared pricing arguments for their methods. If this is not true
    consider using the :class:`FlyX`, cross ``Instrument``.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    _rate_scalar = 1.0

    def __init__(self, instrument1, instrument2, instrument3):
        self.instrument1 = instrument1
        self.instrument2 = instrument2
        self.instrument3 = instrument3

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        leg3_npv = self.instrument3.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) + leg3_npv.get(k, 0)
                for k in set(leg1_npv) | set(leg2_npv) | set(leg3_npv)
            }
        else:
            return leg1_npv + leg2_npv + leg3_npv

    def rate(self, *args, **kwargs):
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        leg3_rate = self.instrument3.rate(*args, **kwargs)
        return -leg3_rate + 2 * leg2_rate - leg1_rate

    def cashflows(self, *args, **kwargs):
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
                self.instrument3.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2", "instrument3"],
        )


# class FlyX:
#     """
#     A butterly instrument which is the spread of two spread instruments
#     """
#     def __init__(self, instrument1, instrument2, instrument3):
#         self.instrument1 = instrument1
#         self.instrument2 = instrument2
#         self.instrument3 = instrument3
#
#     def npv(self, *args, **kwargs):
#         if len(args) == 0:
#             args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
#             args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
#             args3 = (kwargs.get("curve3", None), kwargs.get("disc_curve3", None))
#         else:
#             args1 = args
#             args2 = args
#             args3 = args
#         return self.instrument1.npv(*args1) + self.instrument2.npv(*args2) + self.instrument3.npv(*args3)
#
#     def rate(self, *args, **kwargs):
#         if len(args) == 0:
#             args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
#             args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
#             args3 = (kwargs.get("curve3", None), kwargs.get("disc_curve3", None))
#         else:
#             args1 = args
#             args2 = args
#             args3 = args
#         return 2 * self.instrument2.rate(*args2) - self.instrument1.rate(*args1) - self.instrument3.rate(*args3)


def _instrument_npv(instrument, *args, **kwargs):
    return instrument.npv(*args, **kwargs)


class Portfolio(Sensitivities):
    # TODO document portfolio

    def __init__(self, instruments):
        self.instruments = instruments

    def npv(self, *args, **kwargs):
        # TODO do not permit a mixing of currencies.
        # TODO look at legs.npv where args len is used.

        if defaults.pool == 1:
            return self._npv_single_core(*args, **kwargs)

        from multiprocessing import Pool
        from functools import partial
        func = partial(_instrument_npv, *args, **kwargs)
        p = Pool(defaults.pool)
        results = p.map(func, self.instruments)

        if kwargs.get("local", False):
            _ = DataFrame(results).fillna(0.0)
            _ = _.sum()
            ret = _.to_dict()

            # ret = {}
            # for result in results:
            #     for ccy in result:
            #         if ccy in ret:
            #             ret[ccy] += result[ccy]
            #         else:
            #             ret[ccy] = result[ccy]

        else:
            ret = sum(results)

        return ret

    def _npv_single_core(self, *args, **kwargs):
        if kwargs.get("local", False):
            ret = {}
            for instrument in self.instruments:
                i_npv = instrument.npv(*args, **kwargs)
                for ccy in i_npv:
                    if ccy in ret:
                        ret[ccy] += i_npv[ccy]
                    else:
                        ret[ccy] = i_npv[ccy]
        else:
            ret = 0
            for instrument in self.instruments:
                ret += instrument.npv(*args, **kwargs)
        return ret


def forward_fx(
    date: datetime,
    curve_domestic: Curve,
    curve_foreign: Curve,
    fx_rate: Union[float, Dual],
    fx_settlement: Optional[datetime] = None,
) -> Dual:
    """
    Return the adjusted FX rate based on interest rate parity.

    .. deprecated:: 0.0
       See notes.

    Parameters
    ----------
    date : datetime
        The target date to determine the adjusted FX rate for.
    curve_domestic : Curve
        The discount curve for the domestic currency. Should be FX swap / XCS adjusted.
    curve_foreign : Curve
        The discount curve for the foreign currency. Should be FX swap / XCS consistent
        with ``domestic curve``.
    fx_rate : float or Dual
        The known FX rate, typically spot FX given with a spot settlement date.
    fx_settlement : datetime, optional
        The date the given ``fx_rate`` will settle, i.e spot T+2. If `None` is assumed
        to be immediate settlement, i.e. date upon which both ``curves`` have a DF
        of precisely 1.0. Method is more efficient if ``fx_rate`` is given for
        immediate settlement.

    Returns
    -------
    float, Dual, Dual2

    Notes
    -----
    We use the formula,

    .. math::

       (EURUSD) f_i = \\frac{(EUR:USD-CSA) w^*_i}{(USD:USD-CSA) v_i} F_0 = \\frac{(EUR:EUR-CSA) v^*_i}{(USD:EUR-CSA) w_i} F_0

    where :math:`w` is a cross currency adjusted discount curve and :math:`v` is the
    locally derived discount curve in a given currency, and `*` denotes the domestic
    currency. :math:`F_0` is the immediate FX rate, i.e. aligning with the initial date
    on curves such that discounts factors are precisely 1.0.

    This implies that given the dates and rates supplied,

    .. math::

       f_i = \\frac{w^*_iv_j}{v_iw_j^*} f_j = \\frac{v^*_iw_j}{w_iv_j^*} f_j

    where `j` denotes the settlement date provided.

    **Deprecated**

    This method is deprecated. It should be replaced by the use of
    :class:`~rateslib.fx.FXForwards` objects. See examples.

    Examples
    --------
    Using this function directly.

    .. ipython:: python

       domestic_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96})
       foreign_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
       forward_fx(
           date=dt(2022, 7, 1),
           curve_domestic=domestic_curve,
           curve_foreign=foreign_curve,
           fx_rate=2.0,
           fx_settlement=dt(2022, 1, 3)
       )

    Replacing this deprecated function with object-oriented methods.

    .. ipython:: python

       fxr = FXRates({"usdgbp": 2.0}, settlement=dt(2022, 1, 3))
       fxf = FXForwards(fxr, {
           "usdusd": domestic_curve,
           "gbpgbp": foreign_curve,
           "gbpusd": foreign_curve,
       })
       fxf.rate("usdgbp", dt(2022, 7, 1))
    """
    if date == fx_settlement:
        return fx_rate
    elif date == curve_domestic.node_dates[0] and fx_settlement is None:
        return fx_rate

    _ = curve_domestic[date] / curve_foreign[date]
    if fx_settlement is not None:
        _ *= curve_foreign[fx_settlement] / curve_domestic[fx_settlement]
    _ *= fx_rate
    return _


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


# def _ytm_quadratic_converger(f, y0, y1, y2, tol=1e-9):
#     """
#     Convert a price from yield function `f` into a quadratic approximation and
#     determine the root, yield, which matches the target price.
#     """
#     _A = np.array([[y0**2, y0, 1], [y1**2, y1, 1], [y2**2, y2, 1]])
#     _b = np.array([f(y0), f(y1), f(y2)])[:, None]
#
#     # check tolerance from previous recursive estimations
#     if abs(_b[1, 0]) < tol:
#         return y1
#
#     c = np.linalg.solve(_A, _b)
#
#     yield1 = ((-c[1] + sqrt(c[1]**2 - 4 * c[0] * c[2])) / (2 * c[0]))[0]
#     yield2 = ((-c[1] - sqrt(c[1]**2 - 4 * c[0] * c[2])) / (2 * c[0]))[0]
#     z1, z2 = f(yield1), f(yield2)
#
#     # make a linear guess at new quadratic solution
#     approx_yield = yield1 - (yield2 - yield1) * z1 / (z2 - z1)
#     if abs(z1) < abs(z2):
#         soln_yield = yield1
#         if abs(z1) < tol:
#             return soln_yield
#     else:
#         soln_yield = yield2
#         if abs(z2) < tol:
#             return soln_yield
#     return _ytm_quadratic_converger(
#         f,
#         approx_yield - max(10 * (approx_yield - soln_yield), 0.001),
#         approx_yield,
#         approx_yield + max(10 * (approx_yield - soln_yield), 0.001),
#         tol
#     )


def _ytm_quadratic_converger2(f, y0, y1, y2, f0=None, f1=None, f2=None, tol=1e-9):
    """
    Convert a price from yield function `f` into a quadratic approximation and
    determine the root, yield, which matches the target price.
    """
    # allow function values to be passed recursively to avoid re-calculation
    f0 = f(y0) if f0 is None else f0
    f1 = f(y1) if f1 is None else f1
    f2 = f(y2) if f2 is None else f2

    if f0 < 0 and f1 < 0 and f2 < 0:
        # reassess initial values
        return _ytm_quadratic_converger2(
            f, 2 * y0 - y2, y1 - y2 + y0, y0, None, None, f0, tol
        )
    elif f0 > 0 and f1 > 0 and f2 > 0:
        return _ytm_quadratic_converger2(
            f, y2, y1 + 1 * (y2 - y0), y2 + 2 * (y2 - y0), f2, None, None, tol
        )

    _b = np.array([y0, y1, y2])[:, None]
    # check tolerance from previous recursive estimations
    for i, f_ in enumerate([f0, f1, f2]):
        if abs(f_) < tol:
            return _b[i, 0]

    _A = np.array([[f0**2, f0, 1], [f1**2, f1, 1], [f2**2, f2, 1]])
    c = np.linalg.solve(_A, _b)
    y = c[2, 0]
    f_ = f(y)

    pad = min(tol * 1e8, 0.0001, abs(f_ * 1e4))  # avoids singular matrix error
    if y <= y0:
        # line not hit due to reassessment of initial vars?
        return _ytm_quadratic_converger2(
            f, 2 * y - y0 - pad, y, y0 + pad, None, f_, None, tol
        )  # pragma: no cover
    elif y0 < y <= y1:
        if (y - y0) < (y1 - y):
            return _ytm_quadratic_converger2(
                f, y0 - pad, y, 2 * y - y0 + pad, None, f_, None, tol
            )
        else:
            return _ytm_quadratic_converger2(
                f, 2 * y - y1 - pad, y, y1 + pad, None, f_, None, tol
            )
    elif y1 < y <= y2:
        if (y - y1) < (y2 - y):
            return _ytm_quadratic_converger2(
                f, y1 - pad, y, 2 * y - y1 + pad, None, f_, None, tol
            )
        else:
            return _ytm_quadratic_converger2(
                f, 2 * y - y2 - pad, y, y2 + pad, None, f_, None, tol
            )
    else:  # y2 < y:
        # line not hit due to reassessmemt of initial vars?
        return _ytm_quadratic_converger2(
            f, y2 - pad, y, 2 * y - y2 + pad, None, f_, None, tol
        )  # pragma: no cover


def _brents(f, x0, x1, max_iter=50, tolerance=1e-9):  # pragma: no cover
    """
    Alternative yield converger as an alternative to ytm_converger

    Unused currently within the library
    """
    fx0 = f(x0)
    fx1 = f(x1)

    if float(fx0 * fx1) > 0:
        raise ValueError(
            "`brents` must initiate from function values with opposite signs."
        )

    if abs(fx0) < abs(fx1):
        x0, x1 = x1, x0
        fx0, fx1 = fx1, fx0

    x2, fx2 = x0, fx0

    mflag = True
    steps_taken = 0

    while steps_taken < max_iter and abs(x1 - x0) > tolerance:
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)

        if fx0 != fx2 and fx1 != fx2:
            L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
            L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
            L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
            new = L0 + L1 + L2

        else:
            new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

        if (
            (float(new) < float((3 * x0 + x1) / 4) or float(new) > float(x1))
            or (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2))
            or (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2))
            or (mflag == True and (abs(x1 - x2)) < tolerance)
            or (mflag == False and (abs(x2 - d)) < tolerance)
        ):
            new = (x0 + x1) / 2
            mflag = True

        else:
            mflag = False

        fnew = f(new)
        d, x2 = x2, x1

        if float(fx0 * fnew) < 0:
            x1 = new
        else:
            x0 = new

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0

        steps_taken += 1

    return x1, steps_taken
