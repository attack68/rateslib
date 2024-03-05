# -*- coding: utf-8 -*-

# Sphinx substitutions
"""
.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from rateslib.curves import Curve
   from datetime import datetime as dt
   from pandas import Series, date_range, option_context
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
from datetime import datetime, timedelta
from typing import Optional, Union
import abc
import warnings
from functools import partial

# from math import sqrt

import numpy as np

# from scipy.optimize import brentq
from pandas.tseries.offsets import CustomBusinessDay
from pandas import DataFrame, concat, Series, MultiIndex, isna

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.calendars import add_tenor, get_calendar, dcf, _get_years_and_months, _DCF1d

from rateslib.curves import Curve, index_left, LineCurve, CompositeCurve, IndexCurve, average_rate
from rateslib.solver import Solver
from rateslib.periods import (
    Cashflow,
    FloatPeriod,
    _get_fx_and_base,
    IndexMixin,
    _disc_from_curve,
    _disc_maybe_from_curve,
)
from rateslib.legs import (
    FixedLeg,
    FloatLeg,
    FloatLegMtm,
    FixedLegMtm,
    ZeroFloatLeg,
    ZeroFixedLeg,
    ZeroIndexLeg,
    IndexFixedLeg,
)
from rateslib.dual import Dual, Dual2, DualTypes, dual_log, gradient
from rateslib.fx import FXForwards, FXRates, forward_fx


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_curve_from_solver(curve, solver):
    if isinstance(curve, dict):
        # When supplying a curve as a dictionary of curves (for IBOR stubs) use recursion
        return {k: _get_curve_from_solver(v, solver) for k, v in curve.items()}
    elif getattr(curve, "_is_proxy", False):
        # TODO: (mid) consider also adding CompositeCurves as exceptions under the same rule
        # proxy curves exist outside of solvers but still have Dual variables associated
        # with curves inside the solver, so can still generate risks to calibrating
        # instruments
        return curve
    elif isinstance(curve, str):
        return solver.pre_curves[curve]
    elif curve is NoInput.blank or curve is None:
        # pass through a None curve. This will either raise errors later or not be needed
        return NoInput(0)
    else:
        try:
            # it is a safeguard to load curves from solvers when a solver is
            # provided and multiple curves might have the same id
            _ = solver.pre_curves[curve.id]
            if id(_) != id(curve):  # Python id() is a memory id, not a string label id.
                raise ValueError(
                    "A curve has been supplied, as part of ``curves``, which has the same "
                    f"`id` ('{curve.id}'),\nas one of the curves available as part of the "
                    "Solver's collection but is not the same object.\n"
                    "This is ambiguous and cannot price.\n"
                    "Either refactor the arguments as follows:\n"
                    "1) remove the conflicting curve: [curves=[..], solver=<Solver>] -> "
                    "[curves=None, solver=<Solver>]\n"
                    "2) change the `id` of the supplied curve and ensure the rateslib.defaults "
                    "option 'curve_not_in_solver' is set to 'ignore'.\n"
                    "   This will remove the ability to accurately price risk metrics."
                )
            return _
        except AttributeError:
            raise AttributeError(
                "`curve` has no attribute `id`, likely it not a valid object, got: "
                f"{curve}.\nSince a solver is provided have you missed labelling the `curves` "
                f"of the instrument or supplying `curves` directly?"
            )
        except KeyError:
            if defaults.curve_not_in_solver == "ignore":
                return curve
            elif defaults.curve_not_in_solver == "warn":
                warnings.warn("`curve` not found in `solver`.", UserWarning)
                return curve
            else:
                raise ValueError("`curve` must be in `solver`.")


def _get_curves_fx_and_base_maybe_from_solver(
    curves_attr: Optional[Union[Curve, str, list]],
    solver: Optional[Solver],
    curves: Optional[Union[Curve, str, list]],
    fx: Optional[Union[float, FXRates, FXForwards]],
    base: Optional[str],
    local_ccy: Optional[str],
) -> tuple:
    """
    Parses the ``solver``, ``curves``, ``fx`` and ``base`` arguments in combination.

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
    tuple : (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting), fx, base

    Notes
    -----
    If only one curve is given this is used as all four curves.

    If two curves are given the forecasting curve is used as the forecasting
    curve on both legs and the discounting curve is used as the discounting
    curve for both legs.

    If three curves are given the single discounting curve is used as the
    discounting curve for both legs.
    """

    # First process `base`.
    if fx is NoInput.blank and base is NoInput.blank:
        # base will not be inherited from a 2nd level inherited object, i.e.
        # from solver.fx, to preserve single currency instruments being defaulted
        # to their local currency.
        base_ = local_ccy
    elif isinstance(fx, (FXRates, FXForwards)) and base is NoInput.blank:
        base_ = fx.base
    else:
        base_ = base

    # Second process `fx`
    if fx is NoInput.blank:
        if solver is NoInput.blank:
            fx_ = NoInput(0)
            # fx_ = 1.0
        elif solver is not NoInput.blank:
            if solver.fx is NoInput.blank:
                fx_ = NoInput(0)
                # fx_ = 1.0
            else:
                fx_ = solver.fx
    else:
        fx_ = fx
        if (
            solver is not NoInput.blank
            and solver.fx is not NoInput.blank
            and id(fx) != id(solver.fx)
        ):
            warnings.warn(
                "Solver contains an `fx` attribute but an `fx` argument has been "
                "supplied which will be used but is not the same. This can lead "
                "to calculation inconsistencies, mathematically.",
                UserWarning,
            )

    if curves is NoInput.blank and curves_attr is NoInput.blank:
        return (NoInput(0), NoInput(0), NoInput(0), NoInput(0)), fx_, base_
    elif curves is NoInput.blank:
        curves = curves_attr

    if isinstance(curves, (Curve, str, CompositeCurve, dict)):
        curves = [curves]
    if solver is NoInput.blank:

        def check_curve(curve):
            if isinstance(curve, str):
                raise ValueError("`curves` must contain Curve, not str, if `solver` not given.")
            elif curve is None or curve is NoInput(0):
                return NoInput(0)
            elif isinstance(curve, dict):
                return {k: check_curve(v) for k, v in curve.items()}
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

    return curves_, fx_, base_


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
#     if fx is NoInput.blank:
#         if solver is NoInput.blank:
#             fx_ = None
#             # fx_ = 1.0
#         elif solver is not NoInput.blank:
#             if solver.fx is NoInput.blank:
#                 fx_ = None
#                 # fx_ = 1.0
#             else:
#                 fx_ = solver.fx
#     else:
#         fx_ = fx
#
#     if curves is NoInput.blank:
#         return (None, None, None, None), fx_
#
#     if isinstance(curves, (Curve, str)):
#         curves = [curves]
#     if solver is NoInput.blank:
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
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Calculate delta risk of an *Instrument* against the calibrating instruments in a
        :class:`~rateslib.curves.Solver`.

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
            The :class:`~rateslib.solver.Solver` that calibrates
            *Curves* from given *Instruments*.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore ``base`` - this is equivalent to setting ``base`` to *None*.
            Included only for argument signature consistent with *npv*.

        Returns
        -------
        DataFrame
        """
        if solver is NoInput.blank:
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv = self.npv(curves, solver, fx, base, local=True)
        _, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0), solver, NoInput(0), fx, base, NoInput(0)
        )
        if local:
            base_ = NoInput(0)
        return solver.delta(npv, base_, fx_)

    def gamma(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Calculate cross-gamma risk of an *Instrument* against the calibrating instruments of a
        :class:`~rateslib.curves.Solver`.

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
            The :class:`~rateslib.solver.Solver` that calibrates
            *Curves* from given *Instruments*.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore ``base``. This is equivalent to setting ``base`` to *None*.
            Included only for argument signature consistent with *npv*.

        Returns
        -------
        DataFrame
        """
        if solver is NoInput.blank:
            raise ValueError("`solver` is required for delta/gamma methods.")
        _, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0), solver, NoInput(0), fx, base, NoInput(0)
        )
        if local:
            base_ = NoInput(0)

        # store original order
        if fx_ is not NoInput.blank:
            _ad2 = fx_._ad
            fx_._set_ad_order(2)

        _ad1 = solver._ad
        solver._set_ad_order(2)

        npv = self.npv(curves, solver, fx_, base_, local=True)
        grad_s_sT_P = solver.gamma(npv, base_, fx_)

        # reset original order
        if fx_ is not NoInput.blank:
            fx_._set_ad_order(_ad2)
        solver._set_ad_order(_ad1)

        return grad_s_sT_P

    def cashflows_table(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        cashflows = self.cashflows(curves, solver, fx, base)
        cashflows = cashflows[
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
                defaults.headers["cashflow"],
            ]
        ]
        _ = cashflows.groupby(
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
            ],
            dropna=False,
        )
        _ = _.sum().unstack([0, 1]).droplevel(0, axis=1)
        _.columns.names = ["local_ccy", "collateral_ccy"]
        _.index.names = ["payment"]
        _ = _.sort_index(ascending=True, axis=0).fillna(0.0)
        return _


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
        # if getattr(self, "_float_mixin_leg", None) is NoInput.blank:
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
           ``index_base`` and ``leg2_index_base`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
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
           ``index_base`` and ``leg2_index_base`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        return self._leg2_index_base

    @leg2_index_base.setter
    def leg2_index_base(self, value):
        if not self._leg2_index_base_mixin:
            raise AttributeError("Cannot set `leg2_index_base` for this Instrument.")
        self._leg2_index_base = value
        self.leg2.index_base = value

    @abc.abstractmethod
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

    @abc.abstractmethod
    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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

           irs.cashflows([curve], fx=fxr)
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        df1 = self.leg1.cashflows(curves[0], curves[1], fx_, base_)
        df2 = self.leg2.cashflows(curves[2], curves[3], fx_, base_)
        # filter empty or all NaN
        dfs_filtered = [_ for _ in [df1, df2] if not (_.empty or isna(_).all(axis=None))]

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = concat(dfs_filtered, keys=["leg1", "leg2"])
        return _

    @abc.abstractmethod
    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
           irs.npv([curve], fx=fxr)
           irs.npv([curve], fx=fxr, base="gbp")
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
        leg2_npv = self.leg2.npv(curves[2], curves[3], fx_, base_, local)
        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
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


class Value(BaseMixin):
    """
    A null *Instrument* which can be used within a :class:`~rateslib.solver.Solver`
    to directly parametrise a *Curve* node, via some calculated value.

    Parameters
    ----------
    effective : datetime
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    curves : Curve, LineCurve, str or list of such, optional
        A single :class:`~rateslib.curves.Curve`,
        :class:`~rateslib.curves.LineCurve` or id or a
        list of such. Only uses the first *Curve* in a list.
    convention : str, optional,
        Day count convention used with certain ``metric``.
    metric : str in {"curve_value", "index_value", "cc_zero_rate"}, optional
        Configures which value to extract from the *Curve*.

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
        convention: Union[str, NoInput] = NoInput(0),
        metric: str = "curve_value",
        curves: Optional[Union[list, str, Curve]] = None,
    ):
        self.effective = effective
        self.curves = curves
        self.convention = defaults.convention if convention is NoInput.blank else convention
        self.metric = metric.lower()

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return a value derived from a *Curve*.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            Uses only one *Curve*, the one given or the first in the list.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            Not used.
        base : str, optional
            Not used.
        metric: str in {"curve_value", "index_value", "cc_zero_rate"}, optional
            Configures which type of value to return from the applicable *Curve*.

        Returns
        -------
        float, Dual, Dual2

        """
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, NoInput(0), NoInput(0), "_"
        )
        metric = self.metric if metric is NoInput.blank else metric.lower()
        if metric == "curve_value":
            return curves[0][self.effective]
        elif metric == "cc_zero_rate":
            if curves[0]._base_type != "dfs":
                raise TypeError("`curve` used with `metric`='cc_zero_rate' must be discount factor based.")
            dcf_ = dcf(curves[0].node_dates[0], self.effective, self.convention)
            _ = (dual_log(curves[0][self.effective]) / -dcf_) * 100
            return _
        elif metric == "index_value":
            if not isinstance(curves[0], IndexCurve):
                raise TypeError("`curve` used with `metric`='index_value' must be type IndexCurve.")
            _ = curves[0].index_value(self.effective)
            return _
        raise ValueError("`metric`must be in {'curve_value', 'cc_zero_rate', 'index_value'}.")

    def npv(self, *args, **kwargs):
        raise NotImplementedError("`Value` instrument has no concept of NPV.")

    def cashflows(self, *args, **kwargs):
        raise NotImplementedError("`Value` instrument has no concept of cashflows.")

    def analytic_delta(self, *args, **kwargs):
        raise NotImplementedError("`Value` instrument has no concept of analytic delta.")


class FXExchange(Sensitivities, BaseMixin):
    """
    Create a simple exchange of two currencies.

    Parameters
    ----------
    settlement : datetime
        The date of the currency exchange.
    currency : str
        The currency of the cashflow for which ``notional`` is applicable (3-digit code).
    leg2_currency : str
        The currency of the cashflow on the alternate *Leg*.
    fx_rate : float, optional
        The FX rate used to derive the notional exchange on *Leg2*.
    notional : float
        The cashflow amount for the initial currency.
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
    """

    def __init__(
        self,
        settlement: datetime,
        currency: str,
        leg2_currency: str,
        fx_rate: Union[float, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
    ):
        self.curves = curves
        self.settlement = settlement
        self.pair = f"{currency.lower()}{leg2_currency.lower()}"
        self.leg1 = Cashflow(
            notional=defaults.notional if notional is NoInput.blank else notional,
            currency=currency.lower(),
            payment=settlement,
            stub_type="Exchange",
            rate=NoInput(0),
        )
        self.leg2 = Cashflow(
            notional=1.0,  # will be determined by setting fx_rate
            currency=leg2_currency.lower(),
            payment=settlement,
            stub_type="Exchange",
            rate=fx_rate,
        )
        self.fx_rate = fx_rate

    @property
    def fx_rate(self):
        return self._fx_rate

    @fx_rate.setter
    def fx_rate(self, value):
        self._fx_rate = value
        self.leg2.notional = 0.0 if value is NoInput.blank else value * -self.leg1.notional
        self.leg2._rate = value

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
    ):
        if self.fx_rate is NoInput.blank:
            mid_market_rate = self.rate(curves, solver, fx)
            self.fx_rate = float(mid_market_rate)
            self._fx_rate = NoInput(0)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the *FXExchange* by summing legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.npv>`
        """
        self._set_pricing_mid(curves, solver, fx)

        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        if fx_ is NoInput.blank:
            raise ValueError(
                "Must have some FX information to price FXExchange, either `fx` or "
                "`solver` containing an FX object."
            )
        if not isinstance(fx_, (FXRates, FXForwards)):
            # force base_ leg1 currency to be converted consistent.
            leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
            leg2_npv = self.leg2.npv(curves[2], curves[3], 1.0, base_, local)
            warnings.warn(
                "When valuing multi-currency derivatives it not best practice to "
                "supply `fx` as numeric.\nYour input:\n"
                f"`npv(solver={'None' if solver is NoInput.blank else '<Solver>'}, fx={fx}, base='{base if base is not NoInput.blank else 'None'}')\n"
                "has been implicitly converted into the following by this operation:\n"
                f"`npv(solver={'None' if solver is NoInput.blank else '<Solver>'}, "
                f"fx=FXRates({{'{self.leg2.currency}{self.leg1.currency}: {fx}}}), base='{self.leg2.currency}')\n.",
                UserWarning,
            )
        else:
            leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
            leg2_npv = self.leg2.npv(curves[2], curves[3], fx_, base_, local)

        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the cashflows of the *FXExchange* by aggregating legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.cashflows>`
        """
        self._set_pricing_mid(curves, solver, fx)
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, NoInput(0)
        )
        seq = [
            self.leg1.cashflows(curves[0], curves[1], fx_, base_),
            self.leg2.cashflows(curves[2], curves[3], fx_, base_),
        ]
        _ = DataFrame.from_records(seq)
        _.index = MultiIndex.from_tuples([("leg1", 0), ("leg2", 0)])
        return _

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the mid-market rate of the instrument.

        For arguments see :meth:`BaseMixin.rate<rateslib.instruments.BaseMixin.rate>`
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if isinstance(fx_, (FXRates, FXForwards)):
            imm_fx = fx_.rate(self.pair)
        else:
            imm_fx = fx_

        _ = forward_fx(self.settlement, curves[1], curves[3], imm_fx)
        return _

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


# Securities


class BondMixin:
    def _set_base_index_if_none(self, curve: IndexCurve):
        if self._index_base_mixin and self.index_base is NoInput.blank:
            self.leg1.index_base = curve.index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )

    def ex_div(self, settlement: datetime):
        """
        Return a boolean whether the security is ex-div at the given settlement.

        Parameters
        ----------
        settlement : datetime
            The settlement date to test.

        Returns
        -------
        bool

        Notes
        -----
        By default uses the UK DMO convention of returning *False* if ``settlement``
        **is on or before** the ex-div date.

        Some ``calc_mode`` options return *True* if ``settlement`` **is on** the ex-div date.

        Ex-div dates are determined as measured by the number of ``ex_div`` business days prior
        to the unadjusted coupon end date.

        With an ``ex_div`` of 1, a ``settlement`` that occurs on the coupon payment date will be
        classified as ex-dividend and not receive that coupon.

        With an ``ex_div`` of 0, a ``settlement`` that occurs on the coupon payment date will
        **not** be classified as ex-dividend and will receive that coupon (in the default
        calculation mode).
        """
        prev_a_idx = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        ex_div_date = add_tenor(
            self.leg1.schedule.uschedule[prev_a_idx + 1],
            f"{-self.kwargs['ex_div']}B",
            NoInput(0),  # modifier not required for business day tenor
            self.leg1.schedule.calendar,
        )
        if self.calc_mode in []:  # currently no identified calc_modes
            return True if settlement >= ex_div_date else False  # pragma: no cover
        else:
            return True if settlement > ex_div_date else False

    def _acc_index(self, settlement: datetime):
        """
        Get the coupon period index for that which the settlement date fall within.
        Uses unadjusted dates.
        """
        _ = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        return _

    def _accrued(self, settlement: datetime, calc_mode: Union[str, NoInput]):
        acc_idx = self._acc_index(settlement)
        frac = self._accrued_frac(settlement, calc_mode, acc_idx)
        if self.ex_div(settlement):
            frac = frac - 1  # accrued is negative in ex-div period
        _ = getattr(self.leg1.periods[acc_idx], self._ytm_attribute)
        return frac * _ / -self.leg1.notional * 100

    def _accrued_frac(self, settlement: datetime, calc_mode: Union[str, NoInput], acc_idx: int):
        """
        Return the accrual fraction of period between last coupon and settlement and
        coupon period left index.

        Branches to a calculation based on the bond `calc_mode`.
        """
        acc_frac_funcs = {
            NoInput(0): self._acc_lin_days,
            "ukg": self._acc_lin_days,
            "uktb": self._acc_lin_days,
            "ust": self._acc_lin_days_long_split,
            "ust_31bii": self._acc_lin_days_long_split,
            "ustb": self._acc_lin_days,
            "sgb": self._acc_30e360,
            "sgbb": self._acc_lin_days,
            "cadgb": self._acc_act365_1y_stub,
            "cadgb-ytm": self._acc_lin_days,
        }
        try:
            return acc_frac_funcs[calc_mode](settlement, acc_idx)
        except KeyError:
            raise ValueError(f"Cannot calculate for `calc_mode`: {calc_mode}")

    def _acc_lin_days(self, settlement: datetime, acc_idx: int, *args):
        """
        Method uses a linear proportion of days between payments to allocate accrued interest.
        Measures between unadjusted coupon dates.
        This is a general method, used for example by [UK Gilts].
        """
        r = settlement - self.leg1.schedule.uschedule[acc_idx]
        s = self.leg1.schedule.uschedule[acc_idx + 1] - self.leg1.schedule.uschedule[acc_idx]
        return r / s

    def _acc_lin_days_long_split(self, settlement: datetime, acc_idx: int, *args):
        """
        For long stub periods this splits the accrued interest into two components.
        Otherwise, returns the regular linear proportion.
        [Designed primarily for US Treasuries]
        """
        if self.leg1.periods[acc_idx].stub:
            fm = defaults.frequency_months[self.leg1.schedule.frequency]
            f = 12 / fm
            if self.leg1.periods[acc_idx].dcf * f > 1:
                # long stub
                quasi_coupon = add_tenor(
                    self.leg1.schedule.uschedule[acc_idx + 1],
                    f"-{fm}M",
                    "NONE",
                    NoInput(0),
                    self.leg1.schedule.roll,
                )
                quasi_start = add_tenor(
                    quasi_coupon,
                    f"-{fm}M",
                    "NONE",
                    NoInput(0),
                    self.leg1.schedule.roll,
                )
                if settlement <= quasi_coupon:
                    # then first part of long stub
                    r = quasi_coupon - settlement
                    s = quasi_coupon - quasi_start
                    r_ = quasi_coupon - self.leg1.schedule.uschedule[acc_idx]
                    _ = (r_ - r) / s
                    return _ / (self.leg1.periods[acc_idx].dcf * f)
                else:
                    # then second part of long stub
                    r = self.leg1.schedule.uschedule[acc_idx + 1] - settlement
                    s = self.leg1.schedule.uschedule[acc_idx + 1] - quasi_coupon
                    r_ = quasi_coupon - self.leg1.schedule.uschedule[acc_idx]
                    s_ = quasi_coupon - quasi_start
                    _ = r_ / s_ + (s - r) / s
                    return _ / (self.leg1.periods[acc_idx].dcf * f)

        return self._acc_lin_days(settlement, acc_idx, *args)

    def _acc_30e360(self, settlement: datetime, acc_idx: int, *args):
        """
        Ignoring the convention on the leg uses "30E360" to determine the accrual fraction.
        Measures between unadjusted date and settlement.
        [Designed primarily for Swedish Government Bonds]
        """
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        _ = dcf(settlement, self.leg1.schedule.uschedule[acc_idx + 1], "30e360") * f
        _ = 1 - _
        return _

    def _acc_act365_1y_stub(self, settlement: datetime, acc_idx: int, *args):
        """
        Ignoring the convention on the leg uses "Act365f" to determine the accrual fraction.
        Measures between unadjusted date and settlement.
        Special adjustment if number of days is greater than 365.
        If the period is a stub reverts to a straight line interpolation
        [this is primarily designed for Canadian Government Bonds]
        """
        if self.leg1.periods[acc_idx].stub:
            return self._acc_lin_days(settlement, acc_idx)
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        r = settlement - self.leg1.schedule.uschedule[acc_idx]
        s = self.leg1.schedule.uschedule[acc_idx + 1] - self.leg1.schedule.uschedule[acc_idx]
        if r == s:
            _ = 1.0  # then settlement falls on the coupon date
        elif r.days > 365.0 / f:
            _ = 1.0 - ((s - r).days * f) / 365.0  # counts remaining days
        else:
            _ = f * r.days / 365.0
        return _

    def _generic_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        dirty: bool,
        f1: callable,
        f2: callable,
        f3: callable,
        accrual_calc_mode: Union[str, NoInput],
    ):
        """
        Refer to supplementary material.
        """
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        acc_idx = self._acc_index(settlement)

        v2 = f2(ytm, f, settlement, acc_idx)
        v1 = f1(ytm, f, settlement, acc_idx, v2, accrual_calc_mode)
        v3 = f3(ytm, f, settlement, self.leg1.schedule.n_periods - 1, v2)

        # Sum up the coupon cashflows discounted by the calculated factors
        d = 0
        for i, p_idx in enumerate(range(acc_idx, self.leg1.schedule.n_periods)):
            if i == 0 and self.ex_div(settlement):
                # no coupon cashflow is receiveable so no addition to the sum
                continue
            elif i == 0 and p_idx == (self.leg1.schedule.n_periods - 1):
                # the last period is the first period so discounting handled only by v1 at end
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute)
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # this is last period, but it is not the first (i>0). Tag on v3 at end.
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v2 ** (i - 1) * v3
            else:
                # this is not the first and not the last period. Discount only with v1 and v2.
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v2**i

        # Add the redemption payment discounted by relevant factors
        if i == 0:  # only looped 1 period, no need for v2 and v3
            d += getattr(self.leg1.periods[-1], self._ytm_attribute)
        elif i == 1:  # only looped 2 periods, no need for v2
            d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v3
        else:  # looped more than 2 periods, regular formula applied
            d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v2 ** (i - 1) * v3

        # discount all by the first period factor and scaled to price
        p = v1 * d / -self.leg1.notional * 100

        return p if dirty else p - self._accrued(settlement, accrual_calc_mode)

    def _v2_(self, ytm: DualTypes, f: int, settlement: datetime, acc_idx: int, *args):
        """
        The default method for a single regular period discounted in the regular portion of bond.
        Implies compounding at the same frequency as the coupons.
        """
        return 1 / (1 + ytm / (100 * f))

    def _v2_1y_simple(self, ytm: DualTypes, f: int, settlement: datetime, acc_idx: int, *args):
        """
        The default method for a single regular period discounted in the regular portion of bond.
        Implies compounding at the same frequency as the coupons.
        """
        return 1 / (1 + ytm / (100 * f))

    def _v1_comp(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual_calc_mode: Union[str, NoInput],
        *args,
    ):
        """
        The initial period uses a compounding approach where the power is determined by the
        accrual fraction under the specified accrual mode.
        """
        acc_frac = self._accrued_frac(settlement, accrual_calc_mode, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            fd0 = 1 - acc_frac
        return v**fd0

    def _v1_simple(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual_calc_mode: Union[str, NoInput],
        *args,
    ):
        """
        The initial period discounts by a simple interest amount
        """
        acc_frac = self._accrued_frac(settlement, accrual_calc_mode, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            fd0 = 1 - acc_frac

        if fd0 > 1.0:
            v_ = v * 1 / (1 + (fd0 - 1) * ytm / (100 * f))
        else:
            v_ = 1 / (1 + fd0 * ytm / (100 * f))

        return v_

    def _v3_dcf_comp(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        *args,
    ):
        """
        Final period uses a compounding approach where the power is determined by the DCF of that
        period under the bond's specified convention.
        """
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f
        else:
            fd0 = 1
        return v**fd0

    def _v3_30e360_u_simple(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        *args,
    ):
        """
        The final period is discounted by a simple interest method under a 30E360 convention.
        """
        d_ = dcf(self.leg1.periods[acc_idx].start, self.leg1.periods[acc_idx].end, "30E360")
        return 1 / (1 + d_ * ytm / 100)  # simple interest

    def _price_from_ytm(
        self,
        ytm: float,
        settlement: datetime,
        calc_mode: Union[str, NoInput],
        dirty: bool = False,
    ):
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        # fmt: off
        price_from_ytm_funcs = {
            NoInput(0): partial(self._generic_ytm, f1=self._v1_comp, f2=self._v2_, f3=self._v3_dcf_comp, accrual_calc_mode=NoInput(0)),
            "ukg": partial(self._generic_ytm, f1=self._v1_comp, f2=self._v2_, f3=self._v3_dcf_comp, accrual_calc_mode="ukg"),
            "ust": partial(self._generic_ytm, f1=self._v1_comp, f2=self._v2_, f3=self._v3_dcf_comp, accrual_calc_mode="ust"),
            "ust_31bii": partial(self._generic_ytm, f1=self._v1_simple, f2=self._v2_, f3=self._v3_dcf_comp, accrual_calc_mode="ust"),
            "sgb": partial(self._generic_ytm, f1=self._v1_comp, f2=self._v2_, f3=self._v3_30e360_u_simple, accrual_calc_mode="sgb"),
            "cadgb": partial(self._generic_ytm, f1=self._v1_comp, f2=self._v2_, f3=self._v3_dcf_comp, accrual_calc_mode="cadgb-ytm"),
        }
        # fmt: on
        try:
            return price_from_ytm_funcs[calc_mode](ytm, settlement, dirty)
        except KeyError:
            raise ValueError(f"Cannot calculate with `calc_mode`: {calc_mode}")

    def fwd_from_repo(
        self,
        price: Union[float, Dual, Dual2],
        settlement: datetime,
        forward_settlement: datetime,
        repo_rate: Union[float, Dual, Dual2],
        convention: Union[str, NoInput] = NoInput(0),
        dirty: bool = False,
        method: str = "proceeds"
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
        method : str in {"proceeds", "compounded"}, optional
            The method for determining the forward price.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention = defaults.convention if convention is NoInput.blank else convention
        dcf_ = dcf(settlement, forward_settlement, convention)
        if not dirty:
            d_price = price + self.accrued(settlement)
        else:
            d_price = price
        if self.leg1.amortization != 0:
            raise NotImplementedError(
                "method for forward price not available with amortization"
            )  # pragma: no cover
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
            if method.lower() == "proceeds":
                dcf_ = dcf(self.leg1.periods[p_idx].payment, forward_settlement, convention)
                accrued_coup = self.leg1.periods[p_idx].cashflow * (1 + dcf_ * repo_rate / 100)
                total_rtn -= accrued_coup
            elif method.lower() == "compounded":
                r_bar, d, _ = average_rate(settlement, forward_settlement, convention, repo_rate)
                n = (forward_settlement - self.leg1.periods[p_idx].payment).days
                accrued_coup = self.leg1.periods[p_idx].cashflow * (1 + d * r_bar / 100) ** n
                total_rtn -= accrued_coup
            else:
                raise ValueError("`method` must be in {'proceeds', 'compounded'}.")

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
        convention: Union[str, NoInput] = NoInput(0),
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
        convention = defaults.convention if convention is NoInput.blank else convention
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
            denominator -= 100 * dcf_ * self.leg1.periods[p_idx].cashflow / -self.leg1.notional

        return numerator / denominator * 100

    def _npv_local(
        self,
        curve: Union[Curve, LineCurve],
        disc_curve: Curve,
        fx: Union[float, FXRates, FXForwards, NoInput],
        base: Union[str, NoInput],
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

        if projection is NoInput.blank:
            return npv
        else:
            return npv / disc_curve[projection]

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        settlement = add_tenor(
            curves[1].node_dates[0],
            f"{self.kwargs['settle']}B",
            None,
            self.leg1.schedule.calendar,
        )
        base_ = self.leg1.currency if local else base
        npv = self._npv_local(curves[0], curves[1], fx_, base_, settlement, NoInput(0))
        if local:
            return {self.leg1.currency: npv}
        else:
            return npv

    def analytic_delta(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the analytic delta of the security via summing all periods.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        settlement = add_tenor(
            disc_curve_.node_dates[0],
            f"{self.kwargs['settle']}B",
            None,
            self.leg1.schedule.calendar,
        )
        a_delta = self.leg1.analytic_delta(curve, disc_curve_, fx, base)
        if self.ex_div(settlement):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement,
            )
            a_delta -= self.leg1.periods[current_period].analytic_delta(
                curve, disc_curve_, fx, base
            )
        return a_delta

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        settlement: Union[datetime, NoInput] = NoInput(0),
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        self._set_base_index_if_none(curves[0])

        if settlement is NoInput.blank and curves[1] is NoInput.blank:
            settlement = self.leg1.schedule.effective
        elif settlement is NoInput.blank:
            settlement = add_tenor(
                curves[1].node_dates[0],
                f"{self.kwargs['settle']}B",
                None,
                self.leg1.schedule.calendar,
            )
        cashflows = self.leg1.cashflows(curves[0], curves[1], fx_, base_)
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

    def oaspread(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        price: DualTypes = NoInput(0),
        dirty: bool = False,
    ):
        """
        The option adjusted spread added to the discounting *Curve* to value the security
        at ``price``.

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
        price : float, Dual, Dual2
            The price of the bond to match.
        dirty : bool
            Whether the price is given clean or dirty.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        ad_ = curves[1].ad
        metric = "dirty_price" if dirty else "clean_price"

        curves[1]._set_ad_order(1)
        disc_curve = curves[1].shift(Dual(0, ["z_spread"], []), composite=False)
        npv_price = self.rate(curves=[curves[0], disc_curve], metric=metric)

        # find a first order approximation of z
        b = gradient(npv_price, ["z_spread"], 1)[0]
        c = float(npv_price) - float(price)
        z_hat = -c / b

        # shift the curve to the first order approximation and fine tune with 2nd order approxim.
        curves[1]._set_ad_order(2)
        disc_curve = curves[1].shift(Dual2(z_hat, ["z_spread"], [], []), composite=False)
        npv_price = self.rate(curves=[curves[0], disc_curve], metric=metric)
        a, b = (
            0.5 * gradient(npv_price, ["z_spread"], 2)[0][0],
            gradient(npv_price, ["z_spread"], 1)[0],
        )
        z_hat2 = _quadratic_equation(a, b, float(npv_price) - float(price))

        # perform one final approximation albeit the additional price calculation slows calc time
        curves[1]._set_ad_order(0)
        disc_curve = curves[1].shift(z_hat+z_hat2, composite=False)
        npv_price = self.rate(curves=[curves[0], disc_curve], metric=metric)
        b = b + 2 * a * z_hat2  # forecast the new gradient
        c = float(npv_price) - float(price)
        z_hat3 = -c / b

        z = z_hat + z_hat2 + z_hat3
        curves[1]._set_ad_order(ad_)
        return z


class FixedRateBond(Sensitivities, BondMixin, BaseMixin):
    # TODO (mid) ensure calculations work for amortizing bonds.
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
        The frequency of the schedule. "Z" is **not** permitted. For zero-coupon-bonds use a
        ``fixed_rate`` of zero and set the frequency according to the yield-to-maturity
        convention required.
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
    calc_mode : str
        A calculation mode for dealing with bonds under different conventions. See notes.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    Attributes
    ----------
    ex_div_days : int
    settle : int
    curves : str, list, CurveType
    leg1 : FixedLeg

    Notes
    -----

    **Calculation Modes**

    The ``calc_mode`` parameter allows the calculation for yield-to-maturity and accrued interest
    to branch depending upon the particular convention of different bonds.

    The following modes are currently available with a brief description of its particular
    action:

    - "ukg": UK Gilt convention. Accrued is linearly proportioned, as are stub periods. Stub yields
      are compounded.
    - "ust": US Treasury street convention. Same as "ukg" except long stub periods have linear
      proportioning only in the segregated short stub part.
    - "ust_31bii": US Treasury convention that reprices examples in federal documents: Section
      31-B-ii). Otherwise referred to as the 'Treasury' method.
    - "sgb": Swedish government bond convention. Accrued ignores the convention and calculates
      using 30e360, also for back stubs.
    - "cadgb" Canadian government bond convention. Accrued is calculated using an ACT365F
      convention. Yield calculations are still derived with linearly proportioned compounded
      coupons.

    More details available in supplementary materials. The table below
    outlines the *rateslib* price result relative to the calculation examples provided
    from official sources.

    .. ipython:: python
       :suppress:

       sgb = FixedRateBond(
           effective=dt(2022, 3, 30), termination=dt(2039, 3, 30),
           frequency="A", convention="ActActICMA", calc_mode="SGB",
           fixed_rate=3.5, calendar="stk"
       )
       s1c = sgb.price(ytm=2.261, settlement=dt(2023, 3, 15), dirty=False)
       s1d = sgb.price(ytm=2.261, settlement=dt(2023, 3, 15), dirty=True)

       uk1 = FixedRateBond(
           effective=dt(1995, 1, 1), termination=dt(2015, 12, 7),
           frequency="S", convention="ActActICMA", calc_mode="UKG",
           fixed_rate=8.0, calendar="ldn", ex_div=7,
       )
       uk11c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 24), dirty=False)
       uk11d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 24), dirty=True)
       uk12c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 26), dirty=False)
       uk12d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 26), dirty=True)
       uk13c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=False)
       uk13d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=True)
       uk14c = uk1.price(ytm=4.445, settlement=dt(1999, 6, 7), dirty=False)
       uk14d = uk1.price(ytm=4.445, settlement=dt(1999, 6, 7), dirty=True)

       uk2 = FixedRateBond(
           effective=dt(1998, 11, 26), termination=dt(2004, 11, 26),
           frequency="S", convention="ActActICMA", calc_mode="UKG",
           fixed_rate=6.75, calendar="ldn", ex_div=7,
       )
       uk21c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 10), dirty=False)
       uk21d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 10), dirty=True)
       uk22c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 17), dirty=False)
       uk22d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 17), dirty=True)
       uk23c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 18), dirty=False)
       uk23d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 18), dirty=True)
       uk24c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 26), dirty=False)
       uk24d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 26), dirty=True)

       usA = FixedRateBond(
           effective=dt(1990, 5, 15), termination=dt(2020, 5, 15),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=8.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usAc = usA.price(ytm=8.84, settlement=dt(1990, 5, 15), dirty=False)
       usAd = usA.price(ytm=8.84, settlement=dt(1990, 5, 15), dirty=True)

       usB = FixedRateBond(
           effective=dt(1990, 4, 2), termination=dt(1992, 3, 31),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=8.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usBc = usB.price(ytm=8.59, settlement=dt(1990, 4, 2), dirty=False)
       usBd = usB.price(ytm=8.59, settlement=dt(1990, 4, 2), dirty=True)

       usC = FixedRateBond(
           effective=dt(1990, 3, 1), termination=dt(1995, 5, 15),
           front_stub=dt(1990, 11, 15),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=8.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usCc = usC.price(ytm=8.53, settlement=dt(1990, 3, 1), dirty=False)
       usCd = usC.price(ytm=8.53, settlement=dt(1990, 3, 1), dirty=True)

       usD = FixedRateBond(
           effective=dt(1985, 11, 15), termination=dt(1995, 11, 15),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=9.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usDc = usD.price(ytm=9.54, settlement=dt(1985, 11, 29), dirty=False)
       usDd = usD.price(ytm=9.54, settlement=dt(1985, 11, 29), dirty=True)

       usE = FixedRateBond(
           effective=dt(1985, 7, 2), termination=dt(2005, 8, 15),
           front_stub=dt(1986, 2, 15),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=10.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usEc = usE.price(ytm=10.47, settlement=dt(1985, 11, 4), dirty=False)
       usEd = usE.price(ytm=10.47, settlement=dt(1985, 11, 4), dirty=True)

       usF = FixedRateBond(
           effective=dt(1983, 5, 16), termination=dt(1991, 5, 15), roll=15,
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=10.50, calendar="nyc", ex_div=1, modifier="none",
       )

       usFc = usF.price(ytm=10.53, settlement=dt(1983, 8, 15), dirty=False)
       usFd = usF.price(ytm=10.53, settlement=dt(1983, 8, 15), dirty=True)

       usG = FixedRateBond(
           effective=dt(1988, 10, 15), termination=dt(1994, 12, 15),
           front_stub=dt(1989, 6, 15),
           frequency="S", convention="ActActICMA", calc_mode="UST_31bii",
           fixed_rate=9.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usGc = usG.price(ytm=9.79, settlement=dt(1988, 11, 15), dirty=False)
       usGd = usG.price(ytm=9.79, settlement=dt(1988, 11, 15), dirty=True)

       data = DataFrame(data=[
               ["Riksgalden Website", "Nominal Bond", 116.514000, 119.868393, "sgb", s1c, s1d],
               ["UK DMO Website", "Ex 1, Scen 1", None, 145.012268, "ukg", uk11c, uk11d],
               ["UK DMO Website", "Ex 1, Scen 2", None, 145.047301, "ukg", uk12c, uk12d],
               ["UK DMO Website", "Ex 1, Scen 3", None, 141.070132, "ukg", uk13c, uk13d],
               ["UK DMO Website", "Ex 1, Scen 4", None, 141.257676, "ukg", uk14c, uk14d],
               ["UK DMO Website", "Ex 2, Scen 1", None, 113.315543, "ukg", uk21c, uk21d],
               ["UK DMO Website", "Ex 2, Scen 2", None, 113.415969, "ukg", uk22c, uk22d],
               ["UK DMO Website", "Ex 2, Scen 3", None, 110.058738, "ukg", uk23c, uk23d],
               ["UK DMO Website", "Ex 2, Scen 4", None, 110.170218, "ukg", uk24c, uk24d],
               ["Title-31 Subtitle-B II", "Ex A (reg)",99.057893, 99.057893, "ust_31bii", usAc, usAd],
               ["Title-31 Subtitle-B II", "Ex B (stub)", 99.838183, 99.838183, "ust_31bii", usBc, usBd],
               ["Title-31 Subtitle-B II", "Ex C (stub)", 99.805118, 99.805118, "ust_31bii", usCc, usCd],
               ["Title-31 Subtitle-B II", "Ex D (reg)", 99.730918, 100.098321, "ust_31bii", usDc, usDd],
               ["Title-31 Subtitle-B II", "Ex E (stub)", 102.214586, 105.887384, "ust_31bii", usEc, usEd],
               ["Title-31 Subtitle-B II", "Ex F (stub)", 99.777074, 102.373541, "ust_31bii", usFc, usFd],
               ["Title-31 Subtitle-B II", "Ex G (stub)", 99.738045, 100.563865, "ust_31bii", usGc, usGd],
           ],
           columns=["Source", "Example", "Expected clean", "Expected dirty", "Calc mode", "Rateslib clean", "Rateslib dirty"],
       )

    .. ipython:: python

       from pandas import option_context
       with option_context("display.float_format", lambda x: '%.6f' % x):
           print(data)

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
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        frequency: Union[int, NoInput] = NoInput(0),
        stub: Union[str, NoInput] = NoInput(0),
        front_stub: Union[datetime, NoInput] = NoInput(0),
        back_stub: Union[datetime, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, None, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        amortization: Union[float, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        fixed_rate: Union[float, NoInput] = NoInput(0),
        ex_div: Union[int, NoInput] = NoInput(0),
        settle: Union[int, NoInput] = NoInput(0),
        calc_mode: Union[str, NoInput] = NoInput(0),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ):
        self.kwargs = dict(
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
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode,
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if self.kwargs["frequency"] is NoInput.blank:
            raise ValueError("`frequency` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = self.kwargs["calc_mode"].lower()
        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"]))

        if self.leg1.amortization != 0:
            # Note if amortization is added to FixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FixedRateBond must be zero.")

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

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
        return self._accrued(settlement, self.calc_mode)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric: str = "clean_price",
        forward_settlement: Union[datetime, NoInput] = NoInput(0),
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
            "dirty_price", "ytm"}
        forward_settlement : datetime, optional
            The forward settlement date. If not given the settlement date is inferred from the
            discount *Curve* and the ``settle`` attribute.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "ytm"]:
            if forward_settlement is NoInput.blank:
                settlement = add_tenor(
                    curves[1].node_dates[0],
                    f"{self.kwargs['settle']}B",
                    "none",
                    self.leg1.schedule.calendar,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], fx_, base_, settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "ytm":
                return self.ytm(dirty_price, settlement, True)

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'ytm'}.")

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
            return self._price_from_ytm(y, settlement, self.calc_mode, dirty) - float(price)

        # x = brentq(root, -99, 10000)  # remove dependence to scipy.optimize.brentq
        # x, iters = _brents(root, -99, 10000)  # use own local brents code
        x = _ytm_quadratic_converger2(root, -3.0, 2.0, 12.0)  # use special quad interp

        if isinstance(price, Dual):
            # use the inverse function theorem to express x as a Dual
            p = self._price_from_ytm(Dual(x, ["y"], []), settlement, self.calc_mode, dirty)
            return Dual(x, price.vars, 1 / gradient(p, ["y"])[0] * price.dual)
        elif isinstance(price, Dual2):
            # use the IFT in 2nd order to express x as a Dual2
            p = self._price_from_ytm(Dual2(x, ["y"], [], []), settlement, self.calc_mode, dirty)
            dydP = 1 / gradient(p, ["y"])[0]
            d2ydP2 = -gradient(p, ["y"], order=2)[0][0] * gradient(p, ["y"])[0] ** -3
            dual = dydP * price.dual
            dual2 = 0.5 * (
                dydP * gradient(price, price.vars, order=2)
                + d2ydP2 * np.matmul(price.dual[:, None], price.dual[None, :])
            )

            return Dual2(x, price.vars, dual.tolist(), list(dual2.flat))
        else:
            return x

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

             risk = - \\frac{\\partial P }{\\partial y}

        - *"modified"*: the modified duration which is *risk* divided by price.

          .. math::

             mduration = \\frac{risk}{P} = - \\frac{1}{P} \\frac{\\partial P }{\\partial y}

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
            _ = -gradient(self.price(Dual(float(ytm), ["y"], []), settlement), ["y"])[0]
        elif metric == "modified":
            price = -self.price(Dual(float(ytm), ["y"], []), settlement, dirty=True)
            _ = -gradient(price, ["y"])[0] / float(price) * 100
        elif metric == "duration":
            price = -self.price(Dual(float(ytm), ["y"], []), settlement, dirty=True)
            f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
            v = 1 + float(ytm) / (100 * f)
            _ = -gradient(price, ["y"])[0] / float(price) * v * 100
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
        _ = self.price(Dual2(float(ytm), ["y"], [], []), settlement)
        return gradient(_, ["y"], 2)[0][0]

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
        return self._price_from_ytm(ytm, settlement, self.calc_mode, dirty)

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


class IndexFixedRateBond(FixedRateBond):
    # TODO (mid) ensure calculations work for amortizing bonds.
    """
    Create an indexed fixed rate bond security.

    Parameters
    ----------
    args : tuple
        Required positional args for :class:`~rateslib.instruments.FixedRateBond`.
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
        Required keyword args for :class:`~rateslib.instruments.FixedRateBond`.

    Examples
    --------
    See :class:`~rateslib.instruments.FixedRateBond` for similar.
    """

    _fixed_rate_mixin = True
    _ytm_attribute = "real_cashflow"  # index linked bonds use real cashflows
    _index_base_mixin = True

    def __init__(
        self,
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        frequency: Union[int, NoInput] = NoInput(0),
        stub: Union[str, NoInput] = NoInput(0),
        front_stub: Union[datetime, NoInput] = NoInput(0),
        back_stub: Union[datetime, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, None, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        amortization: Union[float, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        fixed_rate: Union[float, NoInput] = NoInput(0),
        index_base: Union[float, Series, NoInput] = NoInput(0),
        index_fixings: Union[float, Series, NoInput] = NoInput(0),
        index_method: Union[str, NoInput] = NoInput(0),
        index_lag: Union[int, NoInput] = NoInput(0),
        ex_div: Union[int, NoInput] = NoInput(0),
        settle: Union[int, NoInput] = NoInput(0),
        calc_mode: Union[str, NoInput] = NoInput(0),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ):
        self.kwargs = dict(
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
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
            index_base=index_base,
            index_method=index_method,
            index_lag=index_lag,
            index_fixings=index_fixings,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode,
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
            index_method=defaults.index_method,
            index_lag=defaults.index_lag,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if self.kwargs["frequency"] is NoInput.blank:
            raise ValueError("`frequency` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = self.kwargs["calc_mode"].lower()
        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self._index_base = index_base

        self.leg1 = IndexFixedLeg(
            **_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"])
        )
        if self.leg1.amortization != 0:
            # Note if amortization is added to IndexFixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for IndexFixedRateBond must be zero.")

    def index_ratio(self, settlement: datetime, curve: Union[IndexCurve, NoInput]):
        if self.leg1.index_fixings is not NoInput.blank and not isinstance(
            self.leg1.index_fixings, Series
        ):
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
            i_curve=curve,
        )
        return index_val / index_base

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric: str = "clean_price",
        forward_settlement: Union[datetime, NoInput] = NoInput(0),
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
            "dirty_price", "ytm", "index_clean_price", "index_dirty_price"}
        forward_settlement : datetime, optional
            The forward settlement date. If not given uses the discount *Curve* and the ``settle``
            attribute of the bond.

        Returns
        -------
        float, Dual, Dual2
        """

        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        metric = metric.lower()
        if metric in [
            "clean_price",
            "dirty_price",
            "index_clean_price",
            "ytm",
            "index_dirty_price",
        ]:
            if forward_settlement is NoInput.blank:
                settlement = add_tenor(
                    curves[1].node_dates[0],
                    f"{self.kwargs['settle']}B",
                    None,
                    self.leg1.schedule.calendar,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], fx_, base_, settlement, settlement)
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

        raise ValueError(
            "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
            "'index_dirty_price', 'index_clean_price'}."
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
        calc_mode : str in {"ukg", "ust", "sgb"}
        A calculation mode for dealing with bonds that are in short stub or accrual
        periods. All modes give the same value for YTM at issue date for regular
        bonds but differ slightly for bonds with stubs or with accrued.
    calc_mode : str in {"ukg", "ust", "sgb"}
        A calculation mode for dealing with bonds that are in short stub or accrual
        periods. All modes give the same value for YTM at issue date for regular
        bonds but differ slightly for bonds with stubs or with accrued.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

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
           calendar="nyc",
           modifier="NONE",
           currency="usd",
           convention="Act360",
           settle=1,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bill
           curves="bill_curve",
           calc_mode="ustb",
       )
       bill.ex_div(dt(2004, 1, 22))
       bill.price(rate=0.80, settlement=dt(2004, 1, 22))
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
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        modifier: Union[str, None, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        settle: Union[str, NoInput] = NoInput(0),
        calc_mode: Union[str, NoInput] = NoInput(0),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ):
        super().__init__(
            effective=effective,
            termination=termination,
            frequency="z",
            stub=NoInput(0),
            front_stub=NoInput(0),
            back_stub=NoInput(0),
            roll=NoInput(0),
            eom=NoInput(0),
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=NoInput(0),
            convention=convention,
            fixed_rate=0,
            ex_div=0,
            settle=settle,
            curves=curves,
            calc_mode=calc_mode,
            spec=spec,
        )

    @property
    def dcf(self):
        # bills will typically have 1 period since they are configured with frequency "z".
        d = 0.0
        for i in range(self.leg1.schedule.n_periods):
            d += self.leg1.periods[i].dcf
        return d

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        settlement = add_tenor(
            curves[1].node_dates[0],
            f"{self.kwargs['settle']}B",
            None,
            self.leg1.schedule.calendar,
        )
        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves, solver, fx_, base_)
            * 100
            / (-self.leg1.notional * curves[1][settlement])
        )
        if metric in ["price", "clean_price", "dirty_price"]:
            return price
        elif metric == "discount_rate":
            return self.discount_rate(price, settlement)
        elif metric == "simple_rate":
            return self.simple_rate(price, settlement)
        elif metric == "ytm":
            return self.ytm(price, settlement, False)
        raise ValueError("`metric` must be in {'price', 'discount_rate', 'ytm', 'simple_rate'}")

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
        dcf = (1 - self._accrued_frac(settlement, self.calc_mode, 0)) * self.dcf
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
        dcf = (1 - self._accrued_frac(settlement, self.calc_mode, 0)) * self.dcf
        rate = ((1 - price / 100) / dcf) * 100
        return rate

    def price(
        self,
        rate: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        calc_mode: Union[str, NoInput] = NoInput(0),
    ) -> DualTypes:
        """
        Return the price of the bill given the ``discount_rate``.

        Parameters
        ----------
        rate : float
            The rate used by the pricing formula.
        settlement : datetime
            The settlement date.
        dirty : bool, not required
            Discount securities have no coupon, the concept of clean or dirty is not
            relevant. Argument is included for signature consistency with
            :meth:`FixedRateBond.price<rateslib.instruments.FixedRateBond.price>`.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Returns
        -------
        float, Dual, Dual2
        """
        price_funcs = {
            NoInput(0): self._price_discount,
            "sgbb": self._price_simple,
            "uktb": self._price_simple,
            "ustb": self._price_discount,
        }
        if not isinstance(calc_mode, str):
            calc_mode = self.calc_mode
        return price_funcs[calc_mode](rate, settlement)

    def _price_discount(self, rate: DualTypes, settlement: datetime):
        dcf = (1 - self._accrued_frac(settlement, self.calc_mode, 0)) * self.dcf
        return 100 - rate * dcf

    def _price_simple(self, rate: DualTypes, settlement: datetime):
        dcf = (1 - self._accrued_frac(settlement, self.calc_mode, 0)) * self.dcf
        return 100 / (1 + rate * dcf / 100)

    def ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        calc_mode: Union[str, NoInput] = NoInput(0),
    ):
        """
        Calculate the yield-to-maturity on an equivalent bond with a coupon of 0%.

        Parameters
        ----------
        price: float, Dual, Dual2
            The price of the *Bill*.
        settlement: datetime
            The settlement date of the *Bill*.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Notes
        -----
        Maps the following *Bill* ``calc_mode`` to the following *Bond* specifications:

        - *NoInput* -> "ust"
        - *"ustb"* -> "ust"
        - *"uktb"* -> "ukt"
        - *"sgbb"* -> "sgb"

        This method calculates by constructing a :class:`~rateslib.instruments.FixedRateBond`
        with a regular 0% coupon measured from the termination date of the bill.
        """
        spec_map = {
            NoInput(0): "usd_gb",
            "ustb": "usd_gb",
            "uktb": "gbp_gb",
            "sgbb": "sek_gb",
        }
        if isinstance(calc_mode, str):
            calc_mode = calc_mode.lower()
        else:
            calc_mode = self.calc_mode
        spec_kwargs = defaults.spec[spec_map[calc_mode]]
        frequency_months = defaults.frequency_months[spec_kwargs["frequency"].upper()]
        quasi_start = self.leg1.schedule.termination
        while quasi_start > settlement:
            quasi_start = add_tenor(
                quasi_start, f"-{frequency_months}M", "NONE", NoInput(0), NoInput(0)
            )
        equiv_bond = FixedRateBond(
            effective=quasi_start,
            termination=self.leg1.schedule.termination,
            fixed_rate=0.0,
            spec=spec_map[calc_mode],
        )
        return equiv_bond.ytm(price, settlement)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatRateNote(Sensitivities, BondMixin, BaseMixin):
    """
    Create a floating rate note (FRN) security.

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
    leg1 : FloatLeg
    """

    _float_spread_mixin = True

    def __init__(
        self,
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        frequency: Union[int, NoInput] = NoInput(0),
        stub: Union[str, NoInput] = NoInput(0),
        front_stub: Union[datetime, NoInput] = NoInput(0),
        back_stub: Union[datetime, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, None, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        amortization: Union[float, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        float_spread: Union[float] = NoInput(0),
        fixings: Union[float, list, NoInput] = NoInput(0),
        fixing_method: Union[str, NoInput] = NoInput(0),
        method_param: Union[int, NoInput] = NoInput(0),
        spread_compound_method: Union[str, NoInput] = NoInput(0),
        ex_div: Union[int, NoInput] = NoInput(0),
        settle: Union[int, NoInput] = NoInput(0),
        calc_mode: Union[str, NoInput] = NoInput(0),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ):
        self.kwargs = dict(
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
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode,
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if self.kwargs["frequency"] is NoInput.blank:
            raise ValueError("`frequency` must be provided for Bond.")
        elif self.kwargs["frequency"].lower() == "z":
            raise ValueError("FloatRateNote `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = self.kwargs["calc_mode"].lower()
        self.curves = curves
        self.spec = spec

        self._float_spread = float_spread
        self.leg1 = FloatLeg(**_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"]))

        if "rfr" in self.leg1.fixing_method:
            if self.kwargs["ex_div"] > (self.leg1.method_param + 1):
                raise ValueError(
                    "For RFR FRNs `ex_div` must be less than or equal to (`method_param` + 1) "
                    "otherwise negative accrued payments cannot be explicitly "
                    "determined due to unknown fixings."
                )

        if self.leg1.amortization != 0:
            # Note if amortization is added to FloatRateNote must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FloatRateNote must be zero.")

    def _accrual_rate(self, pseudo_period, curve, method_param):
        """
        Take a period and try to forecast the rate which determines the accrual,
        either from known fixings, a curve or forward filling historical fixings.

        This method is required to handle the case where a curve is not provided and
        fixings are enough.
        """
        if pseudo_period.dcf < 1e-10:
            return 0.0  # there are no fixings in the period.

        if curve is not NoInput.blank:
            curve_ = curve
        else:
            # Test to see if any missing fixings are required:
            # The fixings calendar and convention are taken from Curve so the pseudo curve
            # can only get them from the instrument and assume that they align. Otherwise
            # it is best practice to supply a forecast curve when calculating accrued interest.
            pseudo_curve = Curve(
                {},
                calendar=pseudo_period.calendar,
                convention=pseudo_period.convention,
                modifier="F",
            )
            try:
                _ = pseudo_period.rate(pseudo_curve)
                return _
            except IndexError:
                # the pseudo_curve has no nodes so when it needs to calculate a rate it cannot
                # be indexed.
                # Try to revert back to using the last fixing as forward projection.
                try:
                    if isinstance(pseudo_period.fixings, Series):
                        last_fixing = pseudo_period.fixings.iloc[-1]
                    else:
                        last_fixing = pseudo_period.fixings[-1]
                    warnings.warn(
                        "A `Curve` was not supplied. Residual required fixings not yet "
                        "published are forecast from the last known fixing.",
                        UserWarning,
                    )
                    # For negative accr in ex-div we need to forecast unpublished rates.
                    # Build a curve which replicates the last fixing value from fixings.
                except TypeError:
                    # then rfr fixing cannot be fetched from attribute
                    if pseudo_period.dcf < 1e-10:
                        # then settlement is same as period.start so no rate necessary
                        # create a dummy curve
                        last_fixing = 0.0
                    else:
                        raise TypeError(
                            "`fixings` or `curve` are not available for RFR float period. If"
                            "supplying `fixings` must be a Series or list, "
                            f"got: {pseudo_period.fixings}"
                        )
                curve_ = LineCurve(
                    {
                        pseudo_period.start
                        - timedelta(
                            days=0 if isinstance(method_param, NoInput) else method_param
                        ): last_fixing,
                        pseudo_period.end: last_fixing,
                    },
                    convention=pseudo_period.convention,
                    calendar=pseudo_period.calendar,
                )

        # Otherwise rate to settle is determined fully by known fixings.
        _ = float(pseudo_period.rate(curve_))
        return _

    def accrued(
        self,
        settlement: datetime,
        curve: Union[Curve, NoInput] = NoInput(0),
    ):
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.
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
        - In the digital mode where the ``settlement`` is likely in the future we
          use a ``curve`` to forecast rates,

        Examples
        --------
        An RFR based FRN where the fixings are known up to the end of period.

        .. ipython:: python

           fixings = Series(2.0, index=date_range(dt(1999, 12, 1), dt(2000, 6, 2)))
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
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
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
               ex_div=7,
               fixings=fixings,
               fixing_method="ibor",
               method_param=2,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))
        """
        if self.leg1.fixing_method == "ibor":
            acc_idx = self._acc_index(settlement)
            frac = self._accrued_frac(settlement, self.calc_mode, acc_idx)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period

            if curve is not NoInput.blank:
                curve_ = curve
            else:
                curve_ = Curve(
                    {  # create a dummy curve. rate() will return the fixing
                        self.leg1.periods[acc_idx].start: 1.0,
                        self.leg1.periods[acc_idx].end: 1.0,
                    }
                )
            rate = self.leg1.periods[acc_idx].rate(curve_)

            cashflow = (
                -self.leg1.periods[acc_idx].notional * self.leg1.periods[acc_idx].dcf * rate / 100
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
                roll=self.leg1.schedule.roll,
                calendar=self.leg1.schedule.calendar,
            )
            rate_to_settle = self._accrual_rate(p, curve, self.leg1.method_param)
            accrued_to_settle = 100 * p.dcf * rate_to_settle / 100

            if self.ex_div(settlement):
                rate_to_end = self._accrual_rate(
                    self.leg1.periods[acc_idx], curve, self.leg1.method_param
                )
                accrued_to_end = 100 * self.leg1.periods[acc_idx].dcf * rate_to_end / 100
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric="clean_price",
        forward_settlement: Union[datetime, NoInput] = NoInput(0),
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
            "dirty_price", "spread"}
        forward_settlement : datetime, optional
            The forward settlement date. If not give uses the discount *Curve* and the bond's
            ``settle`` attribute.}.

        Returns
        -------
        float, Dual, Dual2

        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "spread"]:
            if forward_settlement is NoInput.blank:
                settlement = add_tenor(
                    curves[1].node_dates[0],
                    f"{self.kwargs['settle']}B",
                    None,
                    self.leg1.schedule.calendar,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], fx_, base_, settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement, curve=curves[0])
            elif metric == "spread":
                _ = self.leg1._spread(-(npv + self.leg1.notional), curves[0], curves[1])
                z = 0.0 if self.float_spread is NoInput.blank else self.float_spread
                return _ + z

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'spread'}.")

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


# Single currency derivatives


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
    calc_mode: str, optional
        The method to calculate conversion factors. See notes.

    Notes
    -----
    Conversion factors (CFs) ``calc_mode`` are:

    - *"ytm"* which calculates the CF as the clean price percent of par with the bond having a
      yield-to-maturity on the first delivery day in the delivery window.
    - *"ust_short"* which applies to CME 2y, 3y and 5y treasury futures. See
      :download:`CME Treasury Conversion Factors<_static/us-treasury-cfs.pdf>`.
    - *"ust_long"* which applies to CME 10y and 30y treasury futures.

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
        nominal: Union[float, NoInput] = NoInput(0),
        contracts: Union[int, NoInput] = NoInput(0),
        calendar: Union[str, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        calc_mode: Union[str, NoInput] = NoInput(0),
    ):
        self.currency = defaults.base_currency if currency is NoInput.blank else currency.lower()
        self.coupon = coupon
        if isinstance(delivery, datetime):
            self.delivery = (delivery, delivery)
        else:
            self.delivery = tuple(delivery)
        self.basket = tuple(basket)
        self.calendar = get_calendar(calendar)
        # self.last_trading = delivery[1] if last_trading is NoInput.blank else
        self.nominal = defaults.notional if nominal is NoInput.blank else nominal
        self.contracts = 1 if contracts is NoInput.blank else contracts
        self.calc_mode = (
            defaults.calc_mode_futures if calc_mode is NoInput.blank else calc_mode.lower()
        )
        self._cfs = NoInput(0)

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
        if self._cfs is NoInput.blank:
            self._cfs = self._conversion_factors()
        return self._cfs

    def _conversion_factors(self):
        if self.calc_mode == "ytm":
            return tuple(bond.price(self.coupon, self.delivery[0]) / 100 for bond in self.basket)
        elif self.calc_mode == "ust_short":
            return tuple(self._cfs_ust(bond, True) for bond in self.basket)
        elif self.calc_mode == "ust_long":
            return tuple(self._cfs_ust(bond, False) for bond in self.basket)
        else:
            raise ValueError("`calc_mode` must be in {'ytm', 'ust_short', 'ust_long'}")

    def _cfs_ust(self, bond: FixedRateBond, short: bool):
        # See CME pdf in doc Notes for formula.
        coupon = bond.fixed_rate / 100.0
        n, z = _get_years_and_months(self.delivery[0], bond.leg1.schedule.termination)
        if not short:
            mapping = {
                0: 0,
                1: 0,
                2: 0,
                3: 3,
                4: 3,
                5: 3,
                6: 6,
                7: 6,
                8: 6,
                9: 9,
                10: 9,
                11: 9,
            }
            z = mapping[z]  # round down number of months to quarters
        if z < 7:
            v = z
        elif short:
            v = z - 6
        else:
            v = 3
        a = 1 / 1.03 ** (v / 6.0)
        b = (coupon / 2) * (6 - v) / 6.0
        if z < 7:
            c = 1 / 1.03 ** (2 * n)
        else:
            c = 1 / 1.03 ** (2 * n + 1)
        d = (coupon / 0.06) * (1 - c)
        factor = a * ((coupon / 2) + c + d) - b
        return round(factor, 4)

    def dlv(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        repo_rate: Union[float, Dual, Dual2, list, tuple],
        settlement: datetime,
        delivery: Union[datetime, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
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
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        convention: str, optional
            The day count convention applied to the repo rates.
        dirty: bool
            Whether the bond prices are given including accrued interest. Default is *False*.

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
            bond.ytm(prices[i], settlement, dirty=dirty) for i, bond in enumerate(self.basket)
        ]
        df["C.Factor"] = self.cfs
        df["Gross Basis"] = self.gross_basis(future_price, prices, settlement, dirty=dirty)
        df["Implied Repo"] = self.implied_repo(
            future_price, prices, settlement, delivery, convention, dirty=dirty
        )
        df["Actual Repo"] = r_
        df["Net Basis"] = self.net_basis(
            future_price, prices, r_, settlement, delivery, convention, dirty=dirty
        )
        df["Bond"] = [
            f"{bond.fixed_rate:,.3f}% " f"{bond.leg1.schedule.termination.strftime('%d-%m-%Y')}"
            for bond in self.basket
        ]
        return df

    def cms(
        self,
        prices: list[float],
        settlement: datetime,
        shifts: list[float],
        delivery: Union[datetime, NoInput] = NoInput(0),
        dirty: bool = False,
    ):
        """
        Perform CTD multi-security analysis.

        Parameters
        ----------
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        shifts : list of float
            The scenarios to analyse.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        dirty: bool
            Whether the bond prices are given including accrued interest. Default is *False*.

        Returns
        -------
        DataFrame

        Notes
        -----
        This method only operates when the CTD basket has multiple securities
        """
        if len(self.basket) == 1:
            raise ValueError("Multi-security analysis cannot be performed with one security.")
        delivery = self.delivery[1] if delivery is NoInput.blank else delivery

        # build a curve for pricing
        today = add_tenor(
            settlement,
            f"-{self.basket[0].kwargs['settle']}B",
            None,
            self.basket[0].leg1.schedule.calendar,
        )
        unsorted_nodes = {
            today: 1.0,
            **{_.leg1.schedule.termination: 1.0 for _ in self.basket}
        }
        bcurve = Curve(
            nodes=dict(sorted(unsorted_nodes.items(), key=lambda _: _[0])),
            convention="act365f"  # use the most natural DCF without scaling
        )
        if dirty:
            metric = "dirty_price"
        else:
            metric = "clean_price"
        solver = Solver(
            curves=[bcurve],
            instruments=[(_, (), {"curves": bcurve, "metric": metric}) for _ in self.basket],
            s=prices,
        )
        if solver.result["status"] != "SUCCESS":
            return ValueError(
                "A bond curve could not be solved for analysis. "
                "See 'Cookbook: Bond Future CTD Multi-Security Analysis'."
            )
        bcurve._set_ad_order(order=0)  # turn of AD for efficiency

        data = {
            "Bond": [
                f"{bond.fixed_rate:,.3f}% " f"{bond.leg1.schedule.termination.strftime('%d-%m-%Y')}"
                for bond in self.basket
            ]
        }
        for shift in shifts:
            _curve = bcurve.shift(shift, composite=False)
            data.update(
                {
                    shift: self.net_basis(
                        future_price=self.rate(curves=_curve),
                        prices=[_.rate(curves=_curve, metric=metric) for _ in self.basket],
                        repo_rate=_curve.rate(settlement, self.delivery[1], "NONE"),
                        settlement=settlement,
                        delivery=delivery,
                        convention=_curve.convention,
                        dirty=dirty,
                    )
                }
            )

        _ = DataFrame(data=data)
        return _

    def gross_basis(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        settlement: Union[datetime, NoInput] = NoInput(0),
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
                prices[i] - bond.accrued(settlement) for i, bond in enumerate(self.basket)
            )
        else:
            prices_ = prices
        return tuple(prices_[i] - self.cfs[i] * future_price for i in range(len(self.basket)))

    def net_basis(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        repo_rate: Union[float, Dual, Dual2, list, tuple],
        settlement: datetime,
        delivery: Union[datetime, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
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
        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery

        if not isinstance(repo_rate, (list, tuple)):
            r_ = (repo_rate,) * len(self.basket)
        else:
            r_ = repo_rate

        if dirty:
            net_basis_ = tuple(
                bond.fwd_from_repo(prices[i], settlement, f_settlement, r_[i], convention, dirty=dirty)
                - self.cfs[i] * future_price - bond.accrued(f_settlement)
                for i, bond in enumerate(self.basket)
            )
        else:
            net_basis_ = tuple(
                bond.fwd_from_repo(prices[i], settlement, f_settlement, r_[i], convention, dirty=dirty)
                - self.cfs[i] * future_price
                for i, bond in enumerate(self.basket)
            )
        return net_basis_

    def implied_repo(
        self,
        future_price: Union[float, Dual, Dual2],
        prices: list[float, Dual, Dual2],
        settlement: datetime,
        delivery: Union[datetime, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
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
        if delivery is NoInput.blank:
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
        delivery: Union[datetime, NoInput] = NoInput(0),
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
        if delivery is NoInput.blank:
            settlement = self.delivery[1]
        else:
            settlement = delivery
        adjusted_prices = [future_price * cf for cf in self.cfs]
        yields = tuple(
            bond.ytm(adjusted_prices[i], settlement) for i, bond in enumerate(self.basket)
        )
        return yields

    def duration(
        self,
        future_price: float,
        metric: str = "risk",
        delivery: Union[datetime, NoInput] = NoInput(0),
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
        if delivery is NoInput.blank:
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
        delivery: Union[datetime, NoInput] = NoInput(0),
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
        if delivery is NoInput.blank:
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
        delivery: Union[datetime, NoInput] = NoInput(0),
        dirty: bool = False,
        ordered: bool = False,
    ):
        """
        Determine the index of the CTD in the basket from implied repo rate.

        Parameters
        ----------
        future_price : float
            The price of the future.
        prices: sequence of float, Dual, Dual2
            The prices of the bonds in the deliverable basket (ordered).
        settlement: datetime
            The settlement date of the bonds.
        delivery: datetime, optional
            The date of the futures delivery. If not given uses the final delivery
            day.
        dirty: bool
            Whether the bond prices are given including accrued interest.
        ordered : bool, optional
            Whether to return the sorted order of CTD indexes and not just a single index for
            the specific CTD.

        Returns
        -------
        int
        """
        implied_repo = self.implied_repo(
            future_price, prices, settlement, delivery, "Act365F", dirty
        )
        if not ordered:
            ctd_index_ = implied_repo.index(max(implied_repo))
            return ctd_index_
        else:
            _ = {i: v for (i, v) in zip(range(len(implied_repo)), implied_repo)}
            _ = {k: v for k, v in sorted(_.items(), key=lambda item: -item[1])}
            return list(_.keys())

    # Digital Methods

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric: str = "future_price",
        delivery: Union[datetime, NoInput] = NoInput(0),
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

        if delivery is NoInput.blank:
            f_settlement = self.delivery[1]
        else:
            f_settlement = delivery
        prices_ = [
            bond.rate(curves, solver, fx, base, "clean_price", f_settlement) for bond in self.basket
        ]
        future_prices_ = [price / self.cfs[i] for i, price in enumerate(prices_)]
        future_price = min(future_prices_)
        ctd_index = future_prices_.index(min(future_prices_))

        if metric == "future_price":
            return future_price
        elif metric == "ytm":
            return self.basket[ctd_index].ytm(future_price * self.cfs[ctd_index], f_settlement)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


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
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

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
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        frequency: Union[int, NoInput] = NoInput(0),
        stub: Union[str, NoInput] = NoInput(0),
        front_stub: Union[datetime, NoInput] = NoInput(0),
        back_stub: Union[datetime, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        amortization: Union[float, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        leg2_effective: Union[datetime, NoInput] = NoInput(1),
        leg2_termination: Union[datetime, str, NoInput] = NoInput(1),
        leg2_frequency: Union[int, NoInput] = NoInput(1),
        leg2_stub: Union[str, NoInput] = NoInput(1),
        leg2_front_stub: Union[datetime, NoInput] = NoInput(1),
        leg2_back_stub: Union[datetime, NoInput] = NoInput(1),
        leg2_roll: Union[str, int, NoInput] = NoInput(1),
        leg2_eom: Union[bool, NoInput] = NoInput(1),
        leg2_modifier: Union[str, NoInput] = NoInput(1),
        leg2_calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(1),
        leg2_payment_lag: Union[int, NoInput] = NoInput(1),
        leg2_notional: Union[float, NoInput] = NoInput(-1),
        leg2_currency: Union[str, NoInput] = NoInput(1),
        leg2_amortization: Union[float, NoInput] = NoInput(-1),
        leg2_convention: Union[str, NoInput] = NoInput(1),
        curves: Union[list, str, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ):
        self.kwargs = dict(
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
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            leg2_effective=leg2_effective,
            leg2_termination=leg2_termination,
            leg2_frequency=leg2_frequency,
            leg2_stub=leg2_stub,
            leg2_front_stub=leg2_front_stub,
            leg2_back_stub=leg2_back_stub,
            leg2_roll=leg2_roll,
            leg2_eom=leg2_eom,
            leg2_modifier=leg2_modifier,
            leg2_calendar=leg2_calendar,
            leg2_payment_lag=leg2_payment_lag,
            leg2_notional=leg2_notional,
            leg2_currency=leg2_currency,
            leg2_amortization=leg2_amortization,
            leg2_convention=leg2_convention,
        )
        self.kwargs = _push(spec, self.kwargs)
        # set some defaults if missing
        self.kwargs["notional"] = (
            defaults.notional
            if self.kwargs["notional"] is NoInput.blank
            else self.kwargs["notional"]
        )
        if self.kwargs["payment_lag"] is NoInput.blank:
            self.kwargs["payment_lag"] = defaults.payment_lag_specific[type(self).__name__]
        self.kwargs = _inherit_or_negate(self.kwargs)  # inherit or negate the complete arg list

        self.curves = curves
        self.spec = spec

        #
        # for attribute in [
        #     "effective",
        #     "termination",
        #     "frequency",
        #     "stub",
        #     "front_stub",
        #     "back_stub",
        #     "roll",
        #     "eom",
        #     "modifier",
        #     "calendar",
        #     "payment_lag",
        #     "convention",
        #     "notional",
        #     "amortization",
        #     "currency",
        # ]:
        #     leg2_val, val = self.kwargs[f"leg2_{attribute}"], self.kwargs[attribute]
        #     if leg2_val is NoInput.inherit:
        #         _ = val
        #     elif leg2_val == NoInput.negate:
        #         _ = NoInput(0) if val is NoInput(0) else val * -1
        #     else:
        #         _ = leg2_val
        #     self.kwargs[attribute] = val
        #     self.kwargs[f"leg2_{attribute}"] = _
        #     # setattr(self, attribute, val)
        #     # setattr(self, f"leg2_{attribute}", _)

    @abstractmethod
    def _set_pricing_mid(self, *args, **kwargs):  # pragma: no cover
        pass

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


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
        fixed_rate: Union[float, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, Series, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)

        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
    ):
        # the test for an unpriced IRS is that its fixed rate is not set.
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of generic methods NPV will be zero.
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
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        leg2_npv = self.leg2.npv(curves[2], curves[3])
        return self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100
        # leg1_analytic_delta = self.leg1.analytic_delta(curves[0], curves[1])
        # return leg2_npv / (leg1_analytic_delta * 100)

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def spread(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        specified_spd = 0 if self.leg2.float_spread is NoInput(0) else self.leg2.float_spread
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        return self.leg2._spread(-irs_npv, curves[2], curves[3]) + specified_spd
        # leg2_analytic_delta = self.leg2.analytic_delta(curves[2], curves[3])
        # return irs_npv / leg2_analytic_delta + specified_spd


class STIRFuture(IRS):
    """
    Create a short term interest rate (STIR) future.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    price : float
        The traded price of the future. Defined as 100 minus the fixed rate.
    contracts : int
        The number of traded contracts.
    bp_value : float.
        The value of 1bp on the contract as specified by the exchange, e.g. SOFR 3M futures are
        $25 per bp. This is not the same as tick value where the tick size can be different across
        different futures.
    nominal : float
        The nominal value of the contract. E.g. SOFR 3M futures are $1mm. If not given will use the
        default notional.
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
           id="usd_stir"
       )

    Create the *STIRFuture*, and demonstrate the :meth:`~rateslib.instruments.STIRFuture.rate`,
    :meth:`~rateslib.instruments.STIRFuture.npv`,

    .. ipython:: python

       stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves=usd,
            price=99.50,
            contracts=10,
        )
       stir.rate(metric="price")
       stir.npv()

    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    def __init__(
        self,
        *args,
        price: Union[float, NoInput] = NoInput(0),
        contracts: int = 1,
        bp_value: Union[float, NoInput] = NoInput(0),
        nominal: Union[float, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, Series, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        nominal = defaults.notional if nominal is NoInput.blank else nominal
        # TODO this overwrite breaks positional arguments
        kwargs["notional"] = nominal * contracts * -1.0
        super(IRS, self).__init__(*args, **kwargs)  # call BaseDerivative.__init__()
        user_kwargs = dict(
            price=price,
            fixed_rate=NoInput(0) if price is NoInput.blank else (100 - price),
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            nominal=nominal,
            bp_value=bp_value,
            contracts=contracts,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)

        self._fixed_rate = self.kwargs["fixed_rate"]
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FixedLeg(
            **_get(self.kwargs, leg=1, filter=["price", "nominal", "bp_value", "contracts"])
        )
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        # the test for an unpriced IRS is that its fixed rate is not set.
        mid_price = self.rate(curves, solver, fx, base, metric="price")
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            self.leg1.fixed_rate = float(100 - mid_price)

        traded_price = 100 - self.leg1.fixed_rate
        _ = (mid_price - traded_price) * 100 * self.kwargs["contracts"] * self.kwargs["bp_value"]
        if local:
            return {self.leg1.currency: _}
        else:
            return _

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        metric: str = "rate",
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
        metric : str in {"rate", "price"}
            The calculation metric that will be returned.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        leg2_npv = self.leg2.npv(curves[2], curves[3])

        _ = self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100
        if metric.lower() == "rate":
            return _
        elif metric.lower() == "price":
            return 100 - _
        else:
            raise ValueError("`metric` must be in {'price', 'rate'}.")

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *STIRFuture*.

        See :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        For *STIRFuture* this method requires no arguments.
        """
        return -1.0 * self.kwargs["contracts"] * self.kwargs["bp_value"]

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        return DataFrame.from_records(
            [
                {
                    defaults.headers["type"]: type(self).__name__,
                    defaults.headers["stub_type"]: "Regular",
                    defaults.headers["currency"]: self.leg1.currency.upper(),
                    defaults.headers["a_acc_start"]: self.leg1.schedule.effective,
                    defaults.headers["a_acc_end"]: self.leg1.schedule.termination,
                    defaults.headers["payment"]: None,
                    defaults.headers["convention"]: "Exchange",
                    defaults.headers["dcf"]: float(self.leg1.notional)
                    / self.kwargs["nominal"]
                    * self.kwargs["bp_value"]
                    / 100.0,
                    defaults.headers["notional"]: float(self.leg1.notional),
                    defaults.headers["df"]: 1.0,
                    defaults.headers["collateral"]: self.leg1.currency.lower(),
                }
            ]
        )

    def spread(self):
        """
        Not implemented for *STIRFuture*.
        """
        return NotImplementedError()


# class Swap(IRS):
#     """
#     Alias for :class:`~rateslib.instruments.IRS`.
#     """


class IIRS(BaseDerivative):
    """
    Create an indexed interest rate swap (IIRS) composing an
    :class:`~rateslib.legs.IndexFixedLeg` and a :class:`~rateslib.legs.FloatLeg`.

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

    Create the IIRS, and demonstrate the :meth:`~rateslib.instruments.IIRS.rate`, and
    :meth:`~rateslib.instruments.IIRS.npv`.

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
        fixed_rate: Union[float, NoInput] = NoInput(0),
        index_base: Union[float, Series, NoInput] = NoInput(0),
        index_fixings: Union[float, Series, NoInput] = NoInput(0),
        index_method: Union[str, NoInput] = NoInput(0),
        index_lag: Union[int, NoInput] = NoInput(0),
        notional_exchange: Union[bool, NoInput] = False,
        payment_lag_exchange: Union[int, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        leg2_payment_lag_exchange: Union[int, NoInput] = NoInput(1),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange is NoInput.inherit:
            leg2_payment_lag_exchange = payment_lag_exchange
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            index_base=index_base,
            index_fixings=index_fixings,
            index_method=index_method,
            index_lag=index_lag,
            initial_exchange=False,
            final_exchange=notional_exchange,
            payment_lag_exchange=payment_lag_exchange,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            leg2_payment_lag_exchange=leg2_payment_lag_exchange,
            leg2_initial_exchange=False,
            leg2_final_exchange=notional_exchange,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)

        self._index_base = self.kwargs["index_base"]
        self._fixed_rate = self.kwargs["fixed_rate"]
        self.leg1 = IndexFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
    ):
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = float(mid_market_rate)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if self.index_base is NoInput.blank:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx_, base_, local)

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if self.index_base is NoInput.blank:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx_, base_)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if self.index_base is NoInput.blank:
            # must forecast for the leg
            self.leg1.index_base = curves[0].index_value(
                self.leg1.schedule.effective, self.leg1.index_method
            )
        leg2_npv = self.leg2.npv(curves[2], curves[3])

        if self.fixed_rate is NoInput.blank:
            self.leg1.fixed_rate = 0.0
        _existing = self.leg1.fixed_rate
        leg1_npv = self.leg1.npv(curves[0], curves[1])

        _ = self.leg1._spread(-leg2_npv - leg1_npv, curves[0], curves[1]) / 100
        return _ + _existing

    def spread(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        specified_spd = 0 if self.leg2.float_spread is NoInput.blank else self.leg2.float_spread
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
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
        fixed_rate: Union[float, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, Series, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = ZeroFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = ZeroFloatLeg(**_get(self.kwargs, leg=2))

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta<rateslib.instruments.BaseDerivative.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def _set_pricing_mid(self, curves, solver):
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = float(mid_market_rate)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        leg2_npv = self.leg2.npv(curves[2], curves[3])
        _ = self.leg1._spread(-leg2_npv, curves[0], curves[1]) / 100
        return _

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
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
        fixed_rate: Union[float, NoInput] = NoInput(0),
        leg2_index_base: Union[float, Series, NoInput] = NoInput(0),
        leg2_index_fixings: Union[float, Series, NoInput] = NoInput(0),
        leg2_index_method: Union[str, NoInput] = NoInput(0),
        leg2_index_lag: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            leg2_index_base=leg2_index_base,
            leg2_index_fixings=leg2_index_fixings,
            leg2_index_lag=leg2_index_lag,
            leg2_index_method=leg2_index_method,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_index_base = leg2_index_base
        self.leg1 = ZeroFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = ZeroIndexLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(self, curves, solver):
        if self.fixed_rate is NoInput.blank:
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = float(mid_market_rate)

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if self.leg2_index_base is NoInput.blank:
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
        float_spread: Union[float, NoInput] = NoInput(0),
        spread_compound_method: Union[str, NoInput] = NoInput(0),
        fixings: Union[float, list, Series, NoInput] = NoInput(0),
        fixing_method: Union[str, NoInput] = NoInput(0),
        method_param: Union[int, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, Series, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)
        self._float_spread = float_spread
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FloatLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(self, curves, solver):
        if self.float_spread is NoInput.blank and self.leg2_float_spread is NoInput.blank:
            # set a pricing parameter for the purpose of pricing NPV at zero.
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
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative object by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        if leg == 1:
            leg_obj, args = self.leg1, (curves[0], curves[1])
        else:
            leg_obj, args = self.leg2, (curves[2], curves[3])

        specified_spd = 0 if leg_obj.float_spread is NoInput.blank else leg_obj.float_spread
        return leg_obj._spread(-core_npv, *args) + specified_spd

        # irs_npv = self.npv(curves, solver)
        # curves, _ = self._get_curves_and_fx_maybe_from_solver(solver, curves, None)
        # if leg == 1:
        #     args = (curves[0], curves[1])
        # else:
        #     args = (curves[2], curves[3])
        # leg_analytic_delta = getattr(self, f"leg{leg}").analytic_delta(*args)
        # adjust = getattr(self, f"leg{leg}").float_spread
        # adjust = 0 if adjust is NoInput.blank else adjust
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
    Create a forward rate agreement composing single period :class:`~rateslib.periods.FixedLeg`
    and :class:`~rateslib.periods.FloatLeg` valued in a customised manner.

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
    FRAs are a legacy derivative whose *FloatLeg* ``fixing_method`` is set to *"ibor"*.

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
        effective: Union[datetime, NoInput] = NoInput(0),
        termination: Union[datetime, str, NoInput] = NoInput(0),
        frequency: Union[int, NoInput] = NoInput(0),
        roll: Union[str, int, NoInput] = NoInput(0),
        eom: Union[bool, NoInput] = NoInput(0),
        modifier: Union[str, None, NoInput] = NoInput(0),
        calendar: Union[CustomBusinessDay, str, NoInput] = NoInput(0),
        payment_lag: Union[int, NoInput] = NoInput(0),
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        method_param: Union[int, NoInput] = NoInput(0),
        fixed_rate: Union[float, NoInput] = NoInput(0),
        fixings: Union[float, Series, NoInput] = NoInput(0),
        curves: Union[str, list, Curve, NoInput] = NoInput(0),
        spec: Union[str, NoInput] = NoInput(0),
    ) -> None:
        self.kwargs = dict(
            effective=effective,
            termination=termination,
            frequency=_upper(frequency),
            roll=roll,
            eom=eom,
            modifier=_upper(modifier),
            calendar=get_calendar(calendar),
            payment_lag=payment_lag,
            notional=notional,
            currency=_lower(currency),
            convention=_upper(convention),
            fixed_rate=fixed_rate,
            leg2_effective=NoInput(1),
            leg2_termination=NoInput(1),
            leg2_convention=NoInput(1),
            leg2_frequency=NoInput(1),
            leg2_notional=NoInput(-1),
            leg2_modifier=NoInput(1),
            leg2_currency=NoInput(1),
            leg2_calendar=NoInput(1),
            leg2_eom=NoInput(1),
            leg2_payment_lag=NoInput(1),
            leg2_fixing_method="ibor",
            leg2_method_param=method_param,
            leg2_spread_compound_method="none_simple",
            leg2_fixings=fixings,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            currency=defaults.base_currency,
            modifier=defaults.modifier,
            eom=defaults.eom,
            convention=defaults.convention,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)
        self.kwargs = _inherit_or_negate(self.kwargs)

        # Build
        self.curves = curves

        self._fixed_rate = self.kwargs["fixed_rate"]
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

        if self.leg1.schedule.n_periods != 1 or self.leg2.schedule.n_periods != 1:
            raise ValueError("FRA scheduling inputs did not define a single period.")

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
    ) -> None:
        if self.fixed_rate is NoInput.blank:
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = mid_market_rate.real

    def analytic_delta(
        self,
        curve: Curve,
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the FRA.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve_: Curve = _disc_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.leg1.currency, fx, base)
        rate = self.rate([curve])
        _ = (
            self.leg1.notional
            * self.leg1.periods[0].dcf
            * disc_curve_[self.leg1.schedule.pschedule[0]]
            / 10000
        )
        return fx * _ / (1 + self.leg1.periods[0].dcf * rate / 100)

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ) -> DualTypes:
        """
        Return the NPV of the derivative.

        See :meth:`BaseDerivative.npv`.
        """

        self._set_pricing_mid(curves, solver)
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        fx, base = _get_fx_and_base(self.leg1.currency, fx_, base_)
        value = self.cashflow(curves[0]) * curves[1][self.leg1.schedule.pschedule[0]]
        if local:
            return {self.leg1.currency: value}
        else:
            return fx * value

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        return self.leg2.periods[0].rate(curves[0])

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
        cf1 = self.leg1.periods[0].cashflow
        cf2 = self.leg2.periods[0].cashflow(curve)
        if cf1 is not NoInput.blank and cf2 is not NoInput.blank:
            cf = cf1 + cf2
        else:
            return None
        rate = (
            None
            if curve is NoInput.blank
            else 100 * cf2 / (-self.leg2.notional * self.leg2.periods[0].dcf)
        )
        cf /= 1 + self.leg1.periods[0].dcf * rate / 100

        # if self.fixed_rate is NoInput.blank:
        #     return 0  # set the fixed rate = to floating rate netting to zero
        # rate = self.leg2.rate(curve)
        # cf = self.notional * self.leg1.dcf * (rate - self.fixed_rate) / 100
        # cf /= 1 + self.leg1.dcf * rate / 100
        return cf

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        self._set_pricing_mid(curves, solver)
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )
        fx_, base_ = _get_fx_and_base(self.leg1.currency, fx_, base_)

        cf = float(self.cashflow(curves[0]))
        df = float(curves[1][self.leg1.schedule.pschedule[0]])
        npv_local = cf * df

        _fix = None if self.fixed_rate is NoInput.blank else -float(self.fixed_rate)
        _spd = None if curves[1] is NoInput.blank else -float(self.rate(curves[1])) * 100
        cfs = self.leg1.periods[0].cashflows(curves[0], curves[1], fx_, base_)
        cfs[defaults.headers["type"]] = "FRA"
        cfs[defaults.headers["payment"]] = self.leg1.schedule.pschedule[0]
        cfs[defaults.headers["cashflow"]] = cf
        cfs[defaults.headers["rate"]] = _fix
        cfs[defaults.headers["spread"]] = _spd
        cfs[defaults.headers["npv"]] = npv_local
        cfs[defaults.headers["df"]] = df
        cfs[defaults.headers["fx"]] = float(fx_)
        cfs[defaults.headers["npv_fx"]] = npv_local * float(fx_)
        return DataFrame.from_records([cfs])

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


# Multi-currency derivatives


class XCS(BaseDerivative):
    """
    Create a cross-currency swap (XCS) composing relevant fixed or floating *Legs*.

    MTM-XCSs will introduce a MTM *Leg* as *Leg2*.

    Parameters
    ----------
    args : tuple
        Required positional arguments for :class:`~rateslib.instruments.BaseDerivative`.
    fixed : bool, optional
        Whether *leg1* is fixed or floating rate. Defaults to *False*.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    fixed_rate : float, optional
        If ``fixed``, the fixed rate of *leg1*.
    float_spread : float, optional
        If not ``fixed``, the spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        If not ``fixed``, the method to use for adding a floating spread to compounded rates.
        Available options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, or Series optional
        If not ``fixed``, then if a float scalar, will be applied as the determined fixing for
        the first period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    fixing_method : str, optional
        If not ``fixed``, the method by which floating rates are determined, set by default.
        See notes.
    method_param : int, optional
        If not ``fixed`` A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_fixed : bool, optional
        Whether *leg2* is fixed or floating rate. Defaults to *False*
    leg2_mtm : bool optional
        Whether *leg2* is a mark-to-market leg. Defaults to *True*
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_fixed_rate : float, optional
        If ``leg2_fixed``, the fixed rate of *leg2*.
    leg2_float_spread : float, optional
        If not ``leg2_fixed``, the spread applied to the :class:`~rateslib.legs.FloatLeg`.
        Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        If not ``leg2_fixed``, the method to use for adding a floating spread to compounded rates.
        Available options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If not ``leg2_fixed``, then if a float scalar, will be applied as the determined fixing for
        the first period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        If not ``leg2_fixed``, the method by which floating rates are determined, set by default.
        See notes.
    leg2_method_param : int, optional
        If not ``leg2_fixed`` A parameter that is used for the various ``fixing_method`` s.
        See notes.
    fx_fixings : float, Dual, Dual2, list of such, optional
        Specify a known initial FX fixing or a list of such for ``mtm`` legs, where leg 1 is
        considered the domestic currency. For example for an ESTR/SOFR XCS in 100mm EUR notional
        a value of 1.10 for fx_fixings implies the notional on leg 2 is 110m USD. Fixings that
        are not specified will be forecast at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    kwargs : dict
        Required keyword arguments for :class:`~rateslib.instruments.BaseDerivative`.
    """

    def __init__(
        self,
        *args,
        fixed: Union[bool, NoInput] = NoInput(0),
        payment_lag_exchange: Union[int, NoInput] = NoInput(0),
        fixed_rate: Union[float, NoInput] = NoInput(0),
        float_spread: Union[float, NoInput] = NoInput(0),
        spread_compound_method: Union[str, NoInput] = NoInput(0),
        fixings: Union[float, list, Series, NoInput] = NoInput(0),
        fixing_method: Union[str, NoInput] = NoInput(0),
        method_param: Union[int, NoInput] = NoInput(0),
        leg2_fixed: Union[bool, NoInput] = NoInput(0),
        leg2_mtm: Union[bool, NoInput] = NoInput(0),
        leg2_payment_lag_exchange: Union[int, NoInput] = NoInput(1),
        leg2_fixed_rate: Union[float, NoInput] = NoInput(0),
        leg2_float_spread: Union[float, NoInput] = NoInput(0),
        leg2_fixings: Union[float, list, NoInput] = NoInput(0),
        leg2_fixing_method: Union[str, NoInput] = NoInput(0),
        leg2_method_param: Union[int, NoInput] = NoInput(0),
        leg2_spread_compound_method: Union[str, NoInput] = NoInput(0),
        fx_fixings: Union[list, DualTypes, FXRates, FXForwards, NoInput] = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # set defaults for missing values
        default_kwargs = dict(
            fixed=False if fixed is NoInput.blank else fixed,
            leg2_fixed=False if leg2_fixed is NoInput.blank else leg2_fixed,
            leg2_mtm=True if leg2_mtm is NoInput.blank else leg2_mtm,
        )
        self.kwargs = _update_not_noinput(self.kwargs, default_kwargs)

        if self.kwargs["fixed"]:
            self._fixed_rate_mixin = True
            self._fixed_rate = fixed_rate
            leg1_user_kwargs = dict(fixed_rate=fixed_rate)
            Leg1 = FixedLeg
        else:
            self._rate_scalar = 100.0
            self._float_spread_mixin = True
            self._float_spread = float_spread
            leg1_user_kwargs = dict(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                fixings=fixings,
                fixing_method=fixing_method,
                method_param=method_param,
            )
            Leg1 = FloatLeg
        leg1_user_kwargs.update(
            dict(
                payment_lag_exchange=payment_lag_exchange,
                initial_exchange=True,
                final_exchange=True,
            )
        )

        if leg2_payment_lag_exchange is NoInput.inherit:
            leg2_payment_lag_exchange = payment_lag_exchange
        if self.kwargs["leg2_fixed"]:
            self._leg2_fixed_rate_mixin = True
            self._leg2_fixed_rate = leg2_fixed_rate
            leg2_user_kwargs = dict(leg2_fixed_rate=leg2_fixed_rate)
            Leg2 = FixedLeg if not leg2_mtm else FixedLegMtm
        else:
            self._leg2_float_spread_mixin = True
            self._leg2_float_spread = leg2_float_spread
            leg2_user_kwargs = dict(
                leg2_float_spread=leg2_float_spread,
                leg2_spread_compound_method=leg2_spread_compound_method,
                leg2_fixings=leg2_fixings,
                leg2_fixing_method=leg2_fixing_method,
                leg2_method_param=leg2_method_param,
            )
            Leg2 = FloatLeg if not leg2_mtm else FloatLegMtm
        leg2_user_kwargs.update(
            dict(
                leg2_payment_lag_exchange=leg2_payment_lag_exchange,
                leg2_initial_exchange=True,
                leg2_final_exchange=True,
            )
        )

        if self.kwargs["leg2_mtm"]:
            self._is_mtm = True
            leg2_user_kwargs.update(
                dict(
                    leg2_alt_currency=self.kwargs["currency"],
                    leg2_alt_notional=-self.kwargs["notional"],
                    leg2_fx_fixings=fx_fixings,
                )
            )
        else:
            self._is_mtm = False

        self.kwargs = _update_not_noinput(self.kwargs, {**leg1_user_kwargs, **leg2_user_kwargs})

        self.leg1 = Leg1(**_get(self.kwargs, leg=1, filter=["fixed"]))
        self.leg2 = Leg2(**_get(self.kwargs, leg=2, filter=["leg2_fixed", "leg2_mtm"]))
        self._initialise_fx_fixings(fx_fixings)

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
            # if self.fx_fixing is NoInput.blank this indicates the swap is unfixed and will be set
            # later. If a fixing is given this means the notional is fixed without any
            # further sensitivity, hence the downcast to a float below.
            if isinstance(fx_fixings, FXForwards):
                self.fx_fixings = float(fx_fixings.rate(self.pair, self.leg2.periods[0].payment))
            elif isinstance(fx_fixings, FXRates):
                self.fx_fixings = float(fx_fixings.rate(self.pair))
            elif isinstance(fx_fixings, (float, Dual, Dual2)):
                self.fx_fixings = float(fx_fixings)
            else:
                self._fx_fixings = NoInput(0)
        else:
            self._fx_fixings = fx_fixings

    def _set_fx_fixings(self, fx):
        """
        Checks the `fx_fixings` and sets them according to given object if null.

        Used by ``rate`` and ``npv`` methods when ``fx_fixings`` are not
        initialised but required for pricing and can be inferred from an FX object.
        """
        if not self._is_mtm:  # then we manage the initial FX from the pricing object.
            if self.fx_fixings is NoInput.blank:
                if fx is NoInput.blank:
                    if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                        raise ValueError(
                            "`fx` is required when `fx_fixings` is not pre-set and "
                            "if rateslib option `no_fx_fixings_for_xcs` is set to "
                            "'raise'."
                        )
                    else:
                        fx_fixing = 1.0
                        if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                            warnings.warn(
                                "Using 1.0 for FX, no `fx` or `fx_fixings` given and "
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
            if self.fixed_rate is NoInput.blank or self.leg2_fixed_rate is NoInput.blank:
                return True
            return False
        elif self._fixed_rate_mixin and self.fixed_rate is NoInput.blank:
            # Fixed/Float where fixed leg is unpriced
            return True
        elif self._float_spread_mixin and self.float_spread is NoInput.blank:
            # Float leg1 where leg1 is
            pass  # goto 2)
        else:
            return False

        # 2) leg1 is Float
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is NoInput.blank:
            return True
        elif self._leg2_float_spread_mixin and self.leg2_float_spread is NoInput.blank:
            return True
        else:
            return False

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
    ):
        leg: int = 1
        lookup = {
            1: ["_fixed_rate_mixin", "_float_spread_mixin"],
            2: ["_leg2_fixed_rate_mixin", "_leg2_float_spread_mixin"],
        }
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is NoInput.blank:
            # Fixed/Fixed or Float/Fixed
            leg = 2

        rate = self.rate(curves, solver, fx, leg=leg)
        if getattr(self, lookup[leg][0]):
            getattr(self, f"leg{leg}").fixed_rate = float(rate)
        elif getattr(self, lookup[leg][1]):
            getattr(self, f"leg{leg}").float_spread = float(rate)
        else:
            # this line should not be hit: internal code check
            raise AttributeError("BaseXCS leg1 must be defined fixed or float.")  # pragma: no cover

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        .. warning::

           If ``fx_fixing`` has not been set for the instrument requires
           ``fx`` as an FXForwards object to dynamically determine this.

        See :meth:`BaseDerivative.npv`.
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx_)

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().npv(curves, solver, fx_, base_, local)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset for next calculation
        return ret

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, NoInput(0), self.leg1.currency
        )

        if leg == 1:
            tgt_fore_curve, tgt_disc_curve = curves[0], curves[1]
            alt_fore_curve, alt_disc_curve = curves[2], curves[3]
        else:
            tgt_fore_curve, tgt_disc_curve = curves[2], curves[3]
            alt_fore_curve, alt_disc_curve = curves[0], curves[1]

        leg2 = 1 if leg == 2 else 2
        # tgt_str, alt_str = "" if leg == 1 else "leg2_", "" if leg2 == 1 else "leg2_"
        tgt_leg, alt_leg = getattr(self, f"leg{leg}"), getattr(self, f"leg{leg2}")
        base_ = tgt_leg.currency

        _is_float_tgt_leg = "Float" in type(tgt_leg).__name__
        _is_float_alt_leg = "Float" in type(alt_leg).__name__
        if not _is_float_alt_leg and getattr(alt_leg, "fixed_rate") is NoInput.blank:
            raise ValueError(
                "Cannot solve for a `fixed_rate` or `float_spread` where the "
                "`fixed_rate` on the non-solvable leg is NoInput.blank."
            )

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
        # Commercial use of this code, and/or copying and redistribution is prohibited.
        # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

        if not _is_float_tgt_leg and getattr(tgt_leg, "fixed_rate") is NoInput.blank:
            # set the target fixed leg to a null fixed rate for calculation
            tgt_leg.fixed_rate = 0.0

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        tgt_leg_npv = tgt_leg.npv(tgt_fore_curve, tgt_disc_curve, fx_, base_)
        alt_leg_npv = alt_leg.npv(alt_fore_curve, alt_disc_curve, fx_, base_)
        fx_a_delta = 1.0 if not tgt_leg._is_mtm else fx_
        _ = tgt_leg._spread(
            -(tgt_leg_npv + alt_leg_npv), tgt_fore_curve, tgt_disc_curve, fx_a_delta
        )

        specified_spd = 0.0
        if _is_float_tgt_leg and not (getattr(tgt_leg, "float_spread") is NoInput.blank):
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
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, base, self.leg1.currency
        )

        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx_)

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().cashflows(curves, solver, fx_, base_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc
        return ret


class FXSwap(XCS):
    """
    Create an FX swap simulated via a *Fixed-Fixed* :class:`XCS`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`XCS`.
    fx_fixings : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for `fx0`
        implies the notional on leg 2 is 110m USD. If `None` determines this
        dynamically.
    points : float, optional
        The pricing parameter for the FX Swap, which will determine the implicit
        fixed rate on leg2.
    split_notional : float, optional
        The accrued notional at termination of the domestic leg accounting for interest
        payable at domestic interest rates.
    kwargs : dict
        Required keyword arguments to :class:`XCS`.

    Notes
    -----
    .. warning::

       ``leg2_notional`` is determined by the ``fx_fixings`` either initialised or at price
       time and the value of ``notional``. The argument value of ``leg2_notional`` does
       not impact calculations.

    *FXSwaps* are technically complicated instruments. To define a fully **priced** *Instrument*
    they require at least two pricing parameters; ``fx_fixings`` and ``points``. If a
    ``split_notional`` is also given at initialisation it will be assumed to be a split notional
    *FXSwap*. If not, then it will not be assumed to be.

    If ``fx_fixings`` is given then the market pricing parameter ``points`` can be calculated.
    This is an unusual partially *priced* parametrisation, however, and a warning will be emitted.
    As before, if ``split_notional`` is given, or not, at initialisation the *FXSwap* will be
    assumed to be split notional or not.

    If the *FXSwap* is not initialised with any parameters this defines an **unpriced**
    *Instrument* and it will be assumed to be split notional, inline with interbank
    market standards. The mid-market rate of an unpriced FXSwap is the same regardless of whether
    it is split notional or not, albeit split notional FXSwaps result in smaller FX rate
    sensitivity.

    Other combinations of arguments, just providing ``points`` or ``split_notional`` or both of
    those will raise an error. An *FXSwap* cannot be parametrised by these in isolation. This is
    summarised in the below table.

    .. list-table::  Resultant initialisation dependent upon given pricing parameters.
       :widths: 10 10 10 70
       :header-rows: 1

       * - fx_fixings
         - points
         - split_notional
         - Result
       * - X
         - X
         - X
         - A fully *priced* instrument defined with split notionals.
       * - X
         - X
         -
         - A fully *priced* instruments without split notionals.
       * -
         -
         -
         - An *unpriced* instrument with assumed split notionals.
       * - X
         -
         - X
         - A partially priced instrument with split notionals. Warns about unconventionality.
       * - X
         -
         -
         - A partially priced instrument without split notionals. Warns about unconventionality.
       * -
         - X
         - X
         - Raises ValueError. Not allowable partially priced instrument.
       * -
         - X
         -
         - Raises ValueError. Not allowable partially priced instrument.
       * -
         -
         - X
         - Raises ValueError. Not allowable partially priced instrument.

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
           effective=dt(2022, 1, 18),
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

       fxs.npv(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.rate(curves=[None, usd, None, eurusd], fx=fxf)

    In the case of *FXSwaps*, whose mid-market price is the difference between two
    forward FX rates we can also derive this quantity using the independent
    :meth:`FXForwards.swap<rateslib.fx.FXForwards.swap>` method.

    .. ipython:: python

       fxf.swap("usdeur", [dt(2022, 1, 18), dt(2022, 4, 19)])

    The following is an example of a fully priced *FXSwap* with split notionals.

    .. ipython:: python

       fxs = FXSwap(
           effective=dt(2022, 1, 18),
           termination=dt(2022, 4, 19),
           calendar="nyc",
           currency="usd",
           notional=1000000,
           leg2_currency="eur",
           curves=["usd", "usd", "eur", "eurusd"],
           fx_fixings=0.90,
           split_notional=1001500,
           points=-49.0
       )
       fxs.npv(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.cashflows(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.cashflows_table(curves=[None, usd, None, eurusd], fx=fxf)

    """

    _unpriced = True

    def _parse_split_flag(self, fx_fixings, points, split_notional):
        """
        Determine the rules for a priced, unpriced or partially priced derivative and whether
        it is inferred as split notional or not.
        """
        is_none = [_ is NoInput.blank for _ in [fx_fixings, points, split_notional]]
        if all(is_none) or not any(is_none):
            self._is_split = True
        elif split_notional is NoInput.blank and not any(
            [_ is NoInput.blank for _ in [fx_fixings, points]]
        ):
            self._is_split = False
        elif fx_fixings is not NoInput.blank:
            warnings.warn(
                "Initialising FXSwap with `fx_fixings` but without `points` is unconventional.\n"
                "Pricing can still be performed to determine `points`.",
                UserWarning,
            )
            if split_notional is not NoInput.blank:
                self._is_split = True
            else:
                self._is_split = False
        else:
            if points is not NoInput.blank:
                raise ValueError("Cannot initialise FXSwap with `points` but without `fx_fixings`.")
            else:
                raise ValueError(
                    "Cannot initialise FXSwap with `split_notional` but without `fx_fixings`"
                )

    def _set_split_notional(self, curve: Union[Curve, NoInput] = NoInput(0), at_init: bool = False):
        """
        Will set the fixed rate, if not zero, for leg1, given provided split not or forecast splnot.

        self._split_notional is used as a temporary storage when mid market price is determined.
        """
        if not self._is_split:
            self._split_notional = self.kwargs["notional"]
            # fixed rate at zero remains
        elif self.kwargs["split_notional"] is not NoInput.blank:
            if at_init:  # this will be run for one time only at initialisation
                self._split_notional = self.kwargs["split_notional"]
                self._set_leg1_fixed_rate()
            else:
                return None
        else:
            if at_init:
                self._split_notional = None
            else:
                dt1, dt2 = self.leg1.periods[0].payment, self.leg1.periods[2].payment
                self._split_notional = self.kwargs["notional"] * curve[dt1] / curve[dt2]
                self._set_leg1_fixed_rate()

    def _set_leg1_fixed_rate(self):
        fixed_rate = (self.leg1.notional - self._split_notional) / (
            -self.leg1.notional * self.leg1.periods[1].dcf
        )
        self.leg1.fixed_rate = fixed_rate * 100

    def __init__(
        self,
        *args,
        fx_fixings: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        points: Union[float, NoInput] = NoInput(0),
        split_notional: Union[float, NoInput] = NoInput(0),
        **kwargs,
    ):
        self._parse_split_flag(fx_fixings, points, split_notional)

        kwargs_overrides = dict(  # specific args for FXSwap passed to the Base XCS
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=False,
            fixed_rate=0.0,
            frequency="Z",
            leg2_frequency="Z",
            leg2_fixed_rate=NoInput(0),
            fx_fixings=fx_fixings,
        )
        super().__init__(*args, **{**kwargs, **kwargs_overrides})

        self.kwargs["split_notional"] = split_notional
        self._set_split_notional(curve=None, at_init=True)
        # self._initialise_fx_fixings(fx_fixings)
        self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._unpriced = False
        self._points = value
        self._leg2_fixed_rate = NoInput(0)

        # setting points requires leg1.notional leg1.split_notional, fx_fixing and points value

        if value is not NoInput.blank:
            # leg2 should have been properly set as part of fx_fixings and set_leg2_notional
            fx_fixing = self.leg2.notional / -self.leg1.notional

            _ = self._split_notional * (fx_fixing + value / 10000) + self.leg2.notional
            fixed_rate = _ / (self.leg2.periods[1].dcf * -self.leg2.notional)

            self.leg2_fixed_rate = fixed_rate * 100

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International

    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _set_pricing_mid(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
    ):
        # This function ASSUMES that the instrument is unpriced, i.e. all of
        # split_notional, fx_fixing and points have been initialised as None.

        # first we set the split notional which is defined by interest rates on leg1.
        points = self.rate(curves, solver, fx)
        self.points = float(points)
        self._unpriced = True  # setting pricing mid does not define a priced instrument

    def rate(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
        fixed_rate: bool = False,
    ):
        """
        Return the mid-market pricing parameter of the FXSwapS.

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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves, solver, curves, fx, NoInput(0), self.leg1.currency
        )
        # set the split notional from the curve if not available
        self._set_split_notional(curve=curves[1])
        # then we will set the fx_fixing and leg2 initial notional.

        # self._set_fx_fixings(fx) # this will be done by super().rate()
        leg2_fixed_rate = super().rate(curves, solver, fx_, leg=2)

        if fixed_rate:
            return leg2_fixed_rate
        else:
            points = -self.leg2.notional * (
                (1 + leg2_fixed_rate * self.leg2.periods[1].dcf / 100) / self._split_notional
                - 1 / self.kwargs["notional"]
            )
            return points * 10000

    def cashflows(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx)
        ret = super().cashflows(curves, solver, fx, base)
        return ret


# Generic Instruments


class Spread(Sensitivities):
    """
    A spread instrument defined as the difference in rate between two *Instruments*.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s.
    instrument2 : Instrument
        The second instrument, usually the longest tenor, e.g. 10Y in 5s10s.

    Notes
    -----
    When using a :class:`Spread` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    Creating a dynamic :class:`Spread` where the *Instruments* are dynamically priced,
    and each share the pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q")
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q")
       spread = Spread(irs1, irs2)
       spread.npv(curve1)
       spread.rate(curve1)
       spread.cashflows(curve1)

    Creating an assigned :class:`Spread`, where each *Instrument* has its own
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

    _rate_scalar = 100.0

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
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
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
        return (leg2_rate - leg1_rate) * 100.0

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

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


# class SpreadX:
#     pass


class Fly(Sensitivities):
    """
    A butterfly instrument which is, mechanically, the spread of two spread instruments.

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
    When using a :class:`Fly` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    _rate_scalar = 100.0

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
        return (-leg3_rate + 2 * leg2_rate - leg1_rate) * 100.0

    def cashflows(self, *args, **kwargs):
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
                self.instrument3.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2", "instrument3"],
        )

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


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


def _instrument_npv(instrument, *args, **kwargs):  # pragma: no cover
    # this function is captured by TestPortfolio pooling but is not registered as a parallel process
    # used for parallel processing with Portfolio.npv
    return instrument.npv(*args, **kwargs)


class Portfolio(Sensitivities):
    """
    Create a collection of *Instruments* to group metrics

    Parameters
    ----------
    instruments : list
        This should be a list of *Instruments*.

    Notes
    -----
    When using a :class:`Portfolio` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    def __init__(self, instruments):
        self.instruments = instruments

    def npv(
        self,
        curves: Union[Curve, str, list, NoInput] = NoInput(0),
        solver: Union[Solver, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
        **kwargs,
    ):
        """
        Return the NPV of the *Portfolio* by summing instrument NPVs.

        For arguments see :meth:`BaseDerivative.npv()<rateslib.instruments.BaseDerivative.npv>`.
        """
        # TODO look at legs.npv where args len is used.
        if not local and base is NoInput.blank and fx is NoInput.blank:
            warnings.warn(
                "No ``base`` currency is inferred, using ``local`` output. To return a single "
                "PV specify a ``base`` currency and ensure an ``fx`` or ``solver.fx`` object "
                "is available to perform the conversion if the currency differs from the local.",
                UserWarning,
            )
            local = True

        # if the pool is 1 do not do any parallel processing and return the single core func
        if defaults.pool == 1:
            return self._npv_single_core(
                curves=curves, solver=solver, fx=fx, base=base, local=local, **kwargs
            )

        from multiprocessing import Pool
        from functools import partial

        func = partial(
            _instrument_npv,
            curves=curves,
            solver=solver,
            fx=fx,
            base=base,
            local=local,
            **kwargs,
        )
        p = Pool(defaults.pool)
        results = p.map(func, self.instruments)
        p.close()

        if local:
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
            # dicts = [instrument.npv(*args, **kwargs) for instrument in self.instruments]
            # result = dict(reduce(operator.add, map(Counter, dicts)))

            ret = {}
            for instrument in self.instruments:
                i_npv = instrument.npv(*args, **kwargs)
                for ccy in i_npv:
                    if ccy in ret:
                        ret[ccy] += i_npv[ccy]
                    else:
                        ret[ccy] = i_npv[ccy]
        else:
            _ = (instrument.npv(*args, **kwargs) for instrument in self.instruments)
            ret = sum(_)
        return ret

    def cashflows(self, *args, **kwargs):
        return concat(
            [_.cashflows(*args, **kwargs) for _ in self.instruments],
            keys=[f"inst{i}" for i in range(len(self.instruments))],
        )

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


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
        return _ytm_quadratic_converger2(f, 2 * y0 - y2, y1 - y2 + y0, y0, None, None, f0, tol)
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
            return _ytm_quadratic_converger2(f, y0 - pad, y, 2 * y - y0 + pad, None, f_, None, tol)
        else:
            return _ytm_quadratic_converger2(f, 2 * y - y1 - pad, y, y1 + pad, None, f_, None, tol)
    elif y1 < y <= y2:
        if (y - y1) < (y2 - y):
            return _ytm_quadratic_converger2(f, y1 - pad, y, 2 * y - y1 + pad, None, f_, None, tol)
        else:
            return _ytm_quadratic_converger2(f, 2 * y - y2 - pad, y, y2 + pad, None, f_, None, tol)
    else:  # y2 < y:
        # line not hit due to reassessment of initial vars?
        return _ytm_quadratic_converger2(
            f, y2 - pad, y, 2 * y - y2 + pad, None, f_, None, tol
        )  # pragma: no cover


# def _brents(f, x0, x1, max_iter=50, tolerance=1e-9):  # pragma: no cover
#     """
#     Alternative yield converger as an alternative to ytm_converger
#
#     Unused currently within the library
#     """
#     fx0 = f(x0)
#     fx1 = f(x1)
#
#     if float(fx0 * fx1) > 0:
#         raise ValueError(
#             "`brents` must initiate from function values with opposite signs."
#         )
#
#     if abs(fx0) < abs(fx1):
#         x0, x1 = x1, x0
#         fx0, fx1 = fx1, fx0
#
#     x2, fx2 = x0, fx0
#
#     mflag = True
#     steps_taken = 0
#
#     while steps_taken < max_iter and abs(x1 - x0) > tolerance:
#         fx0 = f(x0)
#         fx1 = f(x1)
#         fx2 = f(x2)
#
#         if fx0 != fx2 and fx1 != fx2:
#             L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
#             L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
#             L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
#             new = L0 + L1 + L2
#
#         else:
#             new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))
#
#         if (
#             (float(new) < float((3 * x0 + x1) / 4) or float(new) > float(x1))
#             or (mflag is True and (abs(new - x1)) >= (abs(x1 - x2) / 2))
#             or (mflag is False and (abs(new - x1)) >= (abs(x2 - d) / 2))
#             or (mflag is True and (abs(x1 - x2)) < tolerance)
#             or (mflag is False and (abs(x2 - d)) < tolerance)
#         ):
#             new = (x0 + x1) / 2
#             mflag = True
#
#         else:
#             mflag = False
#
#         fnew = f(new)
#         d, x2 = x2, x1
#
#         if float(fx0 * fnew) < 0:
#             x1 = new
#         else:
#             x0 = new
#
#         if abs(fx0) < abs(fx1):
#             x0, x1 = x1, x0
#
#         steps_taken += 1
#
#     return x1, steps_taken


def _quadratic_equation(a: float, b: float, c: float):
    """
    solver the equation ax^2 + bx + c = 0, via the quadratic formula.
    """
    # perform the quadratic solution
    _1 = -c / b  # approximate linear solution: applicable for most situations.
    discriminant = b**2 - 4 * a * c
    if abs(a) > 1e-14:
        _2a = (-b - discriminant**0.5) / (2 * a)
        _2b = (-b + discriminant**0.5) / (2 * a)  # alt quadratic soln
        if abs(_1 - _2a) < abs(_1 - _2b):
            _ = _2a
        else:
            _ = _2b  # select quadratic soln
    else:
        # this is to avoid divide by zero errors and return an approximation
        # also isda_flat_compounding has a=0
        _ = _1

    return _


def _get(kwargs: dict, leg: int = 1, filter=[]):
    """
    A parser to return kwarg dicts for relevant legs.
    Internal structuring only.
    Will return kwargs relevant to leg1 OR leg2.
    Does not return keys that are specified in the filter.
    """
    if leg == 1:
        _ = {k: v for k, v in kwargs.items() if "leg2" not in k and k not in filter}
    else:
        _ = {k[5:]: v for k, v in kwargs.items() if "leg2_" in k and k not in filter}
    return _


def _push(spec: Optional[str], kwargs: dict):
    """
    Push user specified kwargs to a default specification.
    Values from the `spec` dict will not overwrite specific user values already in `kwargs`.
    """
    if spec is NoInput.blank:
        return kwargs
    else:
        try:
            spec_kwargs = defaults.spec[spec.lower()]
        except KeyError:
            raise ValueError(f"Given `spec`, '{spec}', cannot be found in defaults.")

        user = {k: v for k, v in kwargs.items() if v not in [NoInput(0), NoInput(1), NoInput(-1)]}
        return {**kwargs, **spec_kwargs, **user}


def _update_not_noinput(base_kwargs, new_kwargs):
    """
    Update the `base_kwargs` with `new_kwargs` (user values) unless those new values are NoInput.
    """
    updaters = {
        k: v for k, v in new_kwargs.items() if k not in base_kwargs or not isinstance(v, NoInput)
    }
    return {**base_kwargs, **updaters}


def _update_with_defaults(base_kwargs, default_kwargs):
    """
    Update the `base_kwargs` with `default_kwargs` if the values are NoInput.blank.
    """
    updaters = {
        k: v
        for k, v in default_kwargs.items()
        if k in base_kwargs and base_kwargs[k] is NoInput.blank
    }
    return {**base_kwargs, **updaters}


def _inherit_or_negate(kwargs: dict, ignore_blank=False):
    """Amend the values of leg2 kwargs if they are defaulted to inherit or negate from leg1."""

    def _replace(k, v):
        # either inherit or negate the value in leg2 from that in leg1
        if "leg2_" in k:
            if not isinstance(v, NoInput):
                return v  # do nothing if the attribute is an input

            try:
                leg1_v = kwargs[k[5:]]
            except KeyError:
                return v

            if leg1_v is NoInput.blank:
                if ignore_blank:
                    return v  # this allows an inheritor or negator to be called a second time
                else:
                    return NoInput(0)

            if v is NoInput(-1):
                return leg1_v * -1.0
            elif v is NoInput(1):
                return leg1_v
        return v  # do nothing to leg1 attributes

    return {k: _replace(k, v) for k, v in kwargs.items()}


def _lower(val: Union[str, NoInput]):
    if isinstance(val, str):
        return val.lower()
    return val


def _upper(val: Union[str, NoInput]):
    if isinstance(val, str):
        return val.upper()
    return val
