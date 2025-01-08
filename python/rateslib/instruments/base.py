from __future__ import annotations

import abc
import warnings
from typing import Any

from pandas import DataFrame, concat, isna

from rateslib.default import NoInput
from rateslib.dual import DualTypes
from rateslib.instruments.utils import FX, NPV, Curves, _get_curves_fx_and_base_maybe_from_solver
from rateslib.solver import Solver


class BaseMixin:
    _fixed_rate_mixin = False
    _float_spread_mixin = False
    _leg2_fixed_rate_mixin = False
    _leg2_float_spread_mixin = False
    _index_base_mixin = False
    _leg2_index_base_mixin = False
    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes | NoInput:
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
    def fixed_rate(self, value: DualTypes | NoInput) -> None:
        if not self._fixed_rate_mixin:
            raise AttributeError("Cannot set `fixed_rate` for this Instrument.")
        self._fixed_rate = value
        self.leg1.fixed_rate = value

    @property
    def leg2_fixed_rate(self) -> DualTypes | NoInput:
        """
        float or None : If set will also set the ``fixed_rate`` of the contained
        leg2.
        """
        return self._leg2_fixed_rate

    @leg2_fixed_rate.setter
    def leg2_fixed_rate(self, value: DualTypes | NoInput) -> None:
        if not self._leg2_fixed_rate_mixin:
            raise AttributeError("Cannot set `leg2_fixed_rate` for this Instrument.")
        self._leg2_fixed_rate = value
        self.leg2.fixed_rate = value

    @property
    def float_spread(self) -> DualTypes | NoInput:
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg1.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes | NoInput) -> None:
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
    def leg2_float_spread(self) -> DualTypes | NoInput:
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg2.
        """
        return self._leg2_float_spread

    @leg2_float_spread.setter
    def leg2_float_spread(self, value: DualTypes | NoInput) -> None:
        if not self._leg2_float_spread_mixin:
            raise AttributeError("Cannot set `leg2_float_spread` for this Instrument.")
        self._leg2_float_spread = value
        self.leg2.float_spread = value

    @property
    def index_base(self) -> DualTypes | NoInput:
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
    def index_base(self, value: DualTypes | NoInput) -> None:
        if not self._index_base_mixin:
            raise AttributeError("Cannot set `index_base` for this Instrument.")
        self._index_base = value
        self.leg1.index_base = value

    @property
    def leg2_index_base(self) -> DualTypes | NoInput:
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
    def leg2_index_base(self, value: DualTypes | NoInput) -> None:
        if not self._leg2_index_base_mixin:
            raise AttributeError("Cannot set `leg2_index_base` for this Instrument.")
        self._leg2_index_base = value
        self.leg2.index_base = value

    @abc.abstractmethod
    def analytic_delta(self, *args: Any, leg=1, **kwargs: Any) -> DualTypes:
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
           :suppress:

           from rateslib import Curve, FXRates, IRS, dt

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
        curves: Curves = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
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
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
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
        curves: Curves = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
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
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg1_npv: NPV = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
        leg2_npv: NPV = self.leg2.npv(curves[2], curves[3], fx_, base_, local)
        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)
                for k in set(leg1_npv) | set(leg2_npv)  # type: ignore[union-attr]
            }
        else:
            return leg1_npv + leg2_npv  # type: ignore[operator]

    @abc.abstractmethod
    def rate(self, *args: Any, **kwargs: Any) -> DualTypes:
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

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"
