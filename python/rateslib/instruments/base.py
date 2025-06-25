from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame, concat, isna

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.instruments.sensitivities import Sensitivities
from rateslib.instruments.utils import (
    _get_curves_fx_and_base_maybe_from_solver,
    _inherit_or_negate,
    _push,
)
from rateslib.solver import Solver

if TYPE_CHECKING:
    from rateslib.typing import FX_, NPV, Any, CalInput, Curves_, DualTypes, Leg, Solver_, str_


class Metrics:
    """
    Base class for *Instruments* adding optional pricing parameters, such as fixed rates,
    float spreads etc. Also provides key pricing methods.
    """

    _fixed_rate_mixin: bool = False
    _float_spread_mixin: bool = False
    _leg2_fixed_rate_mixin: bool = False
    _leg2_float_spread_mixin: bool = False
    _index_base_mixin: bool = False
    _leg2_index_base_mixin: bool = False
    _rate_scalar: float = 1.0

    leg1: Leg
    leg2: Leg
    curves: Curves_

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
        self.leg1.fixed_rate = value  # type: ignore[union-attr]

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
        self.leg2.fixed_rate = value  # type: ignore[union-attr]

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
        self.leg1.float_spread = value  # type: ignore[union-attr]
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
        self.leg2.float_spread = value  # type: ignore[union-attr]

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
        self.leg1.index_base = value  # type: ignore[union-attr]

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
        self.leg2.index_base = value  # type: ignore[union-attr]

    @abstractmethod
    def analytic_delta(self, *args: Any, leg: int = 1, **kwargs: Any) -> DualTypes:
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
        _: DualTypes = getattr(self, f"leg{leg}").analytic_delta(*args, **kwargs)
        return _

    @abstractmethod
    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
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
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        df1 = self.leg1.cashflows(curves_[0], curves_[1], fx_, base_)
        df2 = self.leg2.cashflows(curves_[2], curves_[3], fx_, base_)
        # filter empty or all NaN
        dfs_filtered = [_ for _ in [df1, df2] if not (_.empty or isna(_).all(axis=None))]

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _: DataFrame = concat(dfs_filtered, keys=["leg1", "leg2"])
        return _

    @abstractmethod
    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
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
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg1_npv: NPV = self.leg1.npv(curves_[0], curves_[1], fx_, base_, local)
        leg2_npv: NPV = self.leg2.npv(curves_[2], curves_[3], fx_, base_, local)
        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)  # type: ignore[union-attr]
                for k in set(leg1_npv) | set(leg2_npv)  # type: ignore[arg-type]
            }
        else:
            return leg1_npv + leg2_npv  # type: ignore[operator]

    @abstractmethod
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

    def cashflows_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> DataFrame:
        """
        Aggregate the values derived from a :meth:`~rateslib.instruments.BaseMixin.cashflows`
        method on an *Instrument*.

        Parameters
        ----------
        curves : CurveType, str or list of such, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        solver : Solver, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        fx : float, FXRates, FXForwards, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        base : str, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        kwargs : dict
            Additional arguments input the underlying ``cashflows`` method of the *Instrument*.

        Returns
        -------
        DataFrame
        """
        cashflows = self.cashflows(curves, solver, fx, base, **kwargs)
        cashflows = cashflows[
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
                defaults.headers["cashflow"],
            ]
        ]
        _: DataFrame = cashflows.groupby(  # type: ignore[assignment]
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
            ],
            dropna=False,
        )
        _ = _.sum().unstack([0, 1]).droplevel(0, axis=1)  # type: ignore[arg-type]
        _.columns.names = ["local_ccy", "collateral_ccy"]
        _.index.names = ["payment"]
        _ = _.sort_index(ascending=True, axis=0).infer_objects().fillna(0.0)
        return _


class BaseDerivative(Sensitivities, Metrics, metaclass=ABCMeta):
    """
    Abstract base class with common parameters for many *Derivative* subclasses.

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
    kwargs: dict[str, Any]
        A record of the input arguments to the *Instrument*.
    curves: Curves_
        Curves associated with the *Instrument* used in pricing methods.
    spec: str_
        The default configuration used to populate arguments.
    """

    @abstractmethod
    def __init__(
        self,
        effective: datetime | NoInput = NoInput(0),
        termination: datetime | str | NoInput = NoInput(0),
        frequency: int | NoInput = NoInput(0),
        stub: str | NoInput = NoInput(0),
        front_stub: datetime | NoInput = NoInput(0),
        back_stub: datetime | NoInput = NoInput(0),
        roll: str | int | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        amortization: float | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        leg2_effective: datetime | NoInput = NoInput(1),
        leg2_termination: datetime | str | NoInput = NoInput(1),
        leg2_frequency: int | NoInput = NoInput(1),
        leg2_stub: str | NoInput = NoInput(1),
        leg2_front_stub: datetime | NoInput = NoInput(1),
        leg2_back_stub: datetime | NoInput = NoInput(1),
        leg2_roll: str | int | NoInput = NoInput(1),
        leg2_eom: bool | NoInput = NoInput(1),
        leg2_modifier: str | NoInput = NoInput(1),
        leg2_calendar: CalInput = NoInput(1),
        leg2_payment_lag: int | NoInput = NoInput(1),
        leg2_notional: float | NoInput = NoInput(-1),
        leg2_currency: str | NoInput = NoInput(1),
        leg2_amortization: float | NoInput = NoInput(-1),
        leg2_convention: str | NoInput = NoInput(1),
        curves: Curves_ = NoInput(0),
        spec: str | NoInput = NoInput(0),
    ):
        self.kwargs: dict[str, Any] = dict(
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

    @abstractmethod
    def _set_pricing_mid(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def exo_delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*, measured against user
        defined :class:`~rateslib.dual.Variable` s.

        For arguments see
        :meth:`Sensitivities.exo_delta()<rateslib.instruments.Sensitivities.exo_delta>`.
        """
        return super().exo_delta(*args, **kwargs)
