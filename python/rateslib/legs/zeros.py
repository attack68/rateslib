from __future__ import annotations

import warnings
from math import prod
from typing import TYPE_CHECKING

from pandas import DataFrame, concat

from rateslib import defaults
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.default import NoInput
from rateslib.dual.utils import _dual_float
from rateslib.legs.base import BaseLeg, _FixedLegMixin, _FloatLegMixin
from rateslib.periods import FixedPeriod, FloatPeriod
from rateslib.periods.utils import _get_fx_and_base, _validate_float_args

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        Curve,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        Schedule,
        datetime,
    )


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
    schedule: Schedule

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
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
        ret: float = sum(_)
        return ret

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
        compounded_rate: DualTypes = prod(rates)
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
            return concat(dfs)

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
        compounded_rate: DualTypes = prod(rates)

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
        disc_curve: Curve_ = NoInput(0),
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
                collateral = disc_curve_.meta.collateral
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
        ret: float = sum(_)
        return ret

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
            collateral = disc_curve_.meta.collateral

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
        ret: DualTypes = sum(_)
        return ret

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
