from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.legs.components import FixedLeg, FloatLeg
from rateslib.scheduling import Adjuster

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        FloatRateSeries,
        Frequency,
        FXForwards_,
        FXVolOption_,
        RollDay,
        Solver_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FRA(_BaseInstrument):
    """
    A *forward rate agreement (FRA)* compositing a
    :class:`~rateslib.legs.components.FixedLeg` and :class:`~rateslib.legs.components.FloatLeg`.

    Parameters
    ----------
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

    Notes
    -----
    A *STIRFuture* is modelled as a single period *IRS* whose payment date is overloaded to always
    result in immediate settlement, with the immediate date derived from the discount curve
    used during pricing.

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
       :suppress:

       from rateslib import STIRFuture

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

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.components.FixedLeg`."""
        return self.leg1.fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self.kwargs.leg1["fixed_rate"] = value
        self.leg1.fixed_rate = value

    @property
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.components.FixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> FloatLeg:
        """The :class:`~rateslib.legs.components.FloatLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An STIRFuture has two curve requirements: a leg2_rate_curve and a disc_curve used by
        both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_rate_curve=_drb(
                    curves.get("rate_curve", NoInput(0)),
                    curves.get("leg2_rate_curve", NoInput(0)),
                ),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    leg2_rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 1:
                return _Curves(
                    leg2_rate_curve=curves[0],
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 4:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                leg2_rate_curve=curves,
                disc_curve=curves,
                leg2_disc_curve=curves,
            )

    def __init__(
        self,
        # scheduling
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: Frequency | str_ = NoInput(0),
        *,
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int = 0,
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        leg2_rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        leg2_method_param: int_ = NoInput(0),
        leg2_fixing_frequency: Frequency | str_ = NoInput(0),
        leg2_fixing_series: FloatRateSeries | str_ = NoInput(0),
        # meta parameters
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> None:
        user_args = dict(
            # scheduling
            effective=effective,
            termination=termination,
            frequency=frequency,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            ex_div=ex_div,
            convention=convention,
            # settlement
            currency=currency,
            notional=notional,
            # rate
            fixed_rate=fixed_rate,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_method_param=leg2_method_param,
            leg2_fixing_series=leg2_fixing_series,
            leg2_fixing_frequency=leg2_fixing_frequency,
            # meta
            curves=self._parse_curves(curves),
            metric=metric,
        )
        instrument_args = dict(
            leg2_effective=NoInput.inherit,
            leg2_termination=NoInput.inherit,
            leg2_frequency=NoInput.inherit,
            leg2_roll=NoInput.inherit,
            leg2_eom=NoInput.inherit,
            leg2_modifier=NoInput.inherit,
            leg2_calendar=NoInput.inherit,
            leg2_payment_lag=NoInput.inherit,
            leg2_ex_div=NoInput.inherit,
            leg2_convention=NoInput.inherit,
            leg2_float_spread=0.0,
            leg2_fixing_method=FloatFixingMethod.IBOR,
            leg2_spread_compound_method=SpreadCompoundMethod.NoneSimple,
            leg2_notional=NoInput.negate,
            leg2_currency=NoInput.inherit,
            payment_lag=Adjuster.BusDaysLagSettleInAdvance(payment_lag),
            initial_exchange=False,
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
        )
        default_args = dict(
            nominal=defaults.notional,
            metric="rate",
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric"],
        )

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

        if self._leg1.schedule.n_periods != 1:
            raise ValueError(
                "The scheduling parameters of the STIRFuture must define exactly "
                f"one regular period. Got '{self.leg1.schedule.n_periods}'."
            )

    def _fra_rate_scalar(self, leg2_rate_curve) -> DualTypes_:
        r = self.leg2._regular_periods[0].rate(rate_curve=leg2_rate_curve)
        return 1 / (1 + self.leg2._regular_periods[0].period_params.dcf * r / 100.0)

    def npv(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        self._set_pricing_mid(
            curves=curves,
            solver=solver,
            settlement=settlement,
            forward=forward,
        )
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _get_maybe_curve_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        fra_scalar = self._fra_rate_scalar(leg2_rate_curve=leg2_rate_curve)

        npv = super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )
        if local:
            return {k: v * fra_scalar for k, v in npv.items()}
        else:
            return npv * fra_scalar

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IRS is that its fixed rate is not set.
        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            mid_market_rate = self.rate(
                curves=curves,
                solver=solver,
                settlement=settlement,
                forward=forward,
            )
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def rate(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes_:
        _curves = self._parse_curves(curves)
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        leg2_npv: DualTypes = self.leg2.local_npv(
            rate_curve=_get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
            ),
            disc_curve=_get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            settlement=settlement,
            forward=forward,
        )
        rate = (
            self.leg1.spread(
                target_npv=-leg2_npv,
                rate_curve=NoInput(0),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
            / 100
        )
        if metric_ == "rate":
            return rate
        else:
            raise ValueError("`metric` must be in {'rate'}.")

    def analytic_delta(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _get_maybe_curve_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        fra_scalar = self._fra_rate_scalar(leg2_rate_curve=leg2_rate_curve)
        return fra_scalar * super().analytic_delta(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
            leg=leg,
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        df = self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        return df * self._fra_rate_scalar(
            leg2_rate_curve=_get_maybe_curve_maybe_from_solver(
                solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
            )
        )

    def cashflows(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        df = super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        leg2_rate_curve = _get_maybe_curve_maybe_from_solver(
            solver=solver, curves_meta=_curves_meta, curves=_curves, name="leg2_rate_curve"
        )
        scalar = _dual_float(self._fra_rate_scalar(leg2_rate_curve=leg2_rate_curve))

        headers = [
            defaults.headers["cashflow"],
            defaults.headers["npv"],
            defaults.headers["npv_fx"],
        ]
        for header in headers:
            df[header] = df[header] * scalar

        return df
