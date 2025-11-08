from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from rateslib import defaults
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.legs.components import FloatLeg

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        FXVolOption_,
        RollDay,
        Solver_,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class SBS(_BaseInstrument):
    _rate_scalar = 100.0

    @property
    def fixed_rate(self) -> DualTypes_:
        raise AttributeError(f"Attribute not available on {type(self).__name__}")

    @property
    def float_spread(self) -> DualTypes_:
        return self.leg1.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self.kwargs.leg1["float_spread"] = value
        self.leg1.float_spread = value

    @property
    def leg2_fixed_rate(self) -> NoReturn:
        raise AttributeError(f"Attribute not available on {type(self).__name__}")

    @property
    def leg2_float_spread(self) -> DualTypes_:
        return self.leg2.float_spread

    @leg2_float_spread.setter
    def leg2_float_spread(self, value: DualTypes) -> None:
        self.kwargs.leg2["float_spread"] = value
        self.leg2.float_spread = value

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: Frequency | str_ = NoInput(0),
        *,
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        notional: float_ = NoInput(0),
        currency: str_ = NoInput(0),
        amortization: float_ = NoInput(0),
        convention: str_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        leg2_rate_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        leg2_fixing_method: str_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_effective: datetime_ = NoInput(1),
        leg2_termination: datetime | str_ = NoInput(1),
        leg2_frequency: Frequency | str_ = NoInput(1),
        leg2_stub: str_ = NoInput(1),
        leg2_front_stub: datetime_ = NoInput(1),
        leg2_back_stub: datetime_ = NoInput(1),
        leg2_roll: int | RollDay | str_ = NoInput(1),
        leg2_eom: bool_ = NoInput(1),
        leg2_modifier: str_ = NoInput(1),
        leg2_calendar: CalInput = NoInput(1),
        leg2_payment_lag: int_ = NoInput(1),
        leg2_payment_lag_exchange: int_ = NoInput(1),
        leg2_notional: float_ = NoInput(-1),
        leg2_amortization: float_ = NoInput(-1),
        leg2_convention: str_ = NoInput(1),
        leg2_ex_div: int_ = NoInput(1),
        curves: Curves_ = NoInput(0),
        metric: str_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        user_args = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            rate_fixings=rate_fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            ex_div=ex_div,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_rate_fixings=leg2_rate_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
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
            leg2_payment_lag_exchange=leg2_payment_lag_exchange,
            leg2_ex_div=leg2_ex_div,
            leg2_notional=leg2_notional,
            leg2_amortization=leg2_amortization,
            leg2_convention=leg2_convention,
            curves=self._parse_curves(curves),
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            leg2_currency=NoInput(1),
            initial_exchange=False,
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
        )

        default_args = dict(
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_exchange,
            metric=defaults.metric[type(self).__name__],
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "metric"],
        )

        self.leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self.leg2 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg2, 1))
        self._legs = [self.leg1, self.leg2]

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
        metric_: str = _drb(self.kwargs.meta["metric"], metric)
        _curves = self._parse_curves(curves)

        if metric_.lower() == "float_spread":
            leg2_npv: DualTypes = self.leg2.local_npv(
                rate_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
                ),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
            return self.leg1.spread(
                target_npv=-leg2_npv,
                rate_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "rate_curve", solver
                ),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
        else:  # metric == "leg2_float_spread"
            leg1_npv: DualTypes = self.leg1.local_npv(
                rate_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "rate_curve", solver
                ),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )
            return self.leg2.spread(
                target_npv=-leg1_npv,
                rate_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "leg2_rate_curve", solver
                ),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
                ),
                index_curve=NoInput(0),
                settlement=settlement,
                forward=forward,
            )

    def spread(
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
    ) -> DualTypes:
        return self.rate(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            settlement=settlement,
            forward=forward,
            metric=metric,
        )

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
        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IRS is that its fixed rate is not set.
        if self.kwargs.meta["metric"].lower() == "float_spread":
            if isinstance(self.kwargs.leg1["float_spread"], NoInput):
                # set a fixed rate for the purpose of generic methods NPV will be zero.
                mid_market_rate = self.rate(
                    curves=curves,
                    solver=solver,
                    settlement=settlement,
                    forward=forward,
                )
                self.leg1.float_spread = _dual_float(mid_market_rate)
        else:  # metric == "leg2_float_spread"
            if isinstance(self.kwargs.leg2["float_spread"], NoInput):
                # set a fixed rate for the purpose of generic methods NPV will be zero.
                mid_market_rate = self.rate(
                    curves=curves,
                    solver=solver,
                    settlement=settlement,
                    forward=forward,
                )
                self.leg2.float_spread = _dual_float(mid_market_rate)

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An SBS has three curve requirements:

        - a rate_curve
        - a disc_curve
        - a leg2_rate_curve

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements this will raise an Exception.
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
            if len(curves) == 2 or len(curves) == 1 or len(curves) > 4:
                raise TypeError(f"Number of `curves` for an SBS must be 3 or 4. Got {len(curves)}.")
            elif len(curves) == 3:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[1],
                )
            else:  # == 4
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[3],
                )
        else:  # `curves` is just a single input
            raise TypeError("Number of `curves` for an SBS must be 3 or 4. Got 1.")

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
        return super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            settlement=settlement,
            forward=forward,
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
        return self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )
