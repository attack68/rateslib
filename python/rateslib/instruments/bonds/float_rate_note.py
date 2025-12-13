from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod
from rateslib.instruments.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FloatLeg
from rateslib.periods import FloatPeriod
from rateslib.scheduling import Frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FloatRateSeries,
        FXForwards_,
        LegFixings,
        Sequence,
        Solver_,
        VolT_,
        _BaseLeg,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FloatRateNote(_BaseBondInstrument):
    """
    Create a floating rate note (FRN) security.

    """

    _rate_scalar = 1.0

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of the composited
        :class:`~rateslib.legs.FloatLeg`."""
        return self.leg1.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self.kwargs.leg1["float_spread"] = value
        self.leg1.float_spread = value

    @property
    def leg1(self) -> FloatLeg:
        """The :class:`~rateslib.legs.FloatLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: int_ = NoInput(0),
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement params
        currency: str_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        # rate params
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        rate_fixings: LegFixings = NoInput(0),
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # meta parameters
        curves: CurvesT_ = NoInput(0),
        calc_mode: BondCalcMode | str_ = NoInput(0),
        settle: int_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ) -> None:
        user_args = dict(
            # scheduling
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
            payment_lag_exchange=payment_lag_exchange,
            ex_div=ex_div,
            convention=convention,
            # settlement
            currency=currency,
            notional=notional,
            amortization=amortization,
            # rate
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            rate_fixings=rate_fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            fixing_frequency=fixing_frequency,
            fixing_series=fixing_series,
            # meta
            curves=self._parse_curves(curves),
            calc_mode=calc_mode,
            settle=settle,
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=False,
            final_exchange=True,
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            calc_mode=defaults.calc_mode[type(self).__name__],
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "calc_mode", "settle", "metric", "vol"],
        )
        self.kwargs.meta["calc_mode"] = _get_bond_calc_mode(self.kwargs.meta["calc_mode"])

        self._leg1 = FloatLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        if self._leg1.schedule.frequency_obj == Frequency.Zero():
            raise ValueError("A `FloatRateNote` cannot have a 'zero' frequency.")
        self._legs = [self.leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An FRN has two curve requirements: a leg2_rate_curve and a disc_curve used by both legs.

        When given as only 1 element this curve is applied to all of the those components

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                )
            elif len(curves) == 1:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[0],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
            )

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        metric = _drb(self.kwargs.meta["metric"], metric).lower()
        _curves = self._parse_curves(curves)
        if metric in ["clean_price", "dirty_price", "spread", "ytm"]:
            disc_curve = _validate_obj_not_no_input(
                _maybe_get_curve_maybe_from_solver(
                    solver=solver,
                    curves=_curves,
                    curves_meta=self.kwargs.meta["curves"],
                    name="disc_curve",
                ),
                "disc_curve",
            )
            rate_curve = _maybe_get_curve_or_dict_maybe_from_solver(
                solver=solver,
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                name="rate_curve",
            )
            settlement_ = self._maybe_get_settlement(settlement, disc_curve)

            if metric == "spread":
                _: DualTypes = self.leg1.spread(
                    # target_npv=-(npv + self.leg1.settlement_params.notional),
                    target_npv=-(self.leg1.settlement_params.notional),
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    settlement=settlement_,
                    forward=settlement_,
                )
                return _
            else:
                npv = self.leg1.local_npv(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    settlement=settlement_,
                    forward=settlement_,
                )
                # scale price to par 100 (npv is already projected forward to settlement)
                dirty_price = npv * 100 / -self.leg1.settlement_params.notional

                if metric == "dirty_price":
                    return dirty_price
                elif metric == "clean_price":
                    return dirty_price - self.accrued(settlement_, rate_curve=rate_curve)
                elif metric == "ytm":
                    return self.ytm(
                        price=dirty_price, settlement=settlement_, dirty=True, rate_curve=rate_curve
                    )

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'spread', 'ytm'}.")

    def accrued(
        self,
        settlement: datetime,
        rate_curve: CurveOption_ = NoInput(0),
    ) -> DualTypes:
        acc_idx = self.leg1._period_index(settlement)
        if self.leg1.rate_params.fixing_method == FloatFixingMethod.IBOR:
            frac = self.kwargs.meta["calc_mode"]._settle_accrual(self, settlement, acc_idx)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period
            rate = self.leg1._regular_periods[acc_idx].rate(rate_curve)
            cashflow = (
                -self.leg1._regular_periods[acc_idx].settlement_params.notional
                * self.leg1._regular_periods[acc_idx].period_params.dcf
                * rate
                / 100.0
            )
            return frac * cashflow / -self.leg1.settlement_params.notional * 100.0  # type: ignore[no-any-return]

        else:  # is "rfr"
            p = FloatPeriod(
                start=self.leg1.schedule.aschedule[acc_idx],
                end=settlement,
                payment=settlement,
                termination=self.leg1.schedule.aschedule[acc_idx + 1],
                stub=True,
                frequency=self.leg1.schedule.frequency_obj,
                notional=-100,
                currency=self.leg1.settlement_params.currency,
                convention=self.leg1._regular_periods[acc_idx].period_params.convention,
                float_spread=self.float_spread,
                fixing_method=self.leg1.rate_params.fixing_method,
                rate_fixings=self.leg1.rate_params.fixing_identifier,
                method_param=self.leg1.rate_params.method_param,
                spread_compound_method=self.leg1.rate_params.spread_compound_method,
                fixing_series=self.leg1.rate_params.fixing_series,
                fixing_frequency=self.leg1.rate_params.fixing_frequency,
                # roll=self.leg1.schedule.roll,
                calendar=self.leg1.schedule.calendar,
                adjuster=self.leg1.schedule.accrual_adjuster,
            )
            if p.period_params.start == p.period_params.end and acc_idx == 0:
                # bond settlement on issue date so there is no accrued
                return 0.0

            is_ex_div = self.ex_div(settlement)
            if is_ex_div and settlement == self.leg1._regular_periods[acc_idx].period_params.end:
                # then settlement is on a coupon date so no accrued
                return 0.0

            rate_to_settle = p.rate(rate_curve)
            accrued_to_settle = 100.0 * p.period_params.dcf * rate_to_settle / 100.0

            if is_ex_div:
                rate_to_end = self.leg1._regular_periods[acc_idx].rate(rate_curve=rate_curve)
                accrued_to_end = (
                    100.0
                    * self.leg1._regular_periods[acc_idx].period_params.dcf
                    * rate_to_end
                    / 100.0
                )
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle
