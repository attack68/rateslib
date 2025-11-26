from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.fx import FXForwards, FXRates, forward_fx
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_fx_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
)
from rateslib.legs.components import CustomLeg
from rateslib.periods.components import Cashflow

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption_,
        Solver_,
        _BaseLeg,
        datetime,
        datetime_,
        str_,
    )


class FXForward(_BaseInstrument):
    """
    Create a simple *FX exchange* composing two
    :class:`~rateslib.legs.components.CustomLeg`
    of individual :class:`~rateslib.periods.components.Cashflow` of different currencies.

    .. rubric:: Examples

    A sold EURUSD *FX forward* at 1.165 expressed in $10mm.

    .. ipython:: python
       :suppress:

       from datetime import datetime as dt
       from rateslib.instruments.components import FXForward

    .. ipython:: python

       fxfwd = FXForward(
           settlement=dt(2022, 2, 24),
           pair="eurusd",
           leg2_notional=10e6,
           fx_rate=1.165
       )
       fxfwd.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    settlement : datetime, :red:`required`
        The date of the currency exchange.
    pair: str, :red:`required`
        The currency pair of the exchange, e.g. "eurusd", using 3-digit iso codes.
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        To define the notional of the trade in units of LHS pair use ``notional``.
    leg2_notional : float, Dual, Dual2, Variable, :green:`optional (negatively inherited from leg1)`
        To define the notional of the trade in units of RHS pair use ``leg2_notional``.
        Only one of ``notional`` or ``leg2_notional`` can be specified.
    fx_rate : float, :green:`optional`
        The FX rate of ``pair`` defining the transaction price. If not given, set at pricing.
    curves : Curve, LineCurve, str or list of such, :green:`optional`
        For *FXExchange* only discounting curves are required in each currency and not rate
        forecasting curves.
        The signature should be: `[None, eur_curve, None, usd_curve]` for a "eurusd" pair.
    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.components.CustomLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.components.CustomLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An FXExchange requires 2 curves a separate discount curve for each currency

        When given as 2 elements the first is treated as the rate curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
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
            if len(curves) == 3:
                return _Curves(
                    disc_curve=curves[1],
                    index_curve=curves[0],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 2:
                return _Curves(
                    disc_curve=curves[1],
                    index_curve=curves[0],
                    leg2_rate_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 4:
                return _Curves(
                    disc_curve=curves[1],
                    index_curve=curves[0],
                    leg2_rate_curve=curves[2],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 2 curve types. Got {len(curves)}."
                )

        else:  # `curves` is just a single input which is copied across all curves
            raise ValueError(f"{type(self).__name__} requires 2 curve types. Got 1.")

    def __init__(
        self,
        settlement: datetime,
        pair: str,
        fx_rate: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        leg2_notional: DualTypes_ = NoInput(0),
        curves: Curves_ = NoInput(0),
    ):
        pair_ = pair.lower()
        if isinstance(notional, NoInput) and isinstance(leg2_notional, NoInput):
            notional = defaults.notional
        elif not isinstance(notional, NoInput) and not isinstance(leg2_notional, NoInput):
            raise ValueError("Only one of `notional` and `leg2_notional` can be given.")

        user_args = dict(
            settlement=settlement,
            currency=pair[:3],
            leg2_currency=pair[3:6],
            notional=notional,
            leg2_notional=leg2_notional,
            curves=self._parse_curves(curves),
        )
        instrument_args = dict(
            leg2_settlement=NoInput.inherit,
            pair=NoInput(0),
            leg2_pair=NoInput(0),
            fx_fixings=NoInput(0),
            leg2_fx_fixings=NoInput(0),
        )  # these are hard coded arguments specific to this instrument
        default_args = dict(
            notional=defaults.notional,
        )
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves"],
        )

        # allocate arguments to correct legs for non-deliverability
        if isinstance(notional, NoInput):
            self.kwargs.leg1["notional"] = -1.0 * self.kwargs.leg2["notional"]
            self.kwargs.leg1["pair"] = pair_
            self.kwargs.leg1["fx_fixings"] = fx_rate
        else:  # notional set on leg1
            self.kwargs.leg2["notional"] = -1.0 * self.kwargs.leg1["notional"]
            self.kwargs.leg2["pair"] = pair_
            self.kwargs.leg2["fx_fixings"] = fx_rate

        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=-1.0 * self.kwargs.leg1["notional"],
                    payment=self.kwargs.leg1["settlement"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"],
                ),
            ]
        )
        self._leg2 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=-1.0 * self.kwargs.leg2["notional"],
                    payment=self.kwargs.leg2["settlement"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"],
                )
            ]
        )
        self._legs = [self._leg1, self._leg2]

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
        fx_ = _get_fx_maybe_from_solver(solver=solver, fx=fx)
        if isinstance(fx_, FXRates | FXForwards):
            imm_fx: DualTypes = fx_.rate(self.kwargs.leg2["pair"])
        elif isinstance(fx_, NoInput):
            raise ValueError(
                "`fx` must be supplied to price FXExchange object.\n"
                "Note: it can be attached to, and then fetched from, a Solver.",
            )
        else:
            imm_fx = fx_

        _: DualTypes = forward_fx(
            date=self.kwargs.leg1["settlement"],
            curve_domestic=_maybe_get_curve_or_dict_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "disc_curve", solver
            ),
            curve_foreign=_maybe_get_curve_or_dict_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "leg2_disc_curve", solver
            ),
            fx_rate=imm_fx,
        )
        return _
