from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput
from rateslib.instruments.components.protocols.curves import _WithCurves
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.utils import (
    _get_curve_maybe_from_solver,
    _get_fx_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Curves_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Solver_,
        _BaseLeg,
        _Curves,
        datetime_,
        str_,
    )


class _WithAnalyticDelta(_WithCurves, Protocol):
    """
    Protocol to establish value of any *Instrument* type.
    """

    _legs: list[_BaseLeg]
    _kwargs: _KWArgs

    @property
    def legs(self) -> list[_BaseLeg]:
        return self._legs

    @property
    def kwargs(self) -> _KWArgs:
        return self._kwargs

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
        """
        Calculate the rate analytic delta of any *Leg* of the *Instrument* converted to any
        other *base* accounting currency.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        base: str, optional
            The currency to convert the *local settlement* NPV to.
        local: bool, optional
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.
        leg: int, optional
            The leg number, 1 or 2, for which to determine the analytic delta.

        Returns
        -------
        float, Dual, Dual2, Variable or dict of such indexed by string currency.

        Notes
        -----
        If ``base`` is not provided then this function will return the value obtained from
        :meth:`~rateslib.periods.components._WithNPV.try_local_npv`.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]

        prefix = "" if leg == 1 else "leg2_"

        return self.legs[leg - 1].analytic_delta(
            rate_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}rate_curve", solver
            ),
            disc_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}disc_curve", solver
            ),
            index_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}index_curve", solver
            ),
            fx_vol=fx_vol,
            fx=_get_fx_maybe_from_solver(fx, solver),
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )
