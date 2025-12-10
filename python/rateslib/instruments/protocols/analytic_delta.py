from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput
from rateslib.instruments.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _Vol,
    _WithPricingObjs,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Sequence,
        Solver_,
        VolT_,
        _BaseLeg,
        _Curves,
        _KWArgs,
        datetime_,
        str_,
    )


class _WithAnalyticDelta(_WithPricingObjs, Protocol):
    """
    Protocol to determine the *analytic rate delta* of a particular *Leg* of an *Instrument*.
    """

    @property
    def kwargs(self) -> _KWArgs: ...

    @property
    def legs(self) -> Sequence[_BaseLeg]: ...

    def analytic_delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        TBD
        """
        _curves: _Curves = self._parse_curves(curves)
        _vol: _Vol = self._parse_vol(vol)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _vol_meta: _Vol = self.kwargs.meta["vol"]

        prefix = "" if leg == 1 else "leg2_"

        value: DualTypes | dict[str, DualTypes] = self.legs[leg - 1].analytic_delta(
            rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}rate_curve", solver
            ),
            disc_curve=_maybe_get_curve_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}disc_curve", solver
            ),
            index_curve=_maybe_get_curve_maybe_from_solver(
                _curves_meta, _curves, f"{prefix}index_curve", solver
            ),
            fx_vol=_maybe_get_fx_vol_maybe_from_solver(_vol_meta, _vol, solver),
            fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver),
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )
        return value
