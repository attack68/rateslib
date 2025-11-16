from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _get_fx_maybe_from_solver,
    _get_maybe_curve_maybe_from_solver,
    _WithPricingObjs,
)
from rateslib.periods.components.utils import _maybe_fx_converted

if TYPE_CHECKING:
    from rateslib.typing import (
        Curves_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Solver_,
        _Curves,
        datetime_,
        str_,
    )


class _WithNPV(_WithPricingObjs, Protocol):
    """
    Protocol to establish value of any *Instrument* type.
    """

    _kwargs: _KWArgs

    @property
    def kwargs(self) -> _KWArgs:
        """The :class:`~rateslib.instruments.components.protocols._KWArgs` container for
        the *Instrument*."""
        return self._kwargs

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

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
        """
        Calculate the NPV of the *Period* converted to any other *base* accounting currency.

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
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "legs")  # noqa: S101

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _fx_maybe_from_solver = _get_fx_maybe_from_solver(fx=fx, solver=solver)

        local_npv = {
            self.legs[0].settlement_params.currency: self.legs[0].local_npv(
                rate_curve=_get_maybe_curve_maybe_from_solver(
                    _curves_meta, _curves, "rate_curve", solver
                ),
                disc_curve=_get_maybe_curve_maybe_from_solver(
                    _curves_meta, _curves, "disc_curve", solver
                ),
                index_curve=_get_maybe_curve_maybe_from_solver(
                    _curves_meta, _curves, "index_curve", solver
                ),
                fx=_fx_maybe_from_solver,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            )
        }

        leg2_local_npv = self.legs[1].local_npv(
            rate_curve=_get_maybe_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_rate_curve", solver
            ),
            disc_curve=_get_maybe_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_disc_curve", solver
            ),
            index_curve=_get_maybe_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_index_curve", solver
            ),
            fx=_fx_maybe_from_solver,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )

        if self.legs[0].settlement_params.currency == self.legs[1].settlement_params.currency:
            # then the two legs share the same currency
            local_npv[self.legs[0].settlement_params.currency] += leg2_local_npv
        else:
            local_npv[self.legs[1].settlement_params.currency] = leg2_local_npv

        if not local:
            single_value: DualTypes = 0.0
            base_ = _drb(self.legs[0].settlement_params.currency, base)
            for k, v in local_npv.items():
                single_value += _maybe_fx_converted(
                    value=v, currency=k, fx=_fx_maybe_from_solver, base=base_
                )
            return single_value
        else:
            return local_npv

    def _npv_single_core(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, DualTypes]:
        """
        Private NPV summation function used with a single thread, over all `self.instruments`.
        """
        assert hasattr(self, "instruments")  # noqa: S101

        local_npv: dict[str, DualTypes] = {}
        for instrument in self.instruments:
            inst_local_npv = instrument.npv(
                curves=curves,
                solver=solver,
                fx=fx,
                fx_vol=fx_vol,
                base=base,
                local=True,
                settlement=settlement,
                forward=forward,
            )

            for k, v in inst_local_npv.items():
                if k in local_npv:
                    local_npv[k] += v
                else:
                    local_npv[k] = v
        return local_npv
