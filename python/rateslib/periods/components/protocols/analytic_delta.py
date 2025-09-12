from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput, Ok
from rateslib.periods.components.utils import (
    _maybe_fx_converted,
)
from rateslib.periods.components.protocols.npv import _WithIndexingStatic, _WithNonDeliverableStatic

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        DualTypes,
        FXForwards_,
        FXRevised_,
        Result,
        _BaseCurve_,
        str_,
    )


class _WithAnalyticDeltaStatic(_WithIndexingStatic, _WithNonDeliverableStatic, Protocol):
    def try_unindexed_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in ``reference_currency``
        without indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        pass

    def try_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in ``reference_currency``
        with indexation.

        """
        rrad = self.try_unindexed_reference_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve
        )
        return self._maybe_index_up(value=rrad, index_curve=index_curve)

    def try_unindexed_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> Result[DualTypes]:
        rrad = self.try_unindexed_reference_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve
        )
        return self._maybe_convert_deliverable(value=rrad, fx=fx)

    def try_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in a base currency.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        base: str, optional
            The currency to return the result in. If not given is set to the *local settlement*
            ``currency``.
        local: bool, optional
            An override flag to return a dict of NPV values indexed by string currency.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        rad = self.try_reference_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve
        )
        lad = self._maybe_convert_deliverable(value=rad, fx=fx)  # type: ignore[arg-type]
        if lad.is_err:
            return lad
        return Ok(
            _maybe_fx_converted(
                value=lad.unwrap(), currency=self.settlement_params.currency, fx=fx, base=base
            )
        )

    def analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the analytic rate delta of a *Period* expressed in a base currency.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        base: str, optional
            The currency to return the result in. If not given is set to the *local settlement*
            ``currency``.
        local: bool, optional
            An override flag to return a dict of NPV values indexed by string currency.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        return self.try_analytic_delta(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            base=base,
        ).unwrap()