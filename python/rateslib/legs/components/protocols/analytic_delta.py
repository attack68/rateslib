from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        FXRevised_,
        Period,
        _BaseCurve_,
        str_,
    )


class _WithAnalyticDelta(Protocol):
    periods: list[Period]

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

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        return sum(
            _.analytic_delta(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                base=base,
            )
            for _ in self.periods
        )
