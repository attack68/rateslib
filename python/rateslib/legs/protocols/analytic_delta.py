# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput
from rateslib.periods.utils import (
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        FXForwards_,
        Sequence,
        _BaseCurve_,
        _BasePeriod,
        _FXVolOption_,
        datetime_,
        str_,
    )


class _WithAnalyticDelta(Protocol):
    """
    Protocol to calculate analytical rate delta sensitivities of any *Leg* type.

    """

    @property
    def periods(self) -> Sequence[_BasePeriod]: ...

    def local_analytic_delta(
        self,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the analytic rate delta of a *Period* expressed in its local settlement currency.

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
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        local_analytic_delta: DualTypes = sum(
            _.try_local_analytic_delta(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            ).unwrap()
            for _ in self.periods
        )
        return local_analytic_delta

    def analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
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
        float, Dual, Dual2, Variable
        """
        local_analytic_delta: DualTypes = sum(
            _.try_local_analytic_delta(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            ).unwrap()
            for _ in self.periods
        )
        return _maybe_local(
            value=local_analytic_delta,
            local=local,
            currency=self.periods[0].settlement_params.currency,
            fx=fx,
            base=base,
            forward=forward,
        )
