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
from rateslib.periods.parameters import _SettlementParams
from rateslib.periods.utils import (
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (
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


class _WithNPV(Protocol):
    """
    Protocol to establish value of any *Leg* type.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithNPV.spread

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithNPV.local_npv
       ~_WithNPV.npv

    """

    @property
    def periods(self) -> Sequence[_BasePeriod]:
        ...
        # """List of *Periods* associated with the *Leg*."""
        # return self._periods

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` of the
        first *Period* of the *Leg*."""
        return self.periods[0].settlement_params

    def local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the NPV of the *Leg* expressed in local settlement currency.

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
        # a Leg only has cashflows in one single currency, so some up those values first
        # then format for necessary dict output if required.
        local_npv: DualTypes = sum(
            _.local_npv(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            )
            for _ in self.periods
        )
        return local_npv

    def npv(
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
        Calculate the NPV of the *Period* converted to any other *base* accounting currency.

        .. hint::

           If the cashflows are unspecified or incalculable due to missing information this method
           will raise an exception. For a function that returns a `Result` indicating success or
           failure use :meth:`~rateslib.periods._WithNPV.try_local_npv`.

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
        :meth:`~rateslib.periods._WithNPV.try_local_npv`.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        # a Leg only has cashflows in one single currency, so some up those values first
        # then format for necessary dict output if required.
        local_npv: DualTypes = sum(
            _.local_npv(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            )
            for _ in self.periods
        )
        return _maybe_local(
            value=local_npv,
            local=local,
            currency=self.settlement_params.currency,
            fx=fx,
            base=base,
            forward=forward,
        )

    def spread(
        self,
        *,
        target_npv: DualTypes,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate a spread metric which when applied to the *Leg* allows it to attain the target
        value.

        Parameters
        ----------
        target_npv: DualTypes, required
            The target value of the *Leg* measured using all of the other given arguments.
            Must be expressed in local settlement currency units.
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
        raise NotImplementedError(f"Method: `spread` is not available for {type(self).__name__}.")
