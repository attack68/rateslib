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

from rateslib import defaults
from rateslib.curves._parsers import (
    _try_disc_required_maybe_from_curve,
)
from rateslib.dual.utils import _dual_float, _float_or_none
from rateslib.enums.generics import Err, NoInput
from rateslib.periods.parameters import (
    _CreditParams,
    _FixedRateParams,
    _FloatRateParams,
    _IndexParams,
    _MtmParams,
    _NonDeliverableParams,
    _PeriodParams,
)
from rateslib.periods.protocols.npv import _WithNPV, _WithNPVStatic
from rateslib.periods.utils import (
    _get_immediate_fx_scalar_and_base,
    _try_validate_base_curve,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        FXForwards_,
        Result,
        _BaseCurve_,
        _FXVolOption_,
        datetime_,
        str_,
    )


class _WithCashflows(_WithNPV, Protocol):
    """
    Protocol for parameter and calculation display for the *Period*.

    .. warning::

       The direct methods of this class are for display convenience.
       Calling these to extract certain values should be avoided. It is more efficient to
       source relevant parameters or calculations from object attributes or other methods directly.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithCashflows.try_cashflow

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithCashflows.cashflows

    """

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the cashflow for the *Period* with any non-deliverable currency adjustment
        **and** indexation.

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

        Returns
        -------
        Result of float, Dual, Dual2, Variable
        """
        return Err(
            NotImplementedError(
                f"`cashflow` is not explicitly implemented for period type: {type(self).__name__}"
            )
        )

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficient to source relevant parameters or calculations
           from object attributes or other methods directly.

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
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        dict of values
        """
        standard_elements = _standard_elements(self=self)
        period_elements = _period_elements(self=self)
        cashflow_elements = _cashflow_elements(
            self=self,
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            forward=forward,
            settlement=settlement,
        )
        rate_elements = _rate_elements(self=self, rate_curve=rate_curve)
        credit_elements = _credit_elements(self=self, rate_curve=rate_curve)
        return {
            **standard_elements,
            **period_elements,
            **rate_elements,
            **cashflow_elements,
            **credit_elements,
        }


class _WithCashflowsStatic(_WithNPVStatic, Protocol):
    """
    Protocol for parameter and calculation display for the *Static Period*.

    .. warning::

       The direct methods of this class are for display convenience.
       Calling these to extract certain values should be avoided. It is more efficient to
       source relevant parameters or calculations from object attributes or other methods directly.

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithCashflowsStatic.cashflows

    """

    def _index_elements(
        self,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> dict[str, Any]:
        # indexing parameters
        index_elements: dict[str, Any] = {}
        if hasattr(self, "index_params") and isinstance(self.index_params, _IndexParams):
            assert isinstance(self.index_params, _IndexParams)  # noqa: S101
            iv = self.index_params.try_index_value(index_curve=index_curve)
            ib = self.index_params.try_index_base(index_curve=index_curve)
            if not isinstance(iv, Err) and not isinstance(ib, Err):
                ir = iv.unwrap() / ib.unwrap()
            else:
                ir = None

            uc = self.try_unindexed_cashflow(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                fx=fx,
                fx_vol=fx_vol,
            )

            index_elements = {
                defaults.headers["index_base"]: _float_or_none(ib),
                defaults.headers["index_value"]: _float_or_none(iv),
                defaults.headers["index_ratio"]: _float_or_none(ir),
                defaults.headers["index_fix_date"]: self.index_params.index_fixing.date,
                defaults.headers["unindexed_cashflow"]: _float_or_none(uc),
            }
        return index_elements

    def _non_deliverable_elements(self, fx: FXForwards_) -> dict[str, Any]:
        # non-deliverable parameters
        non_deliverable_elements: dict[str, Any] = {}
        if hasattr(self, "non_deliverable_params") and isinstance(
            self.non_deliverable_params, _NonDeliverableParams
        ):
            fx_fixing_res: Result[DualTypes] = (
                self.non_deliverable_params.fx_fixing.try_value_or_forecast(fx)
            )
            non_deliverable_elements.update(
                {
                    defaults.headers["fx_fixing"]: _float_or_none(fx_fixing_res),
                    defaults.headers["fx_fixing_date"]: self.non_deliverable_params.fx_fixing.date,
                    defaults.headers[
                        "reference_currency"
                    ]: self.non_deliverable_params.reference_currency.upper(),
                }
            )
        return non_deliverable_elements

    def _mtm_elements(self, fx: FXForwards_) -> dict[str, Any]:
        mtm_elements: dict[str, Any] = {}
        if hasattr(self, "mtm_params") and isinstance(self.mtm_params, _MtmParams):
            # mtm_elements overwrite non_deliverable elements as these are exclusive params.
            fx_fixing_res = self.mtm_params.fx_fixing_end.try_value_or_forecast(fx)
            mtm_elements = {
                defaults.headers["fx_fixing"]: _float_or_none(fx_fixing_res),
                defaults.headers["fx_fixing_date"]: self.mtm_params.fx_fixing_end.date,
                defaults.headers["reference_currency"]: self.mtm_params.reference_currency.upper(),
            }
        return mtm_elements

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficient to source relevant parameters or calculations
           from object attributes or other methods directly.

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
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        dict of values
        """
        standard_elements = _standard_elements(self=self)
        period_elements = _period_elements(self=self)
        cashflow_elements = _cashflow_elements(
            self=self,
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
            forward=forward,
            settlement=settlement,
        )
        rate_elements = _rate_elements(self=self, rate_curve=rate_curve)
        credit_elements = _credit_elements(self=self, rate_curve=rate_curve)
        index_elements = self._index_elements(index_curve=index_curve)
        non_deliverable_elements = self._non_deliverable_elements(fx=fx)
        mtm_elements = self._mtm_elements(fx=fx)
        return {
            **standard_elements,
            **period_elements,
            **cashflow_elements,
            **rate_elements,
            **credit_elements,
            **index_elements,
            **non_deliverable_elements,
            **mtm_elements,
        }


def _standard_elements(self: _WithCashflows | _WithCashflowsStatic) -> dict[str, Any]:
    """Typical cashflow attributes for any constructed *Period*"""
    # standard parameters
    standard_elements: dict[str, Any] = {}
    standard_elements.update(
        {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["currency"]: self.settlement_params.currency.upper(),
            defaults.headers["payment"]: self.settlement_params.payment,
            defaults.headers["notional"]: _dual_float(self.settlement_params.notional),
        }
    )
    return standard_elements


def _period_elements(self: _WithCashflows | _WithCashflowsStatic) -> dict[str, Any]:
    """
    Typical date-like attributes for any constructed *Period* with `period_params`.
    """
    # period parameters
    period_elements: dict[str, Any] = {}
    if hasattr(self, "period_params") and isinstance(self.period_params, _PeriodParams):
        period_elements.update(
            {
                defaults.headers["stub_type"]: "Stub" if self.period_params.stub else "Regular",
                defaults.headers["convention"]: str(self.period_params.convention),
                defaults.headers["dcf"]: self.period_params.dcf,
                defaults.headers["a_acc_start"]: self.period_params.start,
                defaults.headers["a_acc_end"]: self.period_params.end,
            }
        )
    return period_elements


def _rate_elements(
    self: _WithCashflows | _WithCashflowsStatic,
    rate_curve: CurveOption_,
) -> dict[str, Any]:
    """
    Typical rate-like attributes for any constructed *Period* with `rate_params`.
    """
    # rate parameters
    rate_elements: dict[str, Any] = {}
    if hasattr(self, "rate_params"):
        if isinstance(self.rate_params, _FixedRateParams):
            rate_elements.update(
                {
                    defaults.headers["rate"]: _float_or_none(self.rate_params.fixed_rate),
                    defaults.headers["spread"]: None,
                }
            )
        elif isinstance(self.rate_params, _FloatRateParams):
            rate_elements.update(
                {
                    # try_rate is guaranteed by having FloatRateParams but this is poor typing.
                    defaults.headers["rate"]: _float_or_none(self.try_rate(rate_curve=rate_curve)),  # type: ignore[attr-defined]
                    defaults.headers["spread"]: _float_or_none(self.rate_params.float_spread),
                }
            )
    return rate_elements


def _credit_elements(
    self: _WithCashflows | _WithCashflowsStatic,
    rate_curve: CurveOption_,
) -> dict[str, Any]:
    """
    Typical credit-like attributes for any constructed *Period* with `credit_params`.
    """
    credit_elements: dict[str, Any] = {}
    if hasattr(self, "credit_params") and isinstance(self.credit_params, _CreditParams):
        if hasattr(self, "period_params") and isinstance(self.period_params, _PeriodParams):
            rc_res = _try_validate_base_curve(rate_curve)
            if not isinstance(rc_res, Err):
                credit_elements.update(
                    {
                        defaults.headers["survival"]: _dual_float(
                            rc_res.unwrap()[self.period_params.end]
                        ),
                        defaults.headers["recovery"]: _dual_float(
                            rc_res.unwrap().meta.credit_recovery_rate
                        ),
                    }
                )
            else:
                credit_elements.update(
                    {defaults.headers["survival"]: None, defaults.headers["recovery"]: None}
                )
        else:
            pass
    return credit_elements


def _cashflow_elements(
    self: _WithCashflows | _WithCashflowsStatic,
    *,
    rate_curve: CurveOption_ = NoInput(0),
    disc_curve: _BaseCurve_ = NoInput(0),
    index_curve: _BaseCurve_ = NoInput(0),
    fx: FXForwards_ = NoInput(0),
    fx_vol: _FXVolOption_ = NoInput(0),
    base: str_ = NoInput(0),
    settlement: datetime_ = NoInput(0),
    forward: datetime_ = NoInput(0),
) -> dict[str, Any]:
    # cashflow valuation based parameters
    c = self.try_cashflow(
        rate_curve=rate_curve,
        disc_curve=disc_curve,
        index_curve=index_curve,
        fx=fx,
        fx_vol=fx_vol,
    )

    disc_curve_result = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
    if disc_curve_result.is_err:
        # then NPV is impossible
        v, collateral = None, None
    else:
        v = disc_curve_result.unwrap()[self.settlement_params.payment]
        collateral = disc_curve_result.unwrap().meta.collateral

    # Since `cashflows` in not a performance critical function this call duplicates
    # cashflow calculations. A more efficient calculation is possible but the code branching
    # is ugly.
    local_npv_result = self.try_local_npv(
        rate_curve=rate_curve,
        index_curve=index_curve,
        disc_curve=disc_curve,
        fx=fx,
        fx_vol=fx_vol,
        settlement=settlement,
        forward=forward,
    )

    fx_, base_ = _get_immediate_fx_scalar_and_base(self.settlement_params.currency, fx, base)
    if local_npv_result.is_err:
        npv_fx = None
    else:
        npv_fx = local_npv_result.unwrap() * fx_

    return {
        defaults.headers["df"]: _float_or_none(v),
        defaults.headers["cashflow"]: _float_or_none(c),
        defaults.headers["npv"]: _float_or_none(local_npv_result),
        defaults.headers["fx"]: _dual_float(fx_),
        defaults.headers["base"]: base_.upper(),
        defaults.headers["npv_fx"]: _float_or_none(npv_fx),
        defaults.headers["collateral"]: collateral,
    }
