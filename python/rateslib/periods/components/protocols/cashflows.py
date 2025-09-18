from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib import defaults
from rateslib.curves._parsers import (
    _try_disc_required_maybe_from_curve,
)
from rateslib.dual.utils import _dual_float, _float_or_none
from rateslib.enums.generics import Err, NoInput
from rateslib.periods.components.parameters import (
    _CashflowRateParams,
    _FixedRateParams,
    _FloatRateParams,
    _IndexParams,
    _NonDeliverableParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.parameters.fx_volatility import _FXOptionParams
from rateslib.periods.components.protocols.npv import _WithNPV, _WithNPVStatic
from rateslib.periods.components.utils import (
    _get_immediate_fx_scalar_and_base,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        CurveOption_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Result,
        _BaseCurve_,
        datetime_,
        str_,
    )


class _WithNPVCashflows(_WithNPV, Protocol):
    settlement_params: _SettlementParams
    rate_params: _FixedRateParams | _CashflowRateParams | _FloatRateParams | None
    non_deliverable_params: None
    index_params: None
    period_params: _PeriodParams | None
    fx_option_params: _FXOptionParams | None

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the cashflow for the *Period* with settlement currency adjustment
        **and** indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary.

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
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficent to source relevant parameters or calculations
           from object attributes or other methods directly.

        Parameters
        ----------
        XXX

        Returns
        -------
        dict of values
        """

        # standard parameters
        standard_elements = {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["currency"]: self.settlement_params.currency.upper(),
            defaults.headers["payment"]: self.settlement_params.payment,
            defaults.headers["notional"]: _dual_float(self.settlement_params.notional),
        }
        # period parameters
        if self.period_params is not None:
            standard_elements[defaults.headers["stub_type"]] = (
                "Stub" if self.period_params.stub else "Regular"
            )
            period_elements = {
                defaults.headers["convention"]: str(self.period_params.convention),
                defaults.headers["dcf"]: self.period_params.dcf,
                defaults.headers["a_acc_start"]: self.period_params.start,
                defaults.headers["a_acc_end"]: self.period_params.end,
            }
        else:
            period_elements = {}

        # cashflow valuation based parameters
        c = self.try_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )

        disc_curve_result = _try_disc_required_maybe_from_curve(
            curve=rate_curve, disc_curve=disc_curve
        )
        if disc_curve_result.is_err:
            # then NPV is impossible
            v, collateral = None, None
        else:
            v = disc_curve_result.unwrap()[self.settlement_params.payment]
            collateral = disc_curve_result.unwrap().meta.collateral

        # Since `cashflows` in not a performance critical function this call duplicates
        # cashflow calculations. A more efficient calculation is possible but the code branching
        # in ugly.
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

        cashflow_elements = {
            defaults.headers["df"]: _float_or_none(v),
            defaults.headers["cashflow"]: _float_or_none(c),
            defaults.headers["npv"]: _float_or_none(local_npv_result),
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["base"]: base_.upper(),
            defaults.headers["npv_fx"]: _float_or_none(npv_fx),
            defaults.headers["collateral"]: collateral,
        }

        # rate parameters
        if isinstance(self.rate_params, _FixedRateParams):
            rate_elements = {
                defaults.headers["rate"]: _float_or_none(self.rate_params.fixed_rate),
                defaults.headers["spread"]: None,
            }
        elif isinstance(self.rate_params, _FloatRateParams):
            rate_elements = {
                # try_rate is guaranteed by having FloatRateParams but this is poor typing.
                defaults.headers["rate"]: _float_or_none(self.try_rate(rate_curve=rate_curve)),  # type: ignore[attr-defined]
                defaults.headers["spread"]: _float_or_none(self.rate_params.float_spread),
            }
        else:
            rate_elements = {}

        return {
            **standard_elements,
            **period_elements,
            **rate_elements,
            **cashflow_elements,
        }


class _WithNPVCashflowsStatic(_WithNPVStatic, Protocol):
    settlement_params: _SettlementParams
    non_deliverable_params: _NonDeliverableParams | None
    index_params: _IndexParams | None
    period_params: _PeriodParams | None
    rate_params: _FixedRateParams | _CashflowRateParams | _FloatRateParams | None
    fx_option_params: _FXOptionParams | None

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficent to source relevant parameters or calculations
           from object attributes or other methods directly.

        Parameters
        ----------
        XXX

        Returns
        -------
        dict of values
        """

        # standard parameters
        standard_elements = {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["currency"]: self.settlement_params.currency.upper(),
            defaults.headers["payment"]: self.settlement_params.payment,
            defaults.headers["notional"]: _dual_float(self.settlement_params.notional),
        }
        if self.period_params is not None:
            standard_elements[defaults.headers["stub_type"]] = (
                "Stub" if self.period_params.stub else "Regular"
            )

        # indexing parameters
        if self.is_indexed:
            assert isinstance(self.index_params, _IndexParams)  # noqa: S101
            i = self.index_params.try_index_ratio(index_curve)
            if i.is_err:
                ir, iv, ib = (
                    None,
                    None,
                    None,
                )
            else:
                ir, iv, ib = i.unwrap()
            index_elements = {
                defaults.headers["index_base"]: _float_or_none(ib),
                defaults.headers["index_value"]: _float_or_none(iv),
                defaults.headers["index_ratio"]: _float_or_none(ir),
            }
        else:
            index_elements = {}

        # period parameters
        if self.period_params is None:
            period_elements = {}
        else:
            period_elements = {
                defaults.headers["convention"]: str(self.period_params.convention),
                defaults.headers["dcf"]: self.period_params.dcf,
                defaults.headers["a_acc_start"]: self.period_params.start,
                defaults.headers["a_acc_end"]: self.period_params.end,
            }

        # non-deliverable parameters
        if self.non_deliverable_params is not None:
            fx_fixing_res: Result[DualTypes] = self.non_deliverable_params.try_fx_fixing(fx)
            currency_elements = {
                defaults.headers["fx_fixing"]: _float_or_none(fx_fixing_res),
                defaults.headers[
                    "reference_currency"
                ]: self.non_deliverable_params.reference_currency.upper(),
            }
        else:
            currency_elements = {}

        # cashflow valuation based parameters
        uc = self.try_unindexed_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        c = self.try_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )

        disc_curve_result = _try_disc_required_maybe_from_curve(
            curve=rate_curve, disc_curve=disc_curve
        )
        if disc_curve_result.is_err:
            # then NPV is impossible
            v, collateral = None, None
        else:
            v = disc_curve_result.unwrap()[self.settlement_params.payment]
            collateral = disc_curve_result.unwrap().meta.collateral

        # Since `cashflows` in not a performance critical function this call duplicates
        # cashflow calculations. A more efficient calculation is possible but the code branching
        # in ugly.
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

        cashflow_elements = {
            defaults.headers["df"]: _float_or_none(v),
            defaults.headers["cashflow"]: _float_or_none(c),
            defaults.headers["npv"]: _float_or_none(local_npv_result),
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["base"]: base_.upper(),
            defaults.headers["npv_fx"]: _float_or_none(npv_fx),
            defaults.headers["collateral"]: collateral,
        }
        if self.is_indexed:
            cashflow_elements[defaults.headers["unindexed_cashflow"]] = _float_or_none(uc)

        # rate parameters
        if isinstance(self.rate_params, _FixedRateParams):
            rate_elements = {
                defaults.headers["rate"]: _float_or_none(self.rate_params.fixed_rate),
                defaults.headers["spread"]: None,
            }
        elif isinstance(self.rate_params, _FloatRateParams):
            rate_elements = {
                # try_rate is guaranteed by having FloatRateParams but this is poor typing.
                defaults.headers["rate"]: _float_or_none(self.try_rate(rate_curve=rate_curve)),  # type: ignore[attr-defined]
                defaults.headers["spread"]: _float_or_none(self.rate_params.float_spread),
            }
        else:
            rate_elements = {}

        return {
            **standard_elements,
            **currency_elements,
            **period_elements,
            **index_elements,
            **rate_elements,
            **cashflow_elements,
        }
