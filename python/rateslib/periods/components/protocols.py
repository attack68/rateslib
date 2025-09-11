from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves import _BaseCurve
from rateslib.curves._parsers import (
    _disc_maybe_from_curve,
    _try_disc_required_maybe_from_curve,
)
from rateslib.dual.utils import _dual_float, _float_or_none
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.components.parameters import (
    _CashflowRateParams,
    _FixedRateParams,
    _FloatRateParams,
    _IndexParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.utils import (
    _get_immediate_fx_scalar_and_base,
    _maybe_fx_converted,
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        CurveOption_,
        DualTypes,
        FXForwards_,
        FXRevised_,
        Result,
        _BaseCurve_,
        datetime_,
        str_,
    )


class _WithNPV(Protocol):
    settlement_params: _SettlementParams

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def try_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]: ...

    def npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Calculate the NPV of the *Period* converted to any other *base* currency.

        .. hint::

           If the cashflows are unspecified or incalculable due to missing information this method
           will raise an exception. For a function that returns a `Result` indicating success or
           failure use :meth:`~rateslib.periods.components.protocols._WithNPV.try_local_npv`.

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
            The currency to convert the *local settlement* NPV to.
        local: bool, optional
            An override flag to return a dict of NPV values indexed by string currency.

        Returns
        -------
        float, Dual, Dual2, Variable or dict of such indexed by string currency.

        Notes
        -----
        If ``base`` is not provided then this function will return the value obtained from
        :meth:`~rateslib.periods.components.protocols._WithNPV.local_npv`.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        local_npv = self.try_local_npv(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,  # type: ignore[arg-type]
            settlement=settlement,
            forward=forward,
        ).unwrap()
        return _maybe_local(
            value=local_npv, local=local, currency=self.settlement_params.currency, fx=fx, base=base
        )


class _WithIndexingStatic(Protocol):
    index_params: _IndexParams | None

    @property
    def is_indexed(self) -> bool:
        """
        Check whether the *Period* has indexation applied, which means it has associated
        index parameters.
        """
        return self.index_params is not None

    def _maybe_index_up(
        self, value: Result[DualTypes], index_curve: _BaseCurve_
    ) -> Result[DualTypes]:
        """Apply indexation to a value if required."""
        if not self.is_indexed:
            # then no indexation of the cashflow will occur.
            return value
        else:
            if value.is_err:
                return value

            value_: DualTypes = value.unwrap()
            assert isinstance(self.index_params, _IndexParams)  # noqa: S101
            i = self.index_params.try_index_ratio(index_curve)
            if i.is_err:
                return i  # type: ignore[return-value]

            i_: DualTypes = i.unwrap()[0]
            if self.index_params.index_only:
                return Ok(value_ * (i_ - 1))
            else:
                return Ok(value_ * i_)


class _WithNonDeliverableStatic(Protocol):
    settlement_params: _SettlementParams

    @property
    def is_non_deliverable(self) -> bool:
        """
        Check whether the *Period* is non-deliverable,
        which means it has a separate ``currency`` to the ``reference_currency``.
        """
        return self.settlement_params.pair is not None

    def _maybe_convert_deliverable(
        self, value: Result[DualTypes], fx: FXForwards_
    ) -> Result[DualTypes]:
        """Convert a value in reference currency to settlement currency if required."""
        if not self.is_non_deliverable:
            # then cashflow is directly deliverable
            return value
        else:
            if value.is_err:
                return value

            value_: DualTypes = value.unwrap()
            fx_fix_res = self.settlement_params.try_fx_fixing(fx)
            if fx_fix_res.is_err:
                return fx_fix_res
            else:
                fx_fix = fx_fix_res.unwrap()
            c = value_ * (fx_fix if not self.settlement_params.fx_reversed else (1.0 / fx_fix))
            return Ok(c)


class _WithNPVStatic(_WithNPV, _WithIndexingStatic, _WithNonDeliverableStatic, Protocol):
    settlement_params: _SettlementParams

    def try_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        disc_curve = _disc_maybe_from_curve(rate_curve, disc_curve)

        # settlement ex-div indicator function
        if isinstance(settlement, NoInput):
            if not isinstance(disc_curve, NoInput):
                settlement_: datetime = disc_curve.nodes.initial
            else:
                # settlement is assumed to always be prior to ex-dividend
                settlement_ = datetime(1, 1, 1)
        else:
            settlement_ = settlement
        if settlement_ > self.settlement_params.ex_dividend:
            return Ok(0.0)

        # payment in the past indicator function
        if (
            not isinstance(disc_curve, NoInput)
            and self.settlement_params.payment < disc_curve.nodes.initial
        ):
            return Ok(0.0)  # payment date is in the past

        c = self.try_cashflow(rate_curve=rate_curve, index_curve=index_curve, fx=fx)
        if c.is_err:
            return c
        c_: DualTypes = c.unwrap()

        disc_curve_result = _try_disc_required_maybe_from_curve(
            curve=rate_curve, disc_curve=disc_curve
        )
        if disc_curve_result.is_err:
            return disc_curve_result  # type: ignore[return-value]
        disc_curve_: _BaseCurve = disc_curve_result.unwrap()

        vc = c_ * disc_curve_[self.settlement_params.payment]

        if isinstance(forward, NoInput) and isinstance(settlement, NoInput):
            return Ok(vc)  # forward is assumed to be immediate date
        elif isinstance(forward, NoInput):
            forward_ = settlement_
        else:
            forward_ = forward
        return Ok(vc / disc_curve_[forward_])

    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the cashflow for the *Period* before settlement currency and
        indexation adjustments.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.

        Returns
        -------
        Result of float, Dual, Dual2, Variable
        """
        pass

    def try_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the cashflow for the *Period* before settlement currency adjustment
        but with indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.

        Returns
        -------
        float, Dual, Dual2, Variable or None
        """
        rrc = self.try_unindexed_reference_cashflow(rate_curve=rate_curve)
        return self._maybe_index_up(value=rrc, index_curve=index_curve)

    def try_unindexed_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the cashflow for the *Period* with settlement currency adjustment
        but without indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary.

        Returns
        -------
        float, Dual, Dual2, Variable or None
        """
        rrc = self.try_unindexed_reference_cashflow(rate_curve=rate_curve)
        return self._maybe_convert_deliverable(value=rrc, fx=fx)

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
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
        rc = self.try_reference_cashflow(rate_curve=rate_curve, index_curve=index_curve)
        return self._maybe_convert_deliverable(value=rc, fx=fx)


class _WithNPVCashflowsStatic(_WithNPVStatic, Protocol):
    settlement_params: _SettlementParams
    index_params: _IndexParams | None
    period_params: _PeriodParams
    rate_params: _FixedRateParams | _CashflowRateParams | _FloatRateParams

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
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
            defaults.headers["stub_type"]: "Stub" if self.period_params.stub else "Regular",
            defaults.headers["currency"]: self.settlement_params.currency.upper(),
            defaults.headers["payment"]: self.settlement_params.payment,
            defaults.headers["notional"]: _dual_float(self.settlement_params.notional),
        }

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
        if isinstance(self.rate_params, _CashflowRateParams):
            period_elements = {}
        else:
            period_elements = {
                defaults.headers["convention"]: str(self.period_params.convention),
                defaults.headers["dcf"]: self.period_params.dcf,
                defaults.headers["a_acc_start"]: self.period_params.start,
                defaults.headers["a_acc_end"]: self.period_params.end,
            }

        # non-deliverable parameters
        if self.is_non_deliverable:
            fx_fixing_res: Result[DualTypes] = self.settlement_params.try_fx_fixing(fx)  # type: ignore[arg-type]  # validated in function
            currency_elements = {
                defaults.headers["fx_fixing"]: _float_or_none(fx_fixing_res),
                defaults.headers[
                    "reference_currency"
                ]: self.settlement_params.reference_currency.upper(),
            }
        else:
            currency_elements = {}

        # cashflow valuation based parameters
        uc = self.try_unindexed_cashflow(rate_curve=rate_curve, fx=fx)  # type: ignore[arg-type]  # validated in function
        c = self.try_cashflow(rate_curve=rate_curve, index_curve=index_curve, fx=fx)  # type: ignore[arg-type]  # validated in function

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
            fx=fx,  # type: ignore[arg-type]
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


class _WithRateFixingsExposureStatic(Protocol):
    def try_unindexed_reference_fixings_exposure(self) -> Result[DataFrame]:
        return Err(TypeError(err.TE_NO_FIXING_EXPOSURE_ON_OBJ.format(type(self).__name__)))
