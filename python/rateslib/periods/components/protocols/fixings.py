from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, MultiIndex

from rateslib.curves._parsers import (
    _try_disc_required_maybe_from_curve,
)
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.components.parameters import _SettlementParams
from rateslib.periods.components.protocols import _WithIndexingStatic, _WithNonDeliverableStatic
from rateslib.periods.components.protocols.npv import _screen_ex_div_and_forward

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXForwards_,
        FXVolOption_,
        Result,
        _BaseCurve,
        _BaseCurve_,
        datetime_,
    )


class _WithAnalyticRateFixingsSensitivity(Protocol):
    @property
    def settlement_params(self) -> _SettlementParams: ...

    def try_immediate_rate_fixings_sensitivity(
        self,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())

    def local_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of financial sensitivity to published interest rate fixings,
        expressed in local **settlement currency** of the *Period*.

        If the *Period* has no sensitivity to rates fixings this *DataFrame* is empty.

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
        DataFrame
        """
        rfs = self.try_immediate_rate_fixings_sensitivity(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return _screen_ex_div_and_forward(
            local_value=rfs,  # type: ignore[arg-type]
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            ex_dividend=self.settlement_params.ex_dividend,
            forward=forward,
            settlement=settlement,
        ).unwrap()  # type: ignore[return-value]


class _WithAnalyticRateFixingsSensitivityStatic(
    _WithAnalyticRateFixingsSensitivity, _WithIndexingStatic, _WithNonDeliverableStatic, Protocol
):
    def try_unindexed_reference_cashflow_fixings_sensitivity(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())

    def try_unindexed_cashflow_fixings_sensitivity(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        urcfe = self.try_unindexed_reference_cashflow_fixings_sensitivity(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if self.non_deliverable_params is None:
            return urcfe  # no ND modifications required
        if urcfe.is_err:
            return urcfe

        if urcfe.unwrap().empty:
            return urcfe  # nothing to modify

        nd_scalar = self.try_convert_deliverable(value=Ok(1.0), fx=fx)
        if nd_scalar.is_err:
            return nd_scalar  # type: ignore[return-value]

        d = urcfe.unwrap() * nd_scalar.unwrap()
        c = d.columns
        d.columns = MultiIndex.from_tuples(
            tuples=[
                (c.values[0][0], c.values[0][1], self.settlement_params.currency, c.values[0][3])
            ],
            names=c.names,
        )
        return Ok(d)

    def try_reference_cashflow_fixings_sensitivity(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        urcfe = self.try_unindexed_reference_cashflow_fixings_sensitivity(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if urcfe.is_err:
            return urcfe
        index_scalar = self.try_index_up(value=Ok(1.0), index_curve=index_curve)
        if index_scalar.is_err:
            return index_scalar  # type: ignore[return-value]
        return Ok(urcfe.unwrap() * index_scalar.unwrap())

    def try_cashflow_fixings_sensitivity(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        ucfe = self.try_unindexed_cashflow_fixings_sensitivity(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if ucfe.is_err:
            return ucfe
        index_scalar = self.try_index_up(value=Ok(1.0), index_curve=index_curve)
        if index_scalar.is_err:
            return index_scalar  # type: ignore[return-value]
        return Ok(ucfe.unwrap() * index_scalar.unwrap())

    def try_immediate_rate_fixings_sensitivity(
        self,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        dc_res = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_res, Err):
            return dc_res
        disc_curve_: _BaseCurve = dc_res.unwrap()

        cfe = self.try_cashflow_fixings_sensitivity(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if cfe.is_err:
            return cfe

        if self.settlement_params.payment < disc_curve_.nodes.initial:
            # payment date is in the past
            return Ok(cfe.unwrap() * 0.0)

        return Ok(cfe.unwrap() * disc_curve_[self.settlement_params.payment])
