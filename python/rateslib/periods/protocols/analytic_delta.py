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

from rateslib.curves._parsers import (
    _try_disc_required_maybe_from_curve,
)
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.parameters.settlement import _SettlementParams
from rateslib.periods.protocols.npv import (
    _screen_ex_div_and_forward,
    _WithIndexingStatic,
    _WithNonDeliverableStatic,
)
from rateslib.periods.utils import (
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        FXForwards_,
        FXRevised_,
        Result,
        _BaseCurve,
        _BaseCurve_,
        _FXVolOption_,
        datetime_,
        str_,
    )


class _WithAnalyticDelta(Protocol):
    r"""
    Protocol to establish analytical sensitivity to rate type metrics.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithAnalyticDelta.try_immediate_local_analytic_delta

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithAnalyticDelta.try_local_analytic_delta
       ~_WithAnalyticDelta.analytic_delta

    Notes
    -----
    Since this is *analytical*, each *Period* type must define its unique referenced sensitivity
    to interest rates. This protocol ultimately determines the quantity,

    .. math::

       A^{bas}(m_f, m_s) = \frac{\partial P^{bas}(m_f, m_s)}{\partial \xi}, \quad \text{for some quantity, } \xi
    """  # noqa: E501

    _settlement_params: _SettlementParams

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` of the
        *Period*."""
        return self._settlement_params

    def try_immediate_local_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the immediate, analytic rate delta of a *Period* expressed in local
        settlement currency, with lazy error raising.

        This method does **not** adjust for ex-dividend and is an immediate measure according to,

        .. math::

           A_0 = \frac{\partial P_0}{\partial \xi}, \quad \text{for some, } \xi

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
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        pass

    def try_local_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the analytic rate delta of a *Period* expressed in local settlement currency,
        with lazy error raising.

        This method adjusts the immediate NPV for ex-dividend and forward projected value,
        according to,

        .. math::

           A(m_s, m_f) = \mathbb{I}(m_s) \frac{1}{v(m_f)} A_0,  \qquad \; \mathbb{I}(m_s) = \left \{ \begin{matrix} 0 & m_s > m_{ex} \\ 1 & m_s \leq m_{ex} \end{matrix} \right .

        for forward, :math:`m_f`, settlement, :math:`m_s`, and ex-dividend, :math:`m_{ex}`.

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
        Result[float, Dual, Dual2, Variable]
        """  # noqa: E501
        local_immediate_result = self.try_immediate_local_analytic_delta(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return _screen_ex_div_and_forward(
            local_value=local_immediate_result,
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            ex_dividend=self.settlement_params.ex_dividend,
            settlement=settlement,
            forward=forward,
        )

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
        Calculate the analytic rate delta of the *Period* converted to any other
        *base* accounting currency.

        This method converts a local settlement currency value to a base accounting currency
        according to:

        .. math::

           A^{bas}(m_s, m_f) = f_{loc:bas}(m_f) A(m_s, m_f)

        .. hint::

           If the cashflows are unspecified or incalculable due to missing information this method
           will raise an exception. For a function that returns a `Result` indicating success or
           failure use
           :meth:`~rateslib.periods._WithAnalyticDelta.try_local_analytic_delta`.

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
            An override flag to return a dict of values indexed by string currency.
        settlement: datetime, optional, (set as immediate date)
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional, (set as ``settlement``)
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable or dict
        """
        local_delta = self.try_local_analytic_delta(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        ).unwrap()
        return _maybe_local(
            value=local_delta,
            local=local,
            currency=self.settlement_params.currency,
            fx=fx,
            base=base,
            forward=forward,
        )


class _WithAnalyticDeltaStatic(
    _WithAnalyticDelta, _WithIndexingStatic, _WithNonDeliverableStatic, Protocol
):
    r"""
    Protocol to establish analytical sensitivity to rate type metrics for *Static Period* types.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithAnalyticDeltaStatic.try_unindexed_reference_cashflow_analytic_delta

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithAnalyticDeltaStatic.try_reference_cashflow_analytic_delta
       ~_WithAnalyticDeltaStatic.try_unindexed_cashflow_analytic_delta
       ~_WithAnalyticDeltaStatic.try_cashflow_analytic_delta
       ~_WithAnalyticDeltaStatic.try_immediate_local_analytic_delta
       ~_WithAnalyticDeltaStatic.try_local_analytic_delta
       ~_WithAnalyticDeltaStatic.analytic_delta

    Notes
    -----
    Since this is *analytical*, each *Period* type must define its unique referenced sensitivity
    to interest rates. This protocol ultimately determines the quantity,

    .. math::

       A^{bas}(m_f, m_s) = \frac{\partial P^{bas}(m_f, m_s)}{\partial \xi}, \quad \text{for some quantity, } \xi
    """  # noqa: E501

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow analytic delta for the *Static Period* before settlement currency
        adjustment and indexation, with lazy error raising.

        .. math::

           \frac{\partial \mathbb{E^Q}[\bar{C}_t]}{\partial \xi}

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        raise NotImplementedError(
            f"type {type(self).__name__} has not implemented "
            f"`try_unindexed_reference_cashflow_analytic_delta`"
        )

    def try_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow analytic delta for the *Static Period* before settlement currency
        adjustment but after indexation, with lazy error raising.

        .. math::

           I_r \frac{\partial \mathbb{E^Q}[\bar{C}_t]}{\partial \xi}

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        rrad = self.try_unindexed_reference_cashflow_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve
        )
        return self.try_index_up(value=rrad, index_curve=index_curve)

    def try_unindexed_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow analytic delta for the *Static Period* with settlement currency
        adjustment but without indexation, with lazy error raising.

        .. math::

           f(m_d) \frac{\partial \mathbb{E^Q}[\bar{C}_t]}{\partial \xi}

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        rrad = self.try_unindexed_reference_cashflow_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve
        )
        return self.try_convert_deliverable(value=rrad, fx=fx)

    def try_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow for the *Period* with settlement currency adjustment
        and indexation.

        .. math::

           I_r f(m_d) \frac{\partial \mathbb{E^Q}[\bar{C}_t]}{\partial \xi}

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
        Result[float, Dual, Dual2, Variable]

        """
        rad = self.try_reference_cashflow_analytic_delta(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve
        )
        lad = self.try_convert_deliverable(value=rad, fx=fx)  # type: ignore[arg-type]
        if lad.is_err:
            return lad
        return lad

    def try_immediate_local_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        dc_res = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_res, Err):
            return dc_res
        disc_curve_: _BaseCurve = dc_res.unwrap()

        if self.settlement_params.payment < disc_curve_.nodes.initial:
            # payment date is in the past
            return Ok(0.0)

        cad = self.try_cashflow_analytic_delta(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve_,
            fx_vol=fx_vol,
            fx=fx,
        )
        if cad.is_err:
            return cad
        return Ok(cad.unwrap() * disc_curve_[self.settlement_params.payment])
