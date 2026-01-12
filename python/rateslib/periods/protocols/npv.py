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

from rateslib.curves import _BaseCurve
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
    _try_disc_required_maybe_from_curve,
)
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.parameters import (
    _IndexParams,
    _SettlementParams,
)
from rateslib.periods.parameters.settlement import _NonDeliverableParams
from rateslib.periods.utils import (
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        FX_,
        CurveOption_,
        DualTypes,
        FXForwards_,
        Result,
        _BaseCurve_,
        _FXVolOption_,
        datetime,
        datetime_,
        str_,
    )


def _screen_ex_div_and_forward(
    local_value: Result[DualTypes],
    rate_curve: CurveOption_,
    disc_curve: _BaseCurve_,
    ex_dividend: datetime,
    forward: datetime_ = NoInput(0),
    settlement: datetime_ = NoInput(0),
) -> Result[DualTypes]:
    """
    Remap an immediate, local currency value to account for a forward valuation and settlement.

    Parameters
    ----------
    local_value: Result[float, Dual, Dual2, Variable]
        The value measured with immediate effect expressed in local currency.
    rate_curve: _BaseCurve or NoInput
        The rate curve which might be used in place of the ``disc_curve`` if that not given.
    disc_curve: _BaseCurve or NoInput
        The discount curve used to discount units of local currency at an appropriate
        collateral rate.
    ex_dividend: datetime
        The ex-dividend date which, combined with ``settlement``, determines if this value
        is set to zero.
    settlement: datetime
        The settlement date to compare against an ex-dividend date to imply a cashflow.
    forward: datetime
        The projected forward valuation of the PV obtained via the discount curve

    Returns
    -------
    Float, Dual, Dual2, Variable
    """
    if local_value.is_err:
        return local_value

    # determine forward_ and settlement_ if not given
    is_settlement = not isinstance(settlement, NoInput)
    is_forward = not isinstance(forward, NoInput)

    if not is_settlement and not is_forward:
        return local_value  # immediate value is returned unadjusted

    dc_res = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
    if isinstance(dc_res, Err):
        return dc_res
    disc_curve_: _BaseCurve = dc_res.unwrap()

    if not is_settlement:
        # ex-div is assumed to always after a blank settlement
        return Ok(local_value.unwrap() / disc_curve_[forward])  # type: ignore[index]
    else:
        if settlement > ex_dividend:  # type: ignore[operator]
            return Ok(local_value.unwrap() * 0.0)  # TODO: profile this multiplication
            # in the case of Dualtypes this would be faster to just return 0.0
            # but the multiplication is used to handle DataFrame (FixingsSensitivity)
        if not is_forward:
            # forward is assumed to be immediate value if not given.
            # # forward is assumed to be equal to settlement
            return local_value  # / disc_curve_[settlement])  # type: ignore[index]
        else:
            return Ok(local_value.unwrap() / disc_curve_[forward])  # type: ignore[index]


class _WithNPV(Protocol):
    r"""
    Protocol to define value of any *Period* type.

    .. rubric:: Required methods

    .. autosummary::

      ~_WithNPV.immediate_local_npv

    .. rubric:: Provided methods

    .. autosummary::

      ~_WithNPV.local_npv
      ~_WithNPV.npv
      ~_WithNPV.try_immediate_local_npv
      ~_WithNPV.try_local_npv


    Notes
    -----
    Each *Period* type is required to implement the immediate expectation of value of
    its cashflow under the risk neutral measure, expressed in its local settlement currency.

    .. math::

       P_0 = \mathbb{E^Q}[V(m_T) C_T]
    """

    _settlement_params: _SettlementParams

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams` of the
        *Period*."""
        return self._settlement_params

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def immediate_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the immediate NPV of the *Period* in local settlement currency.

        This method does **not** adjust for ex-dividend and is an immediate measure according to,

        .. math::

           P_0 = \mathbb{E^Q} [V(m_T) C(m_T)]

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
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """  # noqa: E501
        raise NotImplementedError(  # pragma: no cover
            f"Period type '{type(self).__name__}' must implement `immediate_local_npv`"
        )

    def try_immediate_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPV.immediate_local_npv` with
        lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.immediate_local_npv(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

        pass

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
        r"""
        Calculate the NPV of the *Period* in local settlement currency.

        This method adjusts the immediate NPV for ex-dividend, settlement and forward projected value,
        according to,

        .. math::

           P(m_s, m_f) = \mathbb{I}(m_s) \frac{1}{v(m_f)} P_0,  \qquad \; \mathbb{I}(m_s) = \left \{ \begin{matrix} 0 & m_s > m_{ex} \\ 1 & m_s \leq m_{ex} \end{matrix} \right .

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
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        settlement: datetime, optional (set as immediate date)
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional (set as ``settlement``)
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable
        """  # noqa: E501
        local_immediate_npv = self.immediate_local_npv(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return _screen_ex_div_and_forward(
            local_value=Ok(local_immediate_npv),
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            ex_dividend=self.settlement_params.ex_dividend,
            settlement=settlement,
            forward=forward,
        ).unwrap()

    def try_local_npv(
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
        Replicate :meth:`~rateslib.periods.protocols._WithNPV.local_npv` with lazy
        exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.local_npv(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                settlement=settlement,
                forward=forward,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

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

        This method converts a local settlement currency value to a base accounting currency
        according to:

        .. math::

           P^{bas}(m_s, m_f) = f_{loc:bas}(m_f) P(m_s, m_f)

        .. hint::

           If the cashflows are unspecified or incalculable due to missing information this method
           will raise an exception. For a function that returns a `Result` indicating success or
           failure use :meth:`~rateslib.periods.protocols._WithNPV.try_local_npv`.

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
        settlement: datetime, optional, (set as immediate date)
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional, (set as ``settlement``)
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable or dict of such indexed by string currency.

        Notes
        -----
        If ``base`` is not provided then this function will return the value obtained from
        :meth:`~rateslib.periods.protocols._WithNPV.local_npv`.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        local_npv = self.local_npv(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )
        return _maybe_local(
            value=local_npv,
            local=local,
            currency=self.settlement_params.currency,
            fx=fx,
            base=base,
            forward=forward,
        )


class _WithIndexingStatic(Protocol):
    """
    Protocol to provide indexation for *Static Period* types.
    """

    _index_params: _IndexParams | None

    @property
    def index_params(self) -> _IndexParams | None:
        """
        The :class:`~rateslib.periods.parameters._IndexParams` of the *Period*,
        if any.
        """
        return self._index_params

    @property
    def is_indexed(self) -> bool:
        """
        Check whether the *Period* has indexation applied, which means it has ``index_params``.
        """
        return self.index_params is not None

    def index_up(self, value: DualTypes, index_curve: _BaseCurve_) -> DualTypes:
        """
        Apply indexation to a *Static Period* value using its ``index_params``.

        Parameters
        ----------
        value: float, Dual, Dual2, Variable
            The possible value to apply indexation to.
        index_curve: _BaseCurve, optional
            The index curve used to forecast index values, if necessary.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        if self.index_params is None:
            # then no indexation of the cashflow will occur.
            return value
        else:
            ir = self.index_params.try_index_ratio(index_curve).unwrap()[0]
            if self.index_params.index_only:
                return value * (ir - 1)
            else:
                return value * ir

    def try_index_up(self, value: Result[DualTypes], index_curve: _BaseCurve_) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithIndexingStatic.index_up`
        with lazy exception handling.

        Parameters
        ----------
        value: Result[float, Dual, Dual2, Variable]
            The possible value to apply indexation to.
        index_curve: _BaseCurve, optional
            The index curve used to forecast index values, if necessary.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.index_up(
                value=value.unwrap(),
                index_curve=index_curve,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)


class _WithNonDeliverableStatic(Protocol):
    """
    Protocol to provide non-deliverable conversion for *Static Period* types.



    """

    _non_deliverable_params: _NonDeliverableParams | None

    @property
    def non_deliverable_params(self) -> _NonDeliverableParams | None:
        """The :class:`~rateslib.periods.parameters._NonDeliverableParams` of the
        *Period*., if any."""
        return self._non_deliverable_params

    @property
    def is_non_deliverable(self) -> bool:
        """
        Check whether the *Period* is non-deliverable,
        which means it has ``non_deliverable_params``.
        """
        return self.non_deliverable_params is not None

    def convert_deliverable(self, value: DualTypes, fx: FXForwards_) -> DualTypes:
        """
        Apply settlement currency conversion to a *Static Period* using its
        ``non_deliverable_params``.

        Parameters
        ----------
        value: float, Dual, Dual2, Variable
            The possible value to apply settlement currency conversion to.
        fx: FXForwards, optional
            The object used to forecast forward FX rates, if necessary.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        if self.non_deliverable_params is None:
            # then cashflow is directly deliverable
            return value
        else:
            fx_fix = self.non_deliverable_params.fx_fixing.try_value_or_forecast(fx).unwrap()
            c = value * (fx_fix if not self.non_deliverable_params.fx_reversed else (1.0 / fx_fix))
            return c

    def try_convert_deliverable(
        self, value: Result[DualTypes], fx: FXForwards_
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNonDeliverableStatic.convert_deliverable`
        with lazy exception handling.

        Parameters
        ----------
        value: Result[float, Dual, Dual2, Variable]
            The possible value to apply settlement currency conversion to.
        fx: FXForwards, optional
            The object used to forecast forward FX rates, if necessary.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """  # noqa: E501
        try:
            v = self.convert_deliverable(
                value=value.unwrap(),
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)


class _WithNPVStatic(_WithNPV, _WithIndexingStatic, _WithNonDeliverableStatic, Protocol):
    r"""
    Protocol to establish value of any *Static Period* type.

    .. rubric:: Required methods

    .. autosummary::

       ~_WithNPVStatic.unindexed_reference_cashflow

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithNPVStatic.reference_cashflow
       ~_WithNPVStatic.unindexed_cashflow
       ~_WithNPVStatic.cashflow
       ~_WithNPVStatic.immediate_local_npv
       ~_WithNPVStatic.local_npv
       ~_WithNPVStatic.npv
       ~_WithNPVStatic.try_unindexed_reference_cashflow
       ~_WithNPVStatic.try_reference_cashflow
       ~_WithNPVStatic.try_unindexed_cashflow
       ~_WithNPVStatic.try_cashflow
       ~_WithNPVStatic.try_immediate_local_npv
       ~_WithNPVStatic.try_local_npv

    Notes
    -----
    A *Static Period* type is one with a defined, non-random cashflow date, and for which
    indexation and non-deliverability components are independent and can be taken outside of
    the expectation of value.

    Each *Static Period* is required to implement the expectation of its unindexed reference
    currency cashflow under the risk neutral measure, paid at the known payment date,
    :math:`m_t`.

    .. math::

       \mathbb{E^Q}[\bar{C}_t]

    """

    # required by each Static Period...
    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the cashflow for the *Static Period* before settlement currency and
        indexation adjustments.

        .. math::

           \mathbb{E^Q}[\bar{C}_t]

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        raise NotImplementedError(  # pragma: no cover
            f"Period type '{type(self).__name__}' must implement `unindexed_reference_cashflow`"
        )

    # automatically provided for each Static Period...

    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.unindexed_reference_cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """  # noqa: E501
        try:
            v = self.unindexed_reference_cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the cashflow for the *Static Period* before settlement currency adjustment
        but after indexation.

        .. math::

           I_r\mathbb{E^Q}[\bar{C}_t]

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        urc = self.unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.index_up(value=urc, index_curve=index_curve)

    def try_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.reference_cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.reference_cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def unindexed_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the cashflow for the *Static Period* with settlement currency adjustment
        but without indexation.

        .. math::

           f(m_d)\mathbb{E^Q}[\bar{C}_t]

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        urc = self.unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.convert_deliverable(value=urc, fx=fx)

    def try_unindexed_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.unindexed_cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.unindexed_cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the cashflow for the *Period* with settlement currency adjustment
        and indexation.

        .. math::

           I_r f(m_d)\mathbb{E^Q}[\bar{C}_t]

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows, if necessary.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        rc = self.reference_cashflow(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.convert_deliverable(value=rc, fx=fx)

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def immediate_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        r"""
        Calculate the NPV of the *Period* in local settlement currency.

        This method does **not** adjust for ex-dividend and is an immediate measure according to,

        .. math::

           P_0 = v(m_t) I_r f(m_d) \mathbb{E^Q} [\bar{C}_t]

        for non-deliverable delivery, :math:`m_d`, and index ratio, :math:`I_r`.

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
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        # dc_res = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        # if isinstance(dc_res, Err):
        #     return dc_res
        # disc_curve_: _BaseCurve = dc_res.unwrap()

        disc_curve_ = _disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if self.settlement_params.payment < disc_curve_.nodes.initial:
            # payment date is in the past
            return 0.0

        c = self.cashflow(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve_,
            fx_vol=fx_vol,
            fx=fx,
        )
        return c * disc_curve_[self.settlement_params.payment]
