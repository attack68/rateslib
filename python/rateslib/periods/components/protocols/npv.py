from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.curves import _BaseCurve
from rateslib.curves._parsers import (
    _try_disc_required_maybe_from_curve,
)
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.components.parameters import (
    _IndexParams,
    _SettlementParams,
)
from rateslib.periods.components.parameters.settlement import _NonDeliverableParams
from rateslib.periods.components.utils import (
    _maybe_local,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        CurveOption_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Result,
        _BaseCurve_,
        datetime_,
        str_,
    )


class _WithNPV(Protocol):
    r"""
    Protocol to establish value of any *Period* type.

    Notes
    -----
    Each *Period* type is required to implement the immediate expectation of value of
    its cashflow under the risk neutral measure, expressed in its local settlement currency.

    .. math::

       P_0 = \mathbb{E^Q}[V(m_T) C_T]
    """

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
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the NPV of the *Period* in local settlement currency, with lazy error raising.

        This method adjusts for ex-dividend and forward projected value, according to,

        .. math::

           P(m_s, m_f) = \mathbb{I}(m_s) \frac{1}{v(m_f)} \mathbb{E^Q} [V(m_T) C(m_T) ],  \qquad \; \mathbb{I}(m_s) = \left \{ \begin{matrix} 0 & m_s > m_{ex} \\ 1 & m_s \leq m_{ex} \end{matrix} \right .

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
        pass

    def _screen_ex_div_and_forward(
        self,
        local_npv: DualTypes,
        disc_curve: _BaseCurve,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Remap an immediate, local currency NPV to account for a forward valuation and settlement.

        Parameters
        ----------
        local_npv: float, Dual, Dual2, Variable
            The local currency PV measured with immediate effect.
        settlement: datetime
            The settlement date to compare against an ex-dividend date to imply a cashflow.
        forward: datetime
            The projected forward valuation of the PV obtained via the discount curve

        Returns
        -------
        Float, Dual, Dual2, Variable
        """
        # determine forward_ and settlement_ if not given
        if isinstance(settlement, NoInput):
            if isinstance(forward, NoInput):
                return local_npv
            else:
                # ex-div is assumed to always after a blank settlement
                return local_npv / disc_curve[forward]
        else:
            if settlement > self.settlement_params.ex_dividend:
                return 0.0
            if isinstance(forward, NoInput):
                # forward is assumed to be equal to settlement
                return local_npv / disc_curve[settlement]
            else:
                return local_npv / disc_curve[forward]

    def npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
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
           failure use :meth:`~rateslib.periods.components._WithNPV.try_local_npv`.

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
        :meth:`~rateslib.periods.components._WithNPV.try_local_npv`.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        local_npv = self.try_local_npv(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        ).unwrap()
        return _maybe_local(
            value=local_npv, local=local, currency=self.settlement_params.currency, fx=fx, base=base
        )


class _WithIndexingStatic(Protocol):
    """
    Protocol to provide indexation for *Static Period* types.
    """

    index_params: _IndexParams | None

    @property
    def is_indexed(self) -> bool:
        """
        Check whether the *Period* has indexation applied, which means it has ``index_params``.
        """
        return self.index_params is not None

    def try_index_up(self, value: Result[DualTypes], index_curve: _BaseCurve_) -> Result[DualTypes]:
        """
        Apply indexation to a *Static Period* using its ``index_params``, with lazy error raising.

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
    """
    Protocol to provide non-deliverable conversion for *Static Period* types.
    """

    settlement_params: _SettlementParams
    non_deliverable_params: _NonDeliverableParams | None

    @property
    def is_non_deliverable(self) -> bool:
        """
        Check whether the *Period* is non-deliverable,
        which means it has ``non_deliverable_params``.
        """
        return self.non_deliverable_params is not None

    def try_convert_deliverable(
        self, value: Result[DualTypes], fx: FXForwards_
    ) -> Result[DualTypes]:
        """
        Apply settlement currency conversion to a *Static Period* using its
        ``non_deliverable_params``, with lazy error raising.

        Parameters
        ----------
        value: Result[float, Dual, Dual2, Variable]
            The possible value to apply settlement currency conversion to.
        fx: FXForwards, optional
            The object used to forecast forward FX rates, if necessary.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        if self.non_deliverable_params is None:
            # then cashflow is directly deliverable
            return value
        else:
            if value.is_err:
                return value

            value_: DualTypes = value.unwrap()
            fx_fix_res = self.non_deliverable_params.fx_fixing.try_value_or_forecast(fx)
            if fx_fix_res.is_err:
                return fx_fix_res
            else:
                fx_fix = fx_fix_res.unwrap()
            c = value_ * (fx_fix if not self.non_deliverable_params.fx_reversed else (1.0 / fx_fix))
            return Ok(c)


class _WithNPVStatic(_WithNPV, _WithIndexingStatic, _WithNonDeliverableStatic, Protocol):
    r"""
    Protocol to establish value of any *Static Period* type.

    Notes
    -----
    A *Static Period* type is one with a defined, non-random cashflow date, and for which
    indexation and non-deliverability components are independent and can be taken outside of
    the expectation of value.

    Each *Static Period* is required to implement the expectation of its unindexed reference
    currency cashflow under the risk neutral measure, paid at the known payment date.

    .. math::

       \mathbb{E^Q}[\bar{C}_t]

    """

    settlement_params: _SettlementParams

    # required by each Static Period...
    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow for the *Static Period* before settlement currency and
        indexation adjustments, with lazy error raising.

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
        Result[float, Dual, Dual2, Variable]
        """
        pass

    # automatically provided for each Static Period...

    def try_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow for the *Static Period* before settlement currency adjustment
        but after indexation, with lazy error raising.

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
        Result[float, Dual, Dual2, Variable]
        """
        rrc = self.try_unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.try_index_up(value=rrc, index_curve=index_curve)

    def try_unindexed_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the cashflow for the *Static Period* with settlement currency adjustment
        but without indexation, with lazy error raising.

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
        Result[float, Dual, Dual2, Variable]
        """
        rrc = self.try_unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.try_convert_deliverable(value=rrc, fx=fx)

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
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
        Result[float, Dual, Dual2, Variable]
        """
        rc = self.try_reference_cashflow(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        return self.try_convert_deliverable(value=rc, fx=fx)

    def try_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the NPV of the *Period* in local settlement currency, with lazy error raising.

        This method adjusts for ex-dividend and forward projected value. It also adds
        indexation and non-deliverability to the unindexed reference cashflow according to,

        .. math::

           P(m_s, m_f) = \mathbb{I}(m_s) \frac{1}{v(m_f)} v(m_t) I_r f(m_d) \mathbb{E^Q} [\bar{C}_t],  \qquad \; \mathbb{I}(m_s) = \left \{ \begin{matrix} 0 & m_s > m_{ex} \\ 1 & m_s \leq m_{ex} \end{matrix} \right .

        for forward, :math:`m_f`, settlement, :math:`m_s`, and ex-dividend, :math:`m_{ex}`,
        non-deliverable delivery, :math:`m_d`, and index ratio, :math:`I_r`.

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
        dc_res = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_res, Err):
            return dc_res
        disc_curve_: _BaseCurve = dc_res.unwrap()

        if self.settlement_params.payment < disc_curve_.nodes.initial:
            # payment date is in the past
            return Ok(0.0)

        c = self.try_cashflow(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve_,
            fx_vol=fx_vol,
            fx=fx,
        )
        if c.is_err:
            return c

        # this will also handle cashflows before the curve as DF is zero.
        pv0 = c.unwrap() * disc_curve_[self.settlement_params.payment]
        return Ok(
            self._screen_ex_div_and_forward(
                local_npv=pv0, disc_curve=disc_curve_, settlement=settlement, forward=forward
            )
        )
