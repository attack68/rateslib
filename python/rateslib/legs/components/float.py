from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Series

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.legs.components.amortization import Amortization, _AmortizationType, _get_amortization
from rateslib.legs.components.protocols import _WithAnalyticDelta, _WithCashflows, _WithNPV
from rateslib.legs.components.utils import _leg_fixings_to_list
from rateslib.periods.components import Cashflow, FloatPeriod
from rateslib.periods.components.parameters import _FloatRateParams, _SettlementParams

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        FX_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FloatRateSeries,
        Frequency,
        IndexMethod,
        Period,
        Schedule,
        int_,
        str_,
    )


class FloatLeg(_WithNPV, _WithCashflows, _WithAnalyticDelta):
    """
    Abstract base class with common parameters for all ``Leg`` subclasses.

    Parameters
    ----------
    schedule: Schedule
        The :class:`~rateslib.scheduling.Schedule` object which structures contiguous *Periods*.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.
    initial_exchange : bool
        Whether to also include an initial notional exchange.
    final_exchange : bool
        Whether to also include a final notional exchange and interim amortization
        notional exchanges.

    Notes
    -----
    The (optional) initial cashflow notional is set as the negative of the notional.
    The final cashflow notional is set as the notional.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods as interim exchanges if
    ``final_exchange`` is *True*.

    Examples
    --------
    See :ref:`Leg Examples<legs-doc>`

    Attributes
    ----------
    schedule : Schedule
    currency : str
    convention : str
    periods : list
    initial_exchange : bool
    final_exchange : bool

    See Also
    --------
    FixedLeg : Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.
    FloatLeg : Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.
    IndexFixedLeg : Create a fixed leg composed of :class:`~rateslib.periods.IndexFixedPeriod` s.
    ZeroFixedLeg : Create a zero coupon leg composed of a :class:`~rateslib.periods.FixedPeriod`.
    ZeroFloatLeg : Create a zero coupon leg composed of a :class:`~rateslib.periods.FloatPeriod` s.
    ZeroIndexLeg : Create a zero coupon leg composed of :class:`~rateslib.periods.IndexFixedPeriod`.
    CustomLeg : Create a leg composed of user specified periods.
    """

    @property
    def rate_params(self) -> _FloatRateParams:
        """The :class:`~rateslib.periods.components.parameters._FloatRateParams` associated with
        the first :class:`~rateslib.periods.components.FloatPeriod`."""
        return self._regular_periods[0].rate_params

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.components.parameters._SettlementParams` associated with
        the first :class:`~rateslib.periods.components.FloatPeriod`."""
        return self._regular_periods[0].settlement_params

    @property
    def periods(self) -> list[Period]:
        """Combine all period collection types into an ordered list."""
        periods_: list[Period] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])

        if self._interim_exchange_periods is not None:
            interleaved_periods_: list[Period] = [
                val
                for pair in zip(self._regular_periods, self._interim_exchange_periods, strict=False)
                for val in pair
            ]
            interleaved_periods_.append(self._regular_periods[-1])  # add last regular period
        else:
            interleaved_periods_ = list(self._regular_periods)
        periods_.extend(interleaved_periods_)

        if self._exchange_periods[1] is not None:
            periods_.append(self._exchange_periods[1])

        return periods_

    @property
    def float_spread(self) -> DualTypes:
        return self._regular_periods[0].rate_params.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        for period in self._regular_periods:
            period.rate_params.float_spread = value

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def __init__(
        self,
        schedule: Schedule,
        *,
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # settlement and currency
        notional: DualTypes_ = NoInput(0),
        amortization: DualTypes_ | list[DualTypes] | Amortization | str = NoInput(0),
        currency: str_ = NoInput(0),
        pair: str_ = NoInput(0),
        # period
        convention: str_ = NoInput(0),
        initial_exchange: bool = False,
        final_exchange: bool = False,
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    ) -> None:
        self._schedule = schedule
        self._notional: DualTypes = _drb(defaults.notional, notional)
        self._amortization: Amortization = _get_amortization(
            amortization, self._notional, self._schedule.n_periods
        )
        self._currency: str = _drb(defaults.base_currency, currency).lower()
        self._convention: str = _drb(defaults.convention, convention)

        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(
                payment=self._schedule.pschedule2[0],
                notional=-self._amortization.outstanding[0],
                currency=self._currency,
                ex_dividend=self._schedule.pschedule3[0],
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings,
                index_base_date=self._schedule.aschedule[0],
                index_reference_date=self._schedule.aschedule[0],
            )
        if not final_exchange:
            _final_cf: Cashflow | None = None
        else:
            _final_cf = Cashflow(
                payment=self._schedule.pschedule2[-1],
                notional=self._amortization.outstanding[-1],
                currency=self._currency,
                ex_dividend=self._schedule.pschedule3[-1],
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings,
                index_base_date=self._schedule.aschedule[0],
                index_reference_date=self._schedule.aschedule[-1],
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        rate_fixings_list = _leg_fixings_to_list(rate_fixings, self._schedule.n_periods)
        self._regular_periods = tuple(
            [
                FloatPeriod(
                    float_spread=float_spread,
                    rate_fixings=rate_fixings_list[i],
                    fixing_method=fixing_method,
                    method_param=method_param,
                    spread_compound_method=spread_compound_method,
                    fixing_frequency=fixing_frequency,
                    fixing_series=fixing_series,
                    # currency args
                    payment=self._schedule.pschedule[i + 1],
                    currency=self._currency,
                    notional=self._amortization.outstanding[i],
                    ex_dividend=self._schedule.pschedule3[i + 1],
                    # period params
                    start=self._schedule.aschedule[i],
                    end=self._schedule.aschedule[i + 1],
                    frequency=self._schedule.frequency_obj,
                    convention=self._convention,
                    termination=self._schedule.aschedule[-1],
                    stub=self._schedule._stubs[i],
                    roll=NoInput(0),  #  defined by Frequency
                    calendar=self._schedule.calendar,
                    adjuster=self._schedule.accrual_adjuster,
                    # index params
                    index_base=index_base,
                    index_lag=index_lag,
                    index_method=index_method,
                    index_fixings=index_fixings,
                    index_base_date=self._schedule.aschedule[0],
                    index_reference_date=self._schedule.aschedule[i + 1],
                )
                for i in range(self._schedule.n_periods)
            ]
        )

        # amortization exchanges
        if not final_exchange or self._amortization._type == _AmortizationType.NoAmortization:
            self._interim_exchange_periods: tuple[Cashflow, ...] | None = None
        else:
            self._interim_exchange_periods = tuple(
                [
                    Cashflow(
                        notional=self._amortization.amortization[i],
                        payment=self._schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self._schedule.pschedule3[i + 1],
                        # index params
                        index_base=index_base,
                        index_lag=index_lag,
                        index_method=index_method,
                        index_fixings=index_fixings,
                        index_base_date=self._schedule.aschedule[0],
                        index_reference_date=self._schedule.aschedule[i + 1],
                    )
                    for i in range(self._schedule.n_periods - 1)
                ]
            )

    @property
    def _is_linear(self) -> bool:
        """
        Tests if analytic delta spread is a linear function affecting NPV.

        This is non-linear if the spread is itself compounded, which only occurs
        on RFR trades with *"isda_compounding"* or *"isda_flat_compounding"*, which
        should typically be avoided anyway.

        Returns
        -------
        bool
        """
        # ruff: noqa: SIM103
        if (
            self.rate_params.fixing_method != FloatFixingMethod.IBOR
            and self.rate_params.spread_compound_method != SpreadCompoundMethod.NoneSimple
        ):
            return False
        return True

    def _spread(
        self,
        target_npv: DualTypes,
        rate_curve: CurveOption_,
        disc_curve: CurveOption_,
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculates an adjustment to the ``fixed_rate`` or ``float_spread`` to match
        a specific target NPV.

        Parameters
        ----------
        target_npv : float, Dual or Dual2
            The target NPV that an adjustment to the parameter will achieve. **Must
            be in local currency of the leg.**
        fore_curve : Curve or LineCurve
            The forecast curve passed to analytic delta calculation.
        disc_curve : Curve
            The discounting curve passed to analytic delta calculation.
        fx : FXForwards, optional
            Required for multi-currency legs which are MTM exchanged.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        ``FixedLeg`` and ``FloatLeg`` with a *"none_simple"* spread compound method have
        linear sensitivity to the spread. This can be calculated directly and
        exactly using an analytic delta calculation.

        *"isda_compounding"* and *"isda_flat_compounding"* spread compound methods
        have non-linear sensitivity to the spread. This requires a root finding,
        iterative algorithm, which, coupled with very poor performance of calculating
        period rates under this method is exceptionally slow. We approximate this
        using first and second order AD and extrapolate a solution as a Taylor
        expansion. This results in approximation error.

        Examples
        --------
        """
        if self._is_linear:
            a_delta: DualTypes = self.analytic_delta(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                fx=fx,
                base=self.settlement_params.currency,
            )
            return -target_npv / a_delta
        else:
            return self._spread_isda_approximated_rate(target_npv, fore_curve, disc_curve)  # type: ignore[arg-type]
            # _ = self._spread_isda_dual2(target_npv, fore_curve, disc_curve, fx)
