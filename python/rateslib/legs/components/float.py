from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Series

import rateslib.errors as err
from rateslib import defaults
from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.legs.components.amortization import Amortization, _AmortizationType, _get_amortization
from rateslib.legs.components.protocols import (
    _BaseLeg,
)
from rateslib.legs.components.utils import _leg_fixings_to_list
from rateslib.periods.components import Cashflow, FloatPeriod, MtmCashflow
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
        LegFixings,
        Schedule,
        _BaseCurve_,
        _BasePeriod,
        datetime,
        int_,
        str_,
    )


class FloatLeg(_BaseLeg):
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
    def periods(self) -> list[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        periods_: list[_BasePeriod] = []

        if self._exchange_periods[0] is not None:
            periods_.append(self._exchange_periods[0])

        args: tuple[tuple[FloatPeriod | MtmCashflow | Cashflow, ...], ...] = (
            self._regular_periods[:-1],
        )
        if self._mtm_exchange_periods is not None:
            args += (self._mtm_exchange_periods,)
        if self._interim_exchange_periods is not None:
            args += (self._interim_exchange_periods,)
        interleaved_periods_: list[_BasePeriod] = [
            item for combination in zip(*args, strict=True) for item in combination
        ]
        interleaved_periods_.append(self._regular_periods[-1])  # add last regular period
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

    @property
    def amortization(self) -> Amortization:
        return self._amortization

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
        # non-deliverable
        pair: str_ = NoInput(0),
        fx_fixings: LegFixings = NoInput(0),  # type: ignore[type-var]
        mtm: bool = False,
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

        index_fixings_ = _leg_fixings_to_list(index_fixings, self.schedule.n_periods)
        fx_fixings_ = _leg_fixings_to_list(fx_fixings, self.schedule.n_periods)
        # Exchange periods
        if not initial_exchange:
            _ini_cf: Cashflow | None = None
        else:
            _ini_cf = Cashflow(
                payment=self.schedule.pschedule2[0],
                notional=-self._amortization.outstanding[0],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[0],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0],
                delivery=self.schedule.pschedule2[0],
                # index params
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[0],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[0],
            )
        final_exchange_ = final_exchange or initial_exchange
        if not final_exchange_:
            _final_cf: Cashflow | None = None
        else:
            _final_cf = Cashflow(
                payment=self.schedule.pschedule2[-1],
                notional=self._amortization.outstanding[-1],
                currency=self._currency,
                ex_dividend=self.schedule.pschedule3[-1],
                # non-deliverable
                pair=pair,
                fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[-1],
                delivery=self.schedule.pschedule2[0] if not mtm else self.schedule.pschedule2[-2],
                # index parameters
                index_base=index_base,
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings_[-1],
                index_base_date=self.schedule.aschedule[0],
                index_reference_date=self.schedule.aschedule[-1],
            )
        self._exchange_periods = (_ini_cf, _final_cf)

        def fx_delivery(i: int) -> datetime:
            if not mtm:
                # then ND type is a one-fixing only
                return self.schedule.pschedule2[0]
            else:
                if final_exchange_:
                    # then ND type is a XCS
                    return self.schedule.pschedule2[i]
                else:
                    # then ND type is IRS
                    return self.schedule.pschedule[i + 1]

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
                    payment=self.schedule.pschedule[i + 1],
                    currency=self._currency,
                    notional=self.amortization.outstanding[i],
                    ex_dividend=self.schedule.pschedule3[i + 1],
                    # period params
                    start=self.schedule.aschedule[i],
                    end=self.schedule.aschedule[i + 1],
                    frequency=self.schedule.frequency_obj,
                    convention=self._convention,
                    termination=self.schedule.aschedule[-1],
                    stub=self.schedule._stubs[i],
                    roll=NoInput(0),  #  defined by Frequency
                    calendar=self.schedule.calendar,
                    adjuster=self.schedule.accrual_adjuster,
                    # non-deliverable : Not allowed with notional exchange
                    pair=pair,
                    fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[i],
                    delivery=fx_delivery(i),
                    # index params
                    index_base=index_base,
                    index_lag=index_lag,
                    index_method=index_method,
                    index_fixings=index_fixings_[i],
                    index_base_date=self._schedule.aschedule[0],
                    index_reference_date=self._schedule.aschedule[i + 1],
                )
                for i in range(self._schedule.n_periods)
            ]
        )

        # amortization exchanges
        if not final_exchange_ or self.amortization._type == _AmortizationType.NoAmortization:
            self._interim_exchange_periods: tuple[Cashflow, ...] | None = None
        else:
            self._interim_exchange_periods = tuple(
                [
                    Cashflow(
                        notional=self.amortization.amortization[i],
                        payment=self.schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self.schedule.pschedule3[i + 1],
                        # non-deliverable params
                        pair=pair,
                        fx_fixings=fx_fixings_[0] if not mtm else fx_fixings_[i + 1],
                        delivery=self.schedule.pschedule2[0]
                        if not mtm
                        else self.schedule.pschedule2[i + 1],  # schedule for exchanges
                        # index params
                        index_base=index_base,
                        index_lag=index_lag,
                        index_method=index_method,
                        index_fixings=index_fixings_[i],
                        index_base_date=self._schedule.aschedule[0],
                        index_reference_date=self._schedule.aschedule[i + 1],
                    )
                    for i in range(self._schedule.n_periods - 1)
                ]
            )

        # mtm exchanges
        if mtm and final_exchange_:
            if isinstance(pair, NoInput):
                raise ValueError(err.VE_PAIR_AND_LEG_MTM)
            self._mtm_exchange_periods: tuple[MtmCashflow, ...] | None = tuple(
                [
                    MtmCashflow(
                        payment=self.schedule.pschedule2[i + 1],
                        notional=-self.amortization.outstanding[i],
                        pair=pair,
                        start=self.schedule.pschedule2[i],
                        end=self.schedule.pschedule2[i + 1],
                        currency=self._currency,
                        ex_dividend=self.schedule.pschedule3[i + 1],
                        fx_fixings_start=fx_fixings_[i],
                        fx_fixings_end=fx_fixings_[i + 1],
                        # index params
                        index_base=index_base,
                        index_lag=index_lag,
                        index_method=index_method,
                        index_fixings=index_fixings_[i],
                        index_base_date=self.schedule.aschedule[0],
                        index_reference_date=self.schedule.aschedule[i + 1],
                    )
                    for i in range(self.schedule.n_periods - 1)
                ]
            )
        else:
            self._mtm_exchange_periods = None

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
        index_curve: _BaseCurve_ = NoInput(0),
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
            original_z = self.float_spread
            original_npv = self.npv(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx=fx,
            )

            def s(g: DualTypes) -> DualTypes:
                """
                This determines the NPV change subject to a given float spread change denoted, g.
                """
                self.float_spread = g + original_z
                return (
                    self.npv(  # type: ignore[operator]
                        rate_curve=rate_curve,
                        disc_curve=disc_curve,
                        index_curve=index_curve,
                        fx=fx,
                    )
                    - original_npv
                )

            result = ift_1dim(
                s=s,
                s_tgt=target_npv,
                h="modified_brent",
                ini_h_args=(-10000, 10000),
                func_tol=1e-9,
                conv_tol=1e-6,
            )

            self.float_spread = original_z
            _: DualTypes = result["g"]
            return _
