from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Series

from rateslib import defaults
from rateslib.curves import index_value
from rateslib.default import NoInput, _drb
from rateslib.legs.base import BaseLeg, _FixedLegMixin
from rateslib.periods import Cashflow, IndexCashflow, IndexFixedPeriod
from rateslib.periods.index import _validate_index_method_and_lag
from rateslib.scheduling import Schedule

if TYPE_CHECKING:
    from rateslib.typing import NPV, Any, Curve_, DataFrame, DualTypes, DualTypes_, int_, str_


class _IndexLegMixin:
    schedule: Schedule
    index_method: str
    _index_fixings: DualTypes | Series[DualTypes] | NoInput = NoInput(0)  # type: ignore[type-var]
    _index_base: DualTypes_ = NoInput(0)
    periods: list[IndexFixedPeriod | IndexCashflow | Cashflow]
    index_lag: int

    @property
    def index_fixings(self) -> DualTypes | list[DualTypes] | Series[DualTypes] | NoInput:  # type: ignore[type-var]
        return self._index_fixings

    @index_fixings.setter
    def index_fixings(
        self,
        value: DualTypes | Series[DualTypes] | NoInput,  # type: ignore[type-var]
    ) -> None:
        if isinstance(value, Series):
            value = _validate_index_fixings_as_series(value)  # type: ignore[arg-type]
        self._index_fixings: DualTypes | Series[DualTypes] | NoInput = value  # type: ignore[type-var]

        if isinstance(value, NoInput):
            for p in [_ for _ in self.periods if type(_) is not Cashflow]:
                p.index_fixings = NoInput(0)  # type: ignore[union-attr]
        elif isinstance(value, Series):
            for p in [_ for _ in self.periods if type(_) is not Cashflow]:
                date_: datetime = p.end if type(p) is IndexFixedPeriod else p.payment
                p.index_fixings = index_value(  # type: ignore[union-attr]
                    self.index_lag,
                    self.index_method,
                    value,  # type: ignore[arg-type]
                    date_,
                    NoInput(0),
                )
        else:
            self.periods[0].index_fixings = value  # type: ignore[union-attr]
            for p in [_ for _ in self.periods[1:] if type(_) is not Cashflow]:
                p.index_fixings = NoInput(0)  # type: ignore[union-attr]

    @property
    def index_base(self) -> DualTypes_:
        return self._index_base

    @index_base.setter
    def index_base(self, value: DualTypes | Series[DualTypes] | NoInput) -> None:  # type: ignore[type-var]
        if isinstance(value, Series):
            value = _validate_index_fixings_as_series(value)  # type: ignore[arg-type]
            ret = index_value(
                self.index_lag, self.index_method, value, self.schedule.effective, NoInput(0)
            )
        else:
            ret = value
        self._index_base = ret
        # if value is not None:
        for period in self.periods:
            if isinstance(period, IndexFixedPeriod | IndexCashflow):
                period.index_base = self._index_base

    # def _regular_period(self, *args: Any, **kwargs: Any) -> Period:  # type: ignore[empty-body]
    #     pass  # pragma: no cover


def _validate_index_fixings_as_series(value: Series[DualTypes]) -> Series[DualTypes]:  # type: ignore[type-var]
    if not value.index.is_monotonic_increasing:
        if value.index.is_monotonic_decreasing:
            value = value[::-1]
        else:
            raise ValueError("`index_fixings` as Series must be monotonic increasing.")
    elif not value.index.is_unique:
        raise ValueError("`index_fixings` as Series must be unique index values.")
    return value


class ZeroIndexLeg(_IndexLegMixin, BaseLeg):
    """
    Create a zero coupon index leg composed of a single
    :class:`~rateslib.periods.IndexFixedPeriod` and
    a :class:`~rateslib.periods.Cashflow`.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    index_base : float or None, optional
        The base index applied to all periods.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the first
        period.
        If a datetime indexed ``Series`` will use the fixings that are available in
        that object, and derive the rest from the ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    .. warning::

       Setting ``convention`` for a *ZeroIndexLeg* has no effect because the determination of the
       cashflow within the *IndexFixedPeriod* will always have a DCF of 1.0.

    The fixed rate of the *IndexFixedPeriod* is set to 100% to index up the
    complete the notional. The offsetting *Cashflow* deducts the real notional.

    The NPV of a *ZeroIndexLeg* is the sum of the period NPVs.

    .. math::

       P = - v(m_n) N \\left ( I(m_n) - 1 \\right )

    The analytic delta is defined as zero due to the lack of rates related attributes.

    .. math::

       A = 0

    Examples
    --------
    .. ipython:: python

       index_curve = Curve({dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.95}, index_base=100.0)
       zil = ZeroIndexLeg(
           effective=dt(2022, 1, 15),
           termination="3Y",
           frequency="S",
           index_method="monthly",
           index_base=100.25,
       )
       zil.cashflows(index_curve, curve)

    """

    periods: list[IndexFixedPeriod | Cashflow]  # type: ignore[assignment]
    schedule: Schedule

    def __init__(
        self,
        *args: Any,
        index_base: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_fixings: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.index_method, self.index_lag = _validate_index_method_and_lag(
            _drb(defaults.index_method, index_method), _drb(defaults.index_lag, index_lag)
        )
        super().__init__(*args, **kwargs)
        self.index_fixings = index_fixings  # set index fixings after periods init
        # set after periods initialised
        self.index_base = index_base

    def _regular_period(  # type: ignore[empty-body]
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        stub: bool,
        notional: DualTypes,
        iterator: int,
    ) -> IndexFixedPeriod:
        # set_periods has override
        pass

    def _set_periods(self) -> None:
        self.periods = [
            IndexFixedPeriod(
                fixed_rate=100.0,
                start=self.schedule.effective,
                end=self.schedule.termination,
                payment=self.schedule.pschedule[-1],
                convention="1",
                frequency=self.schedule.frequency,
                notional=self.notional,
                currency=self.currency,
                termination=self.schedule.termination,
                stub=False,
                roll=self.schedule.roll,
                calendar=self.schedule.calendar,
                index_base=self.index_base,
                index_fixings=NoInput(0),  # set during init
                index_lag=self.index_lag,
                index_method=self.index_method,
            ),
            Cashflow(
                notional=-self.notional,
                payment=self.schedule.pschedule[-1],
                currency=self.currency,
                stub_type=NoInput(0),
                rate=NoInput(0),
            ),
        ]

    def cashflow(self, curve: Curve_ = NoInput(0)) -> DualTypes:
        """Aggregate the cashflows on the *IndexFixedPeriod* and *Cashflow* period using a
        *Curve*."""
        _: DualTypes = self.periods[0].cashflow(curve) + self.periods[1].cashflow  # type: ignore[operator]
        return _

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *ZeroIndexLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        cfs = super().cashflows(*args, **kwargs)
        _: DataFrame = cfs.iloc[[0]].copy()
        for attr in ["Cashflow", "NPV", "NPV Ccy"]:
            _[attr] += cfs.iloc[1][attr]
        _["Type"] = "ZeroIndexLeg"
        _["Period"] = None
        return _

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return 0.0

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *ZeroIndexLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)


class IndexFixedLeg(_IndexLegMixin, _FixedLegMixin, BaseLeg):  # type: ignore[misc]
    """
    Create a leg of :class:`~rateslib.periods.IndexFixedPeriod` s and initial and
    final :class:`~rateslib.periods.IndexCashflow` s.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    index_base : float or None, optional
        The base index to determine the cashflow.
    index_fixings : float, list or Series, optional
        If a float scalar, will be applied as the index fixing for the first period.
        If a datetime indexed ``Series``, will use the fixings that are available
        in that object for relevant periods, and derive the rest from the ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       An initial exchange is not currently implemented for this leg.

    The final cashflow notional is set as the notional. The payment date is set equal
    to the final accrual date adjusted by ``payment_lag_exchange``.

    If ``amortization`` is specified an exchanged notional equivalent to the
    amortization amount is added to the list of periods. For similar examples see
    :class:`~rateslib.legs.FloatLeg`.

    The NPV of a *IndexFixedLeg* is the sum of the period NPVs.

    .. math::

       P = - R \\sum_{i=1}^n N_i d_i v(m_i) I(m_i) - \\sum_{i=1}^{n-1}(N_{i}-N_{i+1})v(m_i)I(m_i)  - N_n v(m_n)I(m_n)

    The analytic delta is defined as that of a *FixedLeg*.

    .. math::

       A = \\sum_{i=1}^n N_i d_i v(m_i) I(m_i)

    Examples
    --------

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       index_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_base=100.0)
       index_leg_exch = IndexFixedLeg(
           dt(2022, 1, 1), "9M", "Q",
           notional=1000000,
           amortization=200000,
           index_base=100.0,
           initial_exchange=False,
           final_exchange=True,
           fixed_rate=1.0,
       )
       index_leg_exch.cashflows(index_curve, curve)
       index_leg_exch.npv(index_curve, curve)

    """  # noqa: E501

    periods: list[IndexCashflow | IndexFixedPeriod]  # type: ignore[assignment]
    _regular_periods: tuple[IndexFixedPeriod, ...]

    # TODO: spread calculations to determine the fixed rate on this leg do not work.
    def __init__(
        self,
        *args: Any,
        index_base: DualTypes,
        index_fixings: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_method: str_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._fixed_rate = fixed_rate
        self.index_method, self.index_lag = _validate_index_method_and_lag(
            _drb(defaults.index_method, index_method), _drb(defaults.index_lag, index_lag)
        )
        super().__init__(*args, **kwargs)
        self.index_fixings: DualTypes_ | Series[DualTypes] = (  # type: ignore[type-var]
            index_fixings  # set index fixings after periods init
        )
        self.index_base = index_base  # set after periods initialised

    def _set_exchange_periods(self) -> None:
        """Set default cashflow exchanges on Legs with `initial_exchange` or `final_exchange`."""

        periods_: list[IndexCashflow | None] = [None, None]

        # initial exchange
        if self.initial_exchange:
            raise NotImplementedError(
                "Cannot construct `IndexFixedLeg` with `initial_exchange` "
                "due to not implemented `index_fixings` input argument applicable to "
                "the indexing-up the initial exchange.",
            )
            # self.periods.append(
            #     IndexCashflow(
            #         notional=-self.notional,
            #         payment=add_tenor(
            #             self.schedule.aschedule[0],
            #             f"{self.payment_lag_exchange}B",
            #             None,
            #             self.schedule.calendar,
            #         ),
            #         currency=self.currency,
            #         stub_type="Exchange",
            #         rate=None,
            #         index_base=self.index_base,
            #         index_fixings=self.index_fixings,
            #         index_method=self.index_method,
            #     )
            # )

        # final cashflow
        if self.final_exchange:
            periods_[1] = IndexCashflow(
                notional=self.notional - self.amortization * (self.schedule.n_periods - 1),
                payment=self.schedule.calendar.lag(
                    self.schedule.aschedule[-1],
                    self.payment_lag_exchange,
                    True,
                ),
                currency=self.currency,
                stub_type="Exchange",
                rate=NoInput(0),
                index_base=NoInput(0),  # set during init
                index_fixings=NoInput(0),  # set during init
                index_method=self.index_method,
                index_lag=self.index_lag,
                index_only=False,
            )

        self._exchange_periods: tuple[IndexCashflow | None, IndexCashflow | None] = tuple(periods_)  # type: ignore[assignment]

    def _set_interim_exchange_periods(self) -> None:
        """Set cashflow exchanges if `amortization` and `final_exchange` are present."""
        if not self.final_exchange or self.amortization == 0:
            self._interim_exchange_periods: tuple[IndexCashflow, ...] | None = None
        else:
            periods_ = [
                IndexCashflow(
                    notional=self.amortization,
                    payment=self.schedule.pschedule[1 + i],
                    currency=self.currency,
                    stub_type="Amortization",
                    rate=NoInput(0),
                    index_base=NoInput(0),  # set during init
                    index_fixings=NoInput(0),  # set during init
                    index_method=self.index_method,
                    index_lag=self.index_lag,
                    index_only=False,
                )
                for i in range(self.schedule.n_periods - 1)
            ]
            self._interim_exchange_periods = tuple(periods_)

    def _set_regular_periods(self) -> None:
        self._regular_periods: tuple[IndexFixedPeriod, ...] = tuple(
            [
                IndexFixedPeriod(
                    fixed_rate=self.fixed_rate,
                    start=period[defaults.headers["a_acc_start"]],
                    end=period[defaults.headers["a_acc_end"]],
                    payment=period[defaults.headers["payment"]],
                    notional=self.notional - self.amortization * i,
                    convention=self.convention,
                    currency=self.currency,
                    termination=self.schedule.termination,
                    frequency=self.schedule.frequency,
                    stub=period[defaults.headers["stub_type"]] == "Stub",
                    roll=self.schedule.roll,
                    calendar=self.schedule.calendar,
                    index_method=self.index_method,
                    index_base=NoInput(0),  # set during init
                    index_fixings=NoInput(0),  # set during init
                    index_lag=self.index_lag,
                )
                for i, period in enumerate(self.schedule.table.to_dict(orient="index").values())
            ]
        )

    def _set_periods(self) -> None:
        return BaseLeg._set_periods(self)

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        return super().npv(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        return super().cashflows(*args, **kwargs)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        return super().analytic_delta(*args, **kwargs)
