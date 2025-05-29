from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Series

from rateslib import defaults
from rateslib.curves import index_value
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.periods.cashflow import Cashflow
from rateslib.periods.rates import FixedPeriod
from rateslib.periods.utils import _float_or_none, _get_fx_and_base, _maybe_local

if TYPE_CHECKING:
    from rateslib.typing import FX_, Any, Curve, Curve_, CurveOption_, DualTypes, DualTypes_, str_


class IndexMixin(metaclass=ABCMeta):
    """
    Abstract base class to include methods and properties related to indexed *Periods*.
    """

    index_base: DualTypes_
    index_method: str
    index_fixings: DualTypes | NoInput
    index_lag: int
    index_only: bool = False  # parameter to deduct real notional from cashflow
    payment: datetime
    end: datetime
    currency: str

    @property
    @abstractmethod
    def real_cashflow(self) -> DualTypes | None:
        pass  # pragma: no cover

    def cashflow(self, curve: CurveOption_ = NoInput(0)) -> DualTypes | None:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional,
        adjusted for the index.
        """
        if self.real_cashflow is None:
            return None
        else:
            index_ratio, _, _ = self.index_ratio(curve)
            if index_ratio is None:
                return None
            else:
                if self.index_only:
                    notional_ = -1.0
                else:
                    notional_ = 0.0
                ret: DualTypes = self.real_cashflow * (index_ratio + notional_)
            return ret

    def index_ratio(
        self, curve: CurveOption_ = NoInput(0)
    ) -> tuple[DualTypes | None, DualTypes | None, DualTypes | None]:
        """
        Calculate the index ratio for the end date of the *IndexPeriod*.

        .. math::

           I(m) = \\frac{I_{val}(m)}{I_{base}}

        Parameters
        ----------
        curve : Curve
            The curve from which index values are forecast.

        Returns
        -------
        float, Dual, Dual2
        """
        # IndexCashflow has no start
        i_date_base: datetime | NoInput = getattr(self, "start", NoInput(0))
        denominator_ = index_value(
            index_fixings=self.index_base,
            index_date=i_date_base,
            index_curve=curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )
        numerator_ = index_value(
            index_fixings=self.index_fixings,
            index_date=self.end,
            index_curve=curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )
        numerator = _drb(None, numerator_)
        denominator = _drb(None, denominator_)
        if numerator is None or denominator is None:
            return None, numerator, denominator
        else:
            return numerator / denominator, numerator, denominator

    def npv(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the cashflows of the *IndexPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        cf_: DualTypes | None = self.cashflow(curve)
        if cf_ is None:
            raise ValueError(
                "`cashflow` could not be determined. Is `curve` or `index_fixings` "
                "supplied correctly?"
            )
        value = cf_ * disc_curve_[self.payment]
        return _maybe_local(value, local, self.currency, fx, base)


class IndexFixedPeriod(IndexMixin, FixedPeriod):  # type: ignore[misc]
    """
    Create a period defined with a real rate adjusted by an index.

    When used with an inflation index this defines a real coupon period with a
    cashflow adjusted upwards by the inflation index.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`FixedPeriod`.
    index_base : float or None, optional
        The base index to determine the cashflow.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the settlement, or
        payment, date.
        If a datetime indexed ``Series`` will use the
        fixings that are available in that object, under the appropriate ``index_method``.
    index_method : str, in {'daily', 'monthly', 'curve'}, optional
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month. Defined by default.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword arguments to :class:`FixedPeriod`.

    Notes
    -----
    The ``real_cashflow`` is defined as follows;

    .. math::

       C_{real} = -NdR

    The ``cashflow`` is defined as follows;

    .. math::

       C = C_{real}I(m) = -NdRI(m)

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv = -NdRv(m)I(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial R} = Ndv(m)I(m)

    Examples
    --------
    .. ipython:: python

       ifp = IndexFixedPeriod(
           start=dt(2022, 2, 1),
           end=dt(2022, 8, 1),
           payment=dt(2022, 8, 2),
           frequency="S",
           notional=1e6,
           currency="eur",
           convention="30e360",
           fixed_rate=5.0,
           index_lag=2,
           index_base=100.25,
       )
       ifp.cashflows(
           curve=Curve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.99}, index_base=100.0, index_lag=2),
           disc_curve=Curve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.98})
       )
    """  # noqa: E501

    def __init__(
        self,
        *args: Any,
        index_base: DualTypes | NoInput = NoInput(0),
        index_fixings: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super(IndexMixin, self).__init__(*args, **kwargs)
        self.index_method, self.index_lag = _validate_index_method_and_lag(
            _drb(defaults.index_method, index_method),
            _drb(defaults.index_lag, index_lag),
        )
        self.index_base = _index_series_to_value(
            self.start, self.index_lag, self.index_method, index_base
        )
        self.index_fixings = _index_series_to_value(
            self.end, self.index_lag, self.index_method, index_fixings
        )

    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *IndexFixedPeriod*.
        See :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        real_a_delta = super().analytic_delta(curve, disc_curve, fx, base)
        index_ratio, _, _ = self.index_ratio(curve)
        if index_ratio is None:
            raise ValueError(
                "`index_ratio` is None. Must supply a `curve` or `index_fixings` to "
                "forecast index values."
            )
        return real_a_delta * index_ratio

    @property
    def real_cashflow(self) -> DualTypes | None:
        """
        float, Dual or Dual2 : The calculated real value from rate, dcf and notional.
        """
        if isinstance(self.fixed_rate, NoInput):
            return None
        else:
            return -self.notional * self.dcf * self.fixed_rate / 100

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere

    def cashflows(
        self,
        curve: Curve_ = NoInput(0),  # type: ignore[override]
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *IndexFixedPeriod*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if isinstance(disc_curve_, NoInput) or isinstance(self.fixed_rate, NoInput):
            npv = None
            npv_fx = None
        else:
            npv_: DualTypes = self.npv(curve, disc_curve_)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx)

        index_ratio_, index_val_, index_base_ = self.index_ratio(curve)

        return {
            **super(FixedPeriod, self).cashflows(curve, disc_curve_, fx, base),
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["real_cashflow"]: self.real_cashflow,
            defaults.headers["index_base"]: _float_or_none(index_base_),
            defaults.headers["index_value"]: _float_or_none(index_val_),
            defaults.headers["index_ratio"]: _float_or_none(index_ratio_),
            defaults.headers["cashflow"]: _float_or_none(self.cashflow(curve)),
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the cashflows of the *IndexFixedPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        return super().npv(*args, **kwargs)


class IndexCashflow(IndexMixin, Cashflow):  # type: ignore[misc]
    """
    Create a cashflow defined with a real rate adjusted by an index.

    When used with an inflation index this defines a real redemption with a
    cashflow adjusted upwards by the inflation index.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`Cashflow`.
    index_base : float, optional
        The base index to determine the cashflow. Required but may be set after initialisation.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the settlement, or
        payment, date.
        If a datetime indexed ``Series`` will use the
        fixings that are available in that object, under the appropriate ``index_method``.
    index_method : str in {'daily', 'monthly', 'curve'}, optional
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month. Defined by default.
    index_lag : int
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    index_only : bool
        If *True* deduct the real notional from the cashflow and produce only the
        indexed component.
    end : datetime, optional
        The registered end of a period when the index value is measured. If *None*
        is set equal to the payment date.
    kwargs : dict
        Required keyword arguments to :class:`Cashflow`.

    Notes
    -----
    The ``real_cashflow`` is defined as follows;

    .. math::

       C_{real} = -N

    The ``cashflow`` is defined as follows;

    .. math::

       C = C_{real}I(m) = -NI(m)

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv(m) = -Nv(m)I(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0

    Example
    -------
    .. ipython:: python

       icf = IndexCashflow(
           notional=1e6,
           end=dt(2022, 8, 1),
           payment=dt(2022, 8, 3),
           currency="usd",
           stub_type="Loan Payment",
           index_base=100.25
       )
       icf.cashflows(
           curve=Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.99}, index_base=100.0),
           disc_curve=Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.98}),
       )
    """

    def __init__(
        self,
        *args: Any,
        index_base: DualTypes_,
        index_fixings: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        index_only: bool = False,
        end: datetime | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super(IndexMixin, self).__init__(*args, **kwargs)
        self.end = self.payment if isinstance(end, NoInput) else end
        self.index_method, self.index_lag = _validate_index_method_and_lag(
            _drb(defaults.index_method, index_method),
            _drb(defaults.index_lag, index_lag),
        )
        self.index_base = index_base
        self.index_fixings = _index_series_to_value(
            self.end, self.index_lag, self.index_method, index_fixings
        )
        self.index_only = index_only

    @property
    def real_cashflow(self) -> DualTypes:
        return -self.notional

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *IndexCashflow*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        index_ratio_, index_val_, index_base_ = self.index_ratio(curve)
        return {
            **super(IndexMixin, self).cashflows(curve, disc_curve, fx, base),
            defaults.headers["a_acc_end"]: self.end,
            defaults.headers["real_cashflow"]: self.real_cashflow,
            defaults.headers["index_base"]: _float_or_none(index_base_),
            defaults.headers["index_value"]: _float_or_none(index_val_),
            defaults.headers["index_ratio"]: _float_or_none(index_ratio_),
            defaults.headers["cashflow"]: _float_or_none(self.cashflow(curve)),
        }

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *IndexCashflow*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        return super().npv(*args, **kwargs)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *IndexCashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0


def _validate_index_method_and_lag(index_method: str, index_lag: int) -> tuple[str, int]:
    lower_method = index_method.lower()
    if lower_method not in ["daily", "monthly", "curve"]:
        raise ValueError("`index_method` must be in {'daily', 'monthly', 'curve'}.")
    if index_lag != 0 and lower_method == "curve":
        raise ValueError("`index_lag` must be zero when using `index_method`='curve'.")
    return lower_method, index_lag


def _index_series_to_value(
    index_date: datetime,
    index_lag: int,
    index_method: str,
    index_fixings: DualTypes_ | Series[DualTypes],  # type: ignore[type-var]
) -> DualTypes_:
    if isinstance(index_fixings, Series):
        val = index_value(index_lag, index_method, index_fixings, index_date, NoInput(0))  # type: ignore[arg-type]
    else:
        val = index_fixings
    return val
