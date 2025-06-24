from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards
from rateslib.periods.utils import (
    _float_or_none,
    _get_fx_and_base,
    _maybe_local,
    _validate_fx_as_forwards,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        Curve,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        datetime,
        str_,
    )


class Cashflow:
    """
    Create a single cashflow amount on a payment date (effectively a CustomPeriod).

    Parameters
    ----------
    notional : float
        The notional amount of the period (positive assumes paying a cashflow).
    payment : Datetime
        The adjusted payment date of the period.
    currency : str
        The currency of the cashflow (3-digit code).
    stub_type : str
        Record of the type of cashflow.
    rate : float
        An associated rate to relate to the cashflow, e.g. an FX fixing.

    Attributes
    ----------
    notional : float
    payment : Datetime
    stub_type : str

    Notes
    -----
    Other common :class:`BasePeriod` parameters not required for single cashflows are
    set to *None*.

    The ``cashflow`` is defined as follows;

    .. math::

       C = -N

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv(m) = -Nv(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0

    Example
    -------
    .. ipython:: python

       cf = Cashflow(
           notional=1e6,
           payment=dt(2022, 8, 3),
           currency="usd",
           stub_type="Loan Payment",
       )
       cf.cashflows(curve=Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.98}))
    """

    def __init__(
        self,
        notional: DualTypes,
        payment: datetime,
        currency: str_ = NoInput(0),
        stub_type: str_ = NoInput(0),
        rate: DualTypes_ = NoInput(0),
    ):
        self.notional, self.payment = notional, payment
        self.currency = _drb(defaults.base_currency, currency).lower()
        self.stub_type = stub_type
        self._rate: DualTypes | NoInput = rate if isinstance(rate, NoInput) else _dual_float(rate)

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def rate(self) -> DualTypes | None:
        """
        Return the associated rate initialised with the *Cashflow*. Not used for calculations.
        """
        return None if isinstance(self._rate, NoInput) else self._rate

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the *Cashflow*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        value: DualTypes = self.cashflow * disc_curve_[self.payment]
        return _maybe_local(value, local, self.currency, fx, base)

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *Cashflow*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx_, _ = _get_fx_and_base(self.currency, fx, base)

        if isinstance(disc_curve_, NoInput):
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            npv_: DualTypes = self.npv(curve, disc_curve_)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx_)
            df, collateral = _dual_float(disc_curve_[self.payment]), disc_curve_.collateral

        try:
            cashflow_ = _dual_float(self.cashflow)
        except TypeError:  # cashflow in superclass not a property
            cashflow_ = None

        rate = None if isinstance(self.rate(), NoInput) else self.rate()
        stub_type = None if isinstance(self.stub_type, NoInput) else self.stub_type
        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: stub_type,
            defaults.headers["currency"]: self.currency.upper(),
            # defaults.headers["a_acc_start"]: None,
            # defaults.headers["a_acc_end"]: None,
            defaults.headers["payment"]: self.payment,
            # defaults.headers["convention"]: None,
            # defaults.headers["dcf"]: None,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["rate"]: rate,
            # defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow_,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: npv_fx,
            defaults.headers["collateral"]: collateral,
        }

    @property
    def cashflow(self) -> DualTypes:
        return -self.notional

    # @property
    # def dcf(self) -> float:
    #     return 0.0

    def analytic_delta(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *Cashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0


class NonDeliverableCashflow:
    """
    Create a cashflow amount associated with a non-deliverable FX forward.

    Parameters
    ----------
    notional : float, Dual, Dual2
        The notional amount of the cashflow expressed in units of the ``reference_currency``,
        unless ``reversed`` in which case in units of ``settlement_currency``.
    reference_currency : str
        The non-deliverable reference currency (3-digit code).
    settlement_currency : str
        The currency of the deliverable currency (3-digit code), e.g. "usd" or "eur".
    settlement : datetime
        The settlement date of the exchange.
    fixing_date: datetime
        The date on which the FX fixings will be recorded.
    fx_rate: float, Dual, Dual2, optional
        The pricing parameter of the period to record the entry level of the transaction.
        The ``reference_currency`` should be the left hand side, e.g. BRLUSD, unless ``reversed``
        in which case should be right hand side, e.g. USDBRL.
    fx_fixing: float, Dual, Dual2, optional
        The FX fixing to determine the settlement amount, expressed in the same way as ``fx_rate``.
    reversed: bool, optional
        If *True* reverses the FX rate, as shown above.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = N (f_2 - f_1)

    where :math:`f_1` is the ``fx_rate``, :math:`f_2` is the ``fx_fixing`` or market forecast
    rate at settlement. This amount is expressed in units of ``settlement_currency``.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv(m) = N (f_2 - f_1) v(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0

    Example
    -------
    .. ipython:: python

       ndc = NonDeliverableCashflow(
           notional=10e6,  # <- this is BRL amount
           reference_currency="brl",
           settlement_currency="usd",
           settlement=dt(2025, 6, 1),
           fixing_date=dt(2025, 5, 29),
           fx_rate=0.200,  # <- this is BRLUSD FX rate
       )
       ndc.cashflows()

       ndc = NonDeliverableCashflow(
           notional=2e6,  # <- this is USD amount
           reference_currency="brl",
           settlement_currency="usd",
           settlement=dt(2025, 6, 1),
           fixing_date=dt(2025, 5, 29),
           fx_rate=5.00,  # <- this is USDBRL FX rate
           reversed=True,
       )
       ndc.cashflows()
    """

    def __init__(
        self,
        notional: DualTypes,
        reference_currency: str,
        settlement_currency: str,
        settlement: datetime,
        fixing_date: datetime,
        fx_rate: DualTypes_ = NoInput(0),
        fx_fixing: DualTypes_ = NoInput(0),
        reversed: bool = False,  # noqa: A002
    ):
        self.notional = notional
        self.settlement = settlement
        self.settlement_currency = settlement_currency.lower()
        self.reference_currency = reference_currency.lower()
        self.reversed = reversed
        if reversed:
            self.pair = f"{self.settlement_currency}{self.reference_currency}"
        else:
            self.pair = f"{self.reference_currency}{self.settlement_currency}"
        self.fixing_date = fixing_date
        self.fx_rate = fx_rate
        self.fx_fixing = fx_fixing

    def analytic_delta(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *NonDeliverableCashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the *NonDeliverableCashflow*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        disc_cashflow = self.cashflow(fx) * disc_curve_[self.settlement]
        return _maybe_local(disc_cashflow, local, self.settlement_currency, fx, base)

    def cashflow(self, fx: FX_) -> DualTypes:
        """
        Determine the cashflow amount, expressed in ``settlement_currency``.

        Parameters
        ----------
        fx: FXForwards, optional
            Required to forecast the FX rate at settlement, if an ``fx_fixing`` is not known.

        Returns
        -------
        float, Dual, Dual2
        """

        if isinstance(self.fx_fixing, NoInput):
            fx_ = _validate_fx_as_forwards(fx)
            fx_fixing: DualTypes = fx_.rate(self.pair, self.settlement)
        else:
            fx_fixing = self.fx_fixing

        try:
            d_value: DualTypes = self.notional * (fx_fixing - self.fx_rate)  # type: ignore[operator]
        except TypeError as e:
            # either fixed rate is None
            if isinstance(self.fx_rate, NoInput):
                raise TypeError("`fx_rate` must be set on the Period for an `npv`.")
            else:
                raise e

        if self.reversed:
            d_value /= fx_fixing

        return d_value

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *NonDeliverableCashflow*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Curve_ = _disc_maybe_from_curve(curve, disc_curve)
        imm_fx_to_base, _ = _get_fx_and_base(self.settlement_currency, fx, base)

        if isinstance(disc_curve_, NoInput) or not isinstance(fx, FXForwards):
            npv, npv_fx, df, collateral, cashflow = None, None, None, None, None
            index_val_ = None
        else:
            npv_: DualTypes = self.npv(curve, disc_curve_, fx)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            index_val_ = self.rate(fx)

            npv_fx = npv * _dual_float(imm_fx_to_base)
            df, collateral = _dual_float(disc_curve_[self.settlement]), disc_curve_.collateral
            cashflow = _dual_float(self.cashflow(fx))

        rate = _float_or_none(_drb(None, self.fx_rate))
        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: f"{self.pair.upper()}",
            defaults.headers["currency"]: self.settlement_currency.upper(),
            # defaults.headers["a_acc_start"]: None,
            # defaults.headers["a_acc_end"]: None,
            defaults.headers["payment"]: self.settlement,
            # defaults.headers["convention"]: None,
            # defaults.headers["dcf"]: None,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["rate"]: rate,
            defaults.headers["index_value"]: _float_or_none(index_val_),
            # defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(imm_fx_to_base),
            defaults.headers["npv_fx"]: npv_fx,
            defaults.headers["collateral"]: collateral,
        }

    def rate(self, fx: FX_) -> DualTypes:
        if isinstance(self.fx_fixing, NoInput):
            fx_ = _validate_fx_as_forwards(fx)
            return fx_.rate(self.pair, self.settlement)
        else:
            return self.fx_fixing


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
