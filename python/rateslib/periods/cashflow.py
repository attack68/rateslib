from __future__ import annotations

from typing import TYPE_CHECKING, overload

from rateslib import defaults
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.dual.utils import _dual_float
from rateslib.enums import NoInput, _drb
from rateslib.periods.utils import (
    _get_fx_and_base,
    _maybe_local,
    _validate_fx_as_forwards,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        _BaseCurve,
        _BaseCurve_,
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
       :suppress:

       from rateslib import Cashflow

    .. ipython:: python

       cf = Cashflow(
           notional=1e6,
           payment=dt(2022, 8, 3),
           currency="usd",
           stub_type="Loan Payment",
       )
       cf.cashflows(curve=Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.98}))
    """

    _non_deliverable = False

    def __init__(
        self,
        notional: DualTypes,
        payment: datetime,
        currency: str_ = NoInput(0),
        stub_type: str_ = NoInput(0),
        rate: DualTypes_ = NoInput(0),
    ):
        self.notional = notional
        self.payment = payment
        self.currency = _drb(defaults.base_currency, currency).lower()
        self.stub_type = stub_type
        self._rate: DualTypes | NoInput = rate if isinstance(rate, NoInput) else _dual_float(rate)

        # non-deliverable temporary attributes
        self._pair = ""
        self.reference_currency = self.currency

    @property
    def fx_reversed(self) -> bool:
        ret: bool = self.pair[0:3] != self.reference_currency
        return ret

    @property
    def pair(self) -> str:
        if not self._non_deliverable:
            raise TypeError("Cashflow is not 'non-deliverable' and has no FX ``pair``.'")
        else:
            return self._pair

    @property
    def fx_fixing(self) -> DualTypes_:
        if not self._non_deliverable:
            raise TypeError("Cashflow is not 'non-deliverable' and has no ``fx_fixing``.'")
        else:
            return self._fx_fixing

    @fx_fixing.setter
    def fx_fixing(self, value: DualTypes_) -> None:
        if not self._non_deliverable:
            raise TypeError("Cashflow is not 'non-deliverable' and has no ``fx_fixing``.'")
        else:
            self._fx_fixing = value

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    @overload
    def rate(self: NonDeliverableCashflow, fx: FX_) -> DualTypes: ...  # type: ignore[misc]

    @overload
    def rate(self: Cashflow, fx: FX_) -> DualTypes | None: ...

    def rate(self, fx: FX_) -> DualTypes | None:
        """
        Return the associated rate initialised with the *Cashflow*.
        """
        if not self._non_deliverable:
            return None if isinstance(self._rate, NoInput) else self._rate
        else:
            if isinstance(self.fx_fixing, NoInput):
                fx_ = _validate_fx_as_forwards(fx)
                return fx_.rate(self.pair, self.payment)
            else:
                return self.fx_fixing

    def _npv_without_conversion(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        disc_curve_: _BaseCurve = _disc_required_maybe_from_curve(curve, disc_curve)
        value: DualTypes = self.cashflow(curve, fx) * disc_curve_[self.payment]
        return value

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the *Cashflow*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        npv_ = self._npv_without_conversion(curve, disc_curve, fx)
        return _maybe_local(npv_, local, self.currency, fx, base)

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *Cashflow*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: _BaseCurve_ = _disc_maybe_from_curve(curve, disc_curve)
        fx_, _ = _get_fx_and_base(self.currency, fx, base)

        if isinstance(disc_curve_, NoInput):
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            npv = _dual_float(self._npv_without_conversion(curve, disc_curve_, fx=fx))
            npv_fx = npv * _dual_float(fx_)
            df, collateral = _dual_float(disc_curve_[self.payment]), disc_curve_.meta.collateral

        try:
            cashflow_ = _dual_float(self.cashflow(curve=curve, fx=fx))
        except (ValueError, TypeError):  # fx is not valid or super class does not yield cf
            cashflow_ = None

        try:
            rate_: DualTypes | None = self.rate(fx)
        except ValueError:
            rate_ = None

        rate = None if rate_ is None else _dual_float(rate_)
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

    def cashflow(
        self,
        curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> DualTypes:
        """Determine the cashflow amount in the settlement ``currency``."""
        c = -self.notional
        if not self._non_deliverable:
            return c
        else:
            if isinstance(self.fx_fixing, NoInput):
                fx_ = _validate_fx_as_forwards(fx)
                fx_fixing: DualTypes = fx_.rate(self.pair, self.payment)
            else:
                fx_fixing = self.fx_fixing

            c *= fx_fixing if not self.fx_reversed else (1 / fx_fixing)
            return c

    # @property
    # def dcf(self) -> float:
    #     return 0.0

    def analytic_delta(
        self,
        curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *Cashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0


class NonDeliverableCashflow(Cashflow):
    """
    Create a cashflow amount associated with a non-deliverable FX forward.

    Parameters
    ----------
    notional : float, Dual, Dual2
        The notional amount of the cashflow expressed in units of the ``currency``,
    currency : str
        The non-deliverable reference currency (3-digit code), e.g. "brl".
    payment : datetime
        The settlement date of the exchange.
    settlement_currency : str
        The currency of the deliverable currency (3-digit code), e.g. "usd" or "eur".
    fx_fixing: float, Dual, Dual2, optional
        The FX fixing to determine the settlement amount. The reference ``currency`` should be
        the left hand side, e.g. BRLUSD, unless ``reversed``
        in which case should be right hand side, e.g. USDBRL.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = - N f

    where :math:`f` is the ``fx_fixing`` (or derivable FX fixing if ``reversed``) or market forecast
    rate at settlement. This amount is expressed in units of ``settlement_currency``.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined in ``settlement_currency`` terms as;

    .. math::

       P = Cv(m) = - N f v(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0

    Example
    -------
    .. ipython:: python
       :suppress:

       from rateslib import NonDeliverableCashflow

    .. ipython:: python

       ndc = NonDeliverableCashflow(
           notional=10e6,                # <- this is BRL amount
           currency="usd",               # <- this is USD settlement currency
           payment=dt(2025, 6, 1),
           pair="brlusd",                # <- this implies BRL reference currency
       )
       ndc.cashflows()

       ndc = NonDeliverableCashflow(
           notional=2e6,                 # <- this is a BRL amount
           currency="usd",               # <- this is USD settlement currency
           payment=dt(2025, 6, 1),
           pair="usdbrl",                # <- this implies BRL reference currency
       )
       ndc.cashflows()
    """

    _non_deliverable = True

    def __init__(
        self,
        notional: DualTypes,
        currency: str,
        payment: datetime,
        pair: str,
        fx_fixing: DualTypes_ = NoInput(0),
    ):
        super().__init__(
            notional=notional,
            payment=payment,
            currency=currency,
            stub_type=pair.upper(),
        )
        self._pair = pair.lower()
        ccy1, ccy2 = self.pair[0:3], self.pair[3:6]
        self.reference_currency = ccy1 if ccy1 != self.currency else ccy2
        self.fx_fixing = fx_fixing


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
