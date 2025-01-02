# This module is a dependent of legs.py

"""
.. ipython:: python
   :suppress:

   from rateslib.periods import *
   from rateslib.curves import Curve
   from datetime import datetime as dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
   )
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from math import comb, log
from typing import Any

import numpy as np
from pandas import NA, DataFrame, Index, MultiIndex, Series, concat, isna, notna

from rateslib import defaults
from rateslib.calendars import CalInput, CalTypes, _get_eom, add_tenor, dcf, get_calendar
from rateslib.curves import Curve, average_rate, index_left
from rateslib.default import NoInput, _drb
from rateslib.dual import (
    Dual,
    Dual2,
    DualTypes,
    Number,
    Variable,
    _dual_float,
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
    dual_norm_pdf,
    gradient,
)
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    FXVolObj,
    FXVols,
    _black76,
    _d_plus_min_u,
    _delta_type_constants,
)
from rateslib.solver import newton_1dim, newton_ndim
from rateslib.splines import evaluate

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_fx_and_base(
    currency: str,
    fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
    base: str | NoInput = NoInput(0),
) -> tuple[DualTypes, str | NoInput]:
    """
    From a local currency and potentially FX Objects determine the conversion rate between
    `currency` and `base`. If `base` is not given it is inferred from the FX Objects.
    """
    # TODO these can be removed when no traces of None remain.
    if fx is None:
        raise NotImplementedError("TraceBack for NoInput")  # pragma: no cover
    if base is None:
        raise NotImplementedError("TraceBack for NoInput")  # pragma: no cover

    if isinstance(fx, FXRates | FXForwards):
        base_: str | NoInput = fx.base if isinstance(base, NoInput) else base.lower()
        if base_ == currency:
            fx_: DualTypes = 1.0
        else:
            fx_ = fx.rate(pair=f"{currency}{base_}")
    elif not isinstance(base, NoInput):  # and fx is then a float or None
        base_ = base
        if isinstance(fx, NoInput):
            if base.lower() != currency.lower():
                raise ValueError(
                    f"`base` ({base}) cannot be requested without supplying `fx` as a "
                    "valid FXRates or FXForwards object to convert to "
                    f"currency ({currency}).\n"
                    "If you are using a `Solver` with multi-currency instruments have you "
                    "forgotten to attach the FXForwards in the solver's `fx` argument?",
                )
            fx_ = 1.0
        else:
            if abs(fx - 1.0) < 1e-10:
                pass  # no warning when fx == 1.0
            else:
                warnings.warn(
                    f"`base` ({base}) should not be given when supplying `fx` as numeric "
                    f"since it will not be used.\nIt may also be interpreted as giving "
                    f"wrong results.\nBest practice is to instead supply `fx` as an "
                    f"FXRates (or FXForwards) object.\n"
                    f"Reformulate: [fx={fx}, base='{base}'] -> "
                    f"[fx=FXRates({{'{currency}{base}': {fx}}}), base='{base}'].",
                    UserWarning,
                )
            fx_ = fx
    else:  # base is None and fx is float or None.
        base_ = NoInput(0)
        if isinstance(fx, NoInput):
            fx_ = 1.0
        else:
            if abs(fx - 1.0) < 1e-12:
                pass  # no warning when fx == 1.0
            else:
                warnings.warn(
                    "It is not best practice to provide `fx` as numeric since this can "
                    "cause errors of output when dealing with multi-currency derivatives,\n"
                    "and it also fails to preserve FX rate sensitivity in calculations.\n"
                    "Instead, supply a 'base' currency and use an "
                    "FXRates or FXForwards object.\n"
                    f"Reformulate: [fx={fx}, base=None] -> "
                    f"[fx=FXRates({{'{currency}bas': {fx}}}), base='bas'].",
                    UserWarning,
                )
            fx_ = fx

    return fx_, base_


def _maybe_local(
    value: DualTypes,
    local: bool,
    currency: str,
    fx: float | FXRates | FXForwards | NoInput,
    base: str | NoInput,
) -> dict[str, DualTypes] | DualTypes:
    """
    Return NPVs in scalar form or dict form.
    """
    if local:
        return {currency: value}
    else:
        return _maybe_fx_converted(value, currency, fx, base)


def _maybe_fx_converted(
    value: DualTypes,
    currency: str,
    fx: float | FXRates | FXForwards | NoInput,
    base: str | NoInput,
) -> DualTypes:
    fx_, _ = _get_fx_and_base(currency, fx, base)
    return value * fx_


def _disc_maybe_from_curve(
    curve: Curve | NoInput | dict[str, Curve],
    disc_curve: Curve | NoInput,
) -> Curve | NoInput:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, NoInput):
        if isinstance(curve, dict):
            raise ValueError("`disc_curve` cannot be inferred from a dictionary of curves.")
        elif isinstance(curve, NoInput):
            return NoInput(0)
        elif curve._base_type == "values":
            raise ValueError("`disc_curve` cannot be inferred from a non-DF based curve.")
        _: Curve | NoInput = curve
    else:
        _ = disc_curve
    return _


def _disc_required_maybe_from_curve(
    curve: Curve | NoInput | dict[str, Curve],
    disc_curve: Curve | NoInput,
) -> Curve:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    _: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
    if isinstance(_, NoInput):
        raise TypeError(
            "`curves` have not been supplied correctly. "
            "A `disc_curve` is required to perform function."
        )
    return _


class BasePeriod(metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``Period`` subclasses.

    See also: :ref:`User guide for Periods <periods-doc>`.

    Parameters
    ----------
    start : Datetime
        The adjusted start date of the calculation period.
    end: Datetime
        The adjusted end date of the calculation period.
    payment : Datetime
        The adjusted payment date of the period.
    frequency : str
        The frequency of the corresponding leg. Also used
        with specific values for ``convention``, or floating rate calculation.
    notional : float, optional, set by Default
        The notional amount of the period (positive implies paying a cashflow).
    currency : str, optional
        The currency of the cashflow (3-digit code), set by default.
    convention : str, optional, set by Default
        The day count convention of the calculation period accrual.
        See :meth:`~rateslib.scheduling.dcf`.
    termination : Datetime, optional
        The termination date of the corresponding leg. Required only with
        specific values for ``convention``.
    stub : bool, optional
        Records whether the period is a stub or regular. Used by certain day count
        convention calculations.
    roll : int, str, optional
        Used only by ``stub`` periods and for specific values of ``convention``.
    calendar : CustomBusinessDay, str, optional
        Used only by ``stub`` periods and for specific values of ``convention``.

    """

    @abstractmethod
    def __init__(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        frequency: str,
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        termination: datetime | NoInput = NoInput(0),
        stub: bool = False,
        roll: int | str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
    ):
        if end < start:
            raise ValueError("`end` cannot be before `start`.")
        self.start, self.end, self.payment = start, end, payment
        self.frequency: str = frequency.upper()
        self.notional: float = _drb(defaults.notional, notional)
        self.currency: str = _drb(defaults.base_currency, currency).lower()
        self.convention: str = _drb(defaults.convention, convention)
        self.termination = termination
        self.freq_months = defaults.frequency_months[self.frequency]
        self.stub: bool = stub
        self.roll: int | str | NoInput = roll
        self.calendar: CalInput = calendar

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def __str__(self) -> str:
        return (
            f"<{type(self).__name__}: {self.start.strftime('%Y-%m-%d')}->"
            f"{self.end.strftime('%Y-%m-%d')},{self.notional},{self.convention}>"
        )

    @property
    def dcf(self) -> float:
        """
        float : Calculated with appropriate ``convention`` over the period.
        """
        return dcf(
            self.start,
            self.end,
            self.convention,
            self.termination,
            self.freq_months,
            self.stub,
            self.roll,
            self.calendar,
        )

    @abstractmethod
    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the period object.

        Parameters
        ----------
        curve : Curve
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        .. ipython:: python

           curve = Curve({dt(2021,1,1): 1.00, dt(2025,1,1): 0.83}, interpolation="log_linear", id="SONIA")
           fxr = FXRates({"gbpusd": 1.25}, base="usd")

        .. ipython:: python

           period = FixedPeriod(
               start=dt(2022, 1, 1),
               end=dt(2022, 7, 1),
               payment=dt(2022, 7, 1),
               frequency="S",
               currency="gbp",
               fixed_rate=4.00,
           )
           period.analytic_delta(curve, curve)
           period.analytic_delta(curve, curve, fxr)
           period.analytic_delta(curve, curve, fxr, "gbp")
        """  # noqa: E501
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx_, _ = _get_fx_and_base(self.currency, fx, base)
        ret: DualTypes = fx_ * self.notional * self.dcf * disc_curve_[self.payment] / 10000
        return ret

    @abstractmethod
    def cashflows(
        self,
        curve: Curve | dict[str, Curve] | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the properties of the period used in calculating cashflows.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults to
            ``fx.base``.

        Returns
        -------
        dict

        Examples
        --------
        .. ipython:: python

           period.cashflows(curve, curve, fxr)
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        if isinstance(disc_curve_, NoInput):
            df: float | None = None
            collateral: str | None = None
        else:
            df = _dual_float(disc_curve_[self.payment])
            collateral = disc_curve_.collateral

        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: "Stub" if self.stub else "Regular",
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["a_acc_start"]: self.start,
            defaults.headers["a_acc_end"]: self.end,
            defaults.headers["payment"]: self.payment,
            defaults.headers["convention"]: self.convention,
            defaults.headers["dcf"]: self.dcf,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["collateral"]: collateral,
        }

    @abstractmethod
    def npv(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the period object.

        Calculates the cashflow for the period and multiplies it by the DF associated
        with the payment date.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object. Not used unless it is set equal to
            ``disc_curve``, or if a rate in a :class:`FloatPeriod` is required.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards`
            object, converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore the ``base`` request and return a dict identifying
            local currency NPV.

        Returns
        -------
        float, Dual, Dual2, or dict of such

        Examples
        --------
        .. ipython:: python

           period.npv(curve, curve)
           period.npv(curve, curve, fxr)
           period.npv(curve, curve, fxr, "gbp")
           period.npv(curve, curve, fxr, local=True)
        """
        pass  # pragma: no cover


class FixedPeriod(BasePeriod):
    """
    Create a period defined with a fixed rate.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    fixed_rate : float or None, optional
        The rate applied to determine the cashflow. If `None`, can be set later,
        typically after a mid-market rate for all periods has been calculated.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = -NdR

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv = -NdRv(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial R} = Ndv(m)

    Examples
    --------
    .. ipython:: python

       fp = FixedPeriod(
           start=dt(2022, 2, 1),
           end=dt(2022, 8, 1),
           payment=dt(2022, 8, 2),
           frequency="S",
           notional=1e6,
           currency="eur",
           convention="30e360",
           fixed_rate=5.0,
       )
       fp.cashflows(curve=Curve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.98}))

    """

    def __init__(self, *args: Any, fixed_rate: float | NoInput = NoInput(0), **kwargs: Any) -> None:
        self.fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *FixedPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return super().analytic_delta(*args, **kwargs)

    @property
    def cashflow(self) -> DualTypes | None:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional.
        """
        if isinstance(self.fixed_rate, NoInput):
            return None
        else:
            _: DualTypes = -self.notional * self.dcf * self.fixed_rate / 100
            return _

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def npv(
        self,
        curve: Curve | dict[str, Curve] | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> dict[str, DualTypes] | DualTypes:
        """
        Return the NPV of the *FixedPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        try:
            value: DualTypes = self.cashflow * disc_curve_[self.payment]  # type: ignore[operator]
        except TypeError as e:
            # either fixed rate is None
            if isinstance(self.fixed_rate, NoInput):
                raise TypeError("`fixed_rate` must be set on the Period for an `npv`.")
            else:
                raise e
        return _maybe_local(value, local, self.currency, fx, base)

    def cashflows(
        self,
        curve: Curve | dict[str, Curve] | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *FixedPeriod*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx_, base_ = _get_fx_and_base(self.currency, fx, base)

        if isinstance(disc_curve_, NoInput) or isinstance(self.fixed_rate, NoInput):
            npv = None
            npv_fx = None
        else:
            npv_dual: DualTypes = self.npv(curve, disc_curve_, local=False)  # type: ignore[assignment]
            npv = _dual_float(npv_dual)
            npv_fx = npv * _dual_float(fx_)

        cashflow = None if self.cashflow is None else _dual_float(self.cashflow)
        return {
            **super().cashflows(curve, disc_curve_, fx_, base_),
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: npv_fx,
        }


def _validate_float_args(
    fixing_method: str | NoInput,
    method_param: int | NoInput,
    spread_compound_method: str | NoInput,
) -> tuple[str, int, str]:
    """
    Validate the argument input to float periods.

    Returns
    -------
    tuple
    """
    fixing_method_: str = _drb(defaults.fixing_method, fixing_method).lower()
    if fixing_method_ not in [
        "ibor",
        "rfr_payment_delay",
        "rfr_observation_shift",
        "rfr_lockout",
        "rfr_lookback",
        "rfr_payment_delay_avg",
        "rfr_observation_shift_avg",
        "rfr_lockout_avg",
        "rfr_lookback_avg",
    ]:
        raise ValueError(
            "`fixing_method` must be in {'rfr_payment_delay', "
            "'rfr_observation_shift', 'rfr_lockout', 'rfr_lookback', 'ibor'}, "
            f"got '{fixing_method_}'.",
        )

    method_param_: int = _drb(defaults.fixing_method_param[fixing_method_], method_param)
    if method_param_ != 0 and fixing_method_ == "rfr_payment_delay":
        raise ValueError(
            "`method_param` should not be used (or a value other than 0) when "
            f"using a `fixing_method` of 'rfr_payment_delay', got {method_param_}. "
            f"Configure the `payment_lag` option instead to have the "
            f"appropriate effect.",
        )
    elif fixing_method_ == "rfr_lockout" and method_param_ < 1:
        raise ValueError(
            f'`method_param` must be >0 for "rfr_lockout" `fixing_method`, ' f"got {method_param_}",
        )

    spread_compound_method_: str = _drb(
        defaults.spread_compound_method, spread_compound_method
    ).lower()
    if spread_compound_method_ not in [
        "none_simple",
        "isda_compounding",
        "isda_flat_compounding",
    ]:
        raise ValueError(
            "`spread_compound_method` must be in {'none_simple', "
            "'isda_compounding', 'isda_flat_compounding'}, "
            f"got {spread_compound_method_}",
        )
    return fixing_method_, method_param_, spread_compound_method_


class FloatPeriod(BasePeriod):
    """
    Create a period defined with a floating rate index.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    float_spread : float or None, optional
        The float spread applied to determine the cashflow. Can be set to `None` and
        set later, typically after a mid-market float spread for all periods
        has been calculated. **Expressed in basis points (bps).**
    spread_compound_method : str, optional
        The method to use for adding a floating rate to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, or Series, optional
        If a float scalar, will be applied as the determined fixing for the whole
        period. If a list of *n* fixings will be used as the first *n* RFR fixings
        in the period and the remaining fixings will be forecast from the curve and
        composed into the overall rate. If a datetime indexed ``Series`` will use the
        fixings that are available in that object, and derive the rest from the
        ``curve``. **Must be input excluding** ``float_spread``.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = -Ndr(r_i, z)

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv(m) = -Ndr(r_i, z)v(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial z} = Ndv(m) \\frac{\\partial r}{\\partial z}

    **Fixing Methods**

    Floating period rates depend on different ``fixing_method`` to determine their
    rates. For further info see
    :download:`ISDA RFR Compounding Memo (2006)<_static/isda-memo-rfrs-2006.pdf>`.
    The available options provided here are:

    - **"rfr_payment_delay"**: this is the standard convention adopted by interbank
      RFR derivative trades, such as SOFR, SONIA, and ESTR OIS etc. ``method_param``
      is not used for this method and defaults to zero, ``payment_lag`` serves as the
      appropriate parameter for this method.
    - **"rfr_observation_shift"**: typical conventions of FRNs. The ``method_param``
      is the integer number of business days by which both the observation
      rates and the DCFs are shifted.
    - **"rfr_lockout"**: this is a convention typically used on floating rate notes
      (FRNs), the ``method_param`` as integer number of business days is
      the number of locked-out days. E.g. SOFR based FRNs generally have 4.
    - **"rfr_lookback"**: this is also a convention typically used on FRNs. The
      ``method_param`` as integer number of business days defines the
      observation offset, the DCFs remain static, measured between the start and end
      dates.
    - **"rfr_payment_delay_avg", "rfr_observation_shift_avg", "rfr_lockout_avg",
      "rfr_lookback_avg"**: these are the same as the previous conventions except that
      the period rate is defined as the arithmetic average of the individual fixings,
      weighted by the relevant DCF depending upon the method.
    - **"ibor"**: this the convention for determining IBOR rates from a curve. The
      ``method_param`` is the number of fixing lag days before the accrual start when
      the fixing is published. For example, Euribor or Stibor have 2.

    The first two are the only methods recommended by Alternative Reference Rates
    Comittee (AARC), although other methods have been implemented in
    financial instruments previously.

    **Spread Compounding Methods**

    The spread compounding methods operate as follows:

    - **"none_simple"**: the float spread added in a simple way to the determined
      compounded period rate.
    - **"isda_compounding"**: the float spread is added to each individual rate
      and then everything is compounded.
    - **"isda_flat_compounding"**: the spread is added to each rate but is not used
      when compounding each previously calculated component.

    The first is the most efficient and most encountered. The second and third are
    rarely encountered in modern financial instruments.
    For further info see
    :download:`ISDA Compounding Memo (2009)<_static/isda-compounding-memo.pdf>`.

    .. _float fixings:

    **Fixings**

    .. warning::

       Providing ``fixings`` as a ``Series`` is **best practice**.

       But, **RFR** and **IBOR** fixings provided as datetime indexed ``Series`` require
       **different formats**:

       - IBOR fixings are indexed by publication date and fixing value.
       - RFR fixings are indexed by reference value date and fixing value.

    If an *"ibor"* ``fixing
    method`` is given the series should index the published IBOR rates by
    **publication date**, which usually lags the reference value dates.
    For example, EURIBOR lags its
    value dates by two business days. 3M EURIBOR was published on Thu-2-Mar-2023 as
    2.801%, which is applicable to the start date of Mon-6-Mar-2023 with value end
    date of Tue-6-Jun-2023.

    .. ipython:: python

       ibor_curve = Curve(
           nodes={dt(2023, 3, 6): 1.0, dt(2024, 3, 6): 0.96},
           calendar="bus"
       )
       fixings = Series(
           [1.00, 2.801, 1.00, 1.00],
           index=[dt(2023, 3, 1), dt(2023, 3, 2), dt(2023, 3, 3), dt(2023, 3, 6)]
       )
       float_period = FloatPeriod(
           start=dt(2023, 3, 6),
           end=dt(2023, 6, 6),
           payment=dt(2023, 6, 6),
           frequency="Q",
           fixing_method="ibor",
           method_param=2,
           fixings=fixings
       )
       float_period.rate(ibor_curve)  # this will return the fixing published 2-Mar-23
       float_period.fixings_table(ibor_curve)

    RFR rates tend to be maintained by central banks. The modern tendency seems to be
    to present historical RFR data indexed by **reference value date** and not
    publication date, which is usually 1 business day in arrears. If the
    ``fixing_method`` is *"rfr"* based then the given series should be indexed in a
    similar manner. ESTR was published as 2.399% on Fri-3-Mar-2023 for the reference
    value start date of Thu-2-Mar-2023 (and end date of Fri-3-Mar-2023).

    .. ipython:: python

       rfr_curve = Curve(
           nodes={dt(2023, 3, 3): 1.0, dt(2024, 3, 3): 0.96},
           calendar="bus"
       )
       fixings = Series(
           [1.00, 1.00, 2.399],
           index=[dt(2023, 2, 28), dt(2023, 3, 1), dt(2023, 3, 2)]
       )
       float_period = FloatPeriod(
           start=dt(2023, 3, 2),
           end=dt(2023, 3, 3),
           payment=dt(2023, 3, 3),
           frequency="A",
           fixing_method="rfr_payment_delay",
           fixings=fixings
       )
       float_period.rate(rfr_curve)  # this will return the fixing for reference 2-Mar-23
       float_period.fixings_table(rfr_curve)

    Examples
    --------
    Create a stepped (log-linear interpolated) curve with 2 relevant steps:

    .. ipython:: python

       curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999, dt(2022, 3, 1): 0.997})
       curve.rate(dt(2022, 1, 1), "1D")
       curve.rate(dt(2022, 2, 1), "1D")

    A standard `"rfr_payment_delay"` period, which is the default for this library.

    .. ipython:: python

       period = FloatPeriod(
           start=dt(2022, 1, 1),
           end=dt(2022, 3, 1),
           payment=dt(2022, 3, 1),
           frequency="M",
       )
       period.fixing_method
       period.rate(curve)

    An `"rfr_lockout"` period, here with 28 business days lockout (under a curve
    with a no holidays) meaning the observation period
    ends on the 1st Feb 2022 and the 1D rate between 31st Jan 2022 and 1st Feb is used
    consistently as the fixing for the remaining fixing dates.

    .. ipython:: python

       period = FloatPeriod(
           start=dt(2022, 1, 1),
           end=dt(2022, 3, 1),
           payment=dt(2022, 3, 1),
           frequency="M",
           fixing_method="rfr_lockout",
           method_param=28,
       )
       period.rate(curve)

    An `"rfr_lookback"` period, here with 5 days offset meaning the observation period
    starts on 27th Jan and ends on 6th Feb.

    .. ipython:: python

       period = FloatPeriod(
           start=dt(2022, 2, 1),
           end=dt(2022, 2, 11),
           payment=dt(2022, 2, 11),
           frequency="M",
           fixing_method="rfr_lookback",
           method_param=5,
       )
       period.rate(curve)

    An `"ibor"` period, with a 2 day fixing lag

    .. ipython:: python
       :okexcept:

       period = FloatPeriod(
           start=dt(2022, 1, 3),
           end=dt(2022, 3, 3),
           payment=dt(2022, 3, 3),
           frequency="B",
           fixing_method="ibor",
           method_param=2,
       )
       period.rate(curve)
       curve.rate(dt(2022, 1, 3), "2M")
    """

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes | NoInput = NoInput(0),
        fixings: float | list[float] | Series[float] | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.float_spread: DualTypes = _drb(0.0, float_spread)

        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        self.fixings = fixings
        if isinstance(self.fixings, list) and self.fixing_method == "ibor":
            raise ValueError("`fixings` cannot be supplied as list, under 'ibor' `fixing_method`.")

        # self.calendar = get_calendar(calendar)
        # self.modifier = modifier
        super().__init__(*args, **kwargs)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *FloatPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        if self.spread_compound_method == "none_simple" or self.float_spread == 0:
            # then analytic_delta is not impacted by float_spread compounding
            dr_dz: float = 1.0
        elif isinstance(curve, Curve):
            _ = self.float_spread
            DualType: type[Dual] | type[Dual2] = Dual if curve.ad in [0, 1] else Dual2
            DualArgs: tuple[list[Any]] | tuple[list[Any], list[Any]] = (
                ([],) if curve.ad in [0, 1] else ([], [])
            )
            self.float_spread = DualType(_dual_float(_), ["z_float_spread"], *DualArgs)
            rate: Dual | Dual2 = self.rate(curve)  # type: ignore[assignment]
            dr_dz = gradient(rate, ["z_float_spread"])[0] * 100
            self.float_spread = _
        else:
            raise TypeError("`curve` must be supplied for given `spread_compound_method`")

        return dr_dz * super().analytic_delta(curve, disc_curve, fx, base)

    def cashflows(
        self,
        curve: Curve | dict[str, Curve] | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *FloatPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx_, base_ = _get_fx_and_base(self.currency, fx, base)
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)

        try:
            cashflow = self.cashflow(curve)
        except ValueError:
            # curve may not have been provided or other fixings errors have occured
            cashflow, rate = None, None
        else:
            if cashflow is None:
                rate = None
            else:
                rate = 100 * cashflow / (-self.notional * self.dcf)

        if not isinstance(disc_curve_, NoInput):
            npv_: DualTypes | None = self.npv(curve, disc_curve_)  # type: ignore[assignment]
            npv_fx_: DualTypes | None = npv_ * _dual_float(fx_)  # type: ignore[operator]
        else:
            npv_, npv_fx_ = None, None

        return {
            **super().cashflows(curve, disc_curve_, fx_, base_),
            defaults.headers["rate"]: _float_or_none(rate),
            defaults.headers["spread"]: _dual_float(self.float_spread),
            defaults.headers["cashflow"]: _float_or_none(cashflow),
            defaults.headers["npv"]: _float_or_none(npv_),
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: _float_or_none(npv_fx_),
        }

    def npv(
        self,
        curve: Curve | dict[str, Curve] | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> dict[str, DualTypes] | DualTypes:
        """
        Return the NPV of the *FloatPeriod*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        if not isinstance(disc_curve_, Curve):
            raise TypeError("`curves` have not been supplied correctly.")
        if self.payment < disc_curve_.node_dates[0]:
            if local:
                return {self.currency: 0.0}
            else:
                return 0.0  # payment date is in the past avoid issues with fixings or rates
        value = self.rate(curve) / 100 * self.dcf * disc_curve_[self.payment] * -self.notional

        return _maybe_local(value, local, self.currency, fx, base)

    def cashflow(self, curve: Curve | dict[str, Curve] | NoInput = NoInput(0)) -> DualTypes | None:
        """
        Forecast the *Period* cashflow based on a *Curve* providing index rates.

        Parameters
        ----------
        curve : Curve, LineCurve, dict of such keyed by string tenor
            The forecast *Curve* for calculating the *Period* rate. If not given returns *None*.

        Returns
        -------
        float, Dual, Dual2, None

        """
        curve = NoInput(0) if curve is None else curve  # backwards compat
        try:
            _: DualTypes = -self.notional * self.dcf * self.rate(curve) / 100
            return _
        except ValueError as e:
            if isinstance(curve, Curve | dict):
                raise e
            # probably "needs a `curve` to forecast rate
            return None

    def _maybe_get_cal_and_conv_from_curve(
        self, curve: Curve | dict[str, Curve] | NoInput
    ) -> tuple[CalTypes, str]:
        if isinstance(curve, NoInput):
            cal_: CalTypes = get_calendar(self.calendar)
            conv_: str = self.convention
            warnings.warn(
                "A `curve` has not been supplied to FloatPeriod.rate().\n"
                "For 'ibor' method a `calendar` is required to determine the fixing date.\n"
                "For 'rfr' methods a `calendar` and `convention` are required to determine "
                "fixing dates and compounding formulae.\n"
                "In this case these values have been set to use the FloatPeriod's `calendar` and "
                "`convention` as fallbacks to those usually provided by the Curve.",
                UserWarning,
            )
        else:
            if isinstance(curve, dict):
                cal_ = list(curve.values())[0].calendar
                conv_ = list(curve.values())[0].convention
            else:
                cal_ = curve.calendar
                conv_ = curve.convention
        return cal_, conv_

    def rate(self, curve: Curve | dict[str, Curve] | NoInput = NoInput(0)) -> DualTypes:
        """
        Calculating the floating rate for the period.

        Parameters
        ----------
        curve : Curve, LineCurve, dict of curves, optional
            The forecasting curve object. If ``fixings`` are available on the Period may be able
            to return a value without, otherwise will raise.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        Using a single forecast *Curve*.

        .. ipython:: python

           period.rate(curve)

        Using a dict of *Curves* for stub periods calculable under an *"ibor"* ``fixing_method``.

        .. ipython:: python

           period.rate({"1m": curve, "3m": curve, "6m": curve, "12m": curve})
        """
        if "ibor" in self.fixing_method:
            return self._rate_ibor(curve)
        elif "rfr" in self.fixing_method:
            if isinstance(curve, dict):
                curve_: Curve | NoInput = _get_rfr_curve_from_dict(curve)
            else:
                curve_ = curve
            return self._rate_rfr(curve_)
        else:
            raise ValueError(  # pragma: no cover
                f"`fixing_method`: '{self.fixing_method}' not valid for a FloatPeriod."
            )

    def _rate_ibor(self, curve: Curve | dict[str, Curve] | NoInput) -> DualTypes:
        # function will try to forecast a rate without a `curve` when fixings are available.
        if isinstance(self.fixings, float | Dual | Dual2 | Variable):
            return self.fixings + self.float_spread / 100
        elif isinstance(self.fixings, Series):
            # check if we return published IBOR rate
            cal_, _ = self._maybe_get_cal_and_conv_from_curve(curve)
            fixing_date = cal_.lag(self.start, -self.method_param, False)
            try:
                return self.fixings[fixing_date] + self.float_spread / 100
            except KeyError:
                warnings.warn(
                    "A FloatPeriod `fixing date` was not found in the given `fixings` Series.\n"
                    "Using the fallback method of forecasting the fixing from any given `curve`.",
                    UserWarning,
                )
                # fixing not available: use curve
                pass
        elif isinstance(self.fixings, list):  # this is also validated in __init__
            raise ValueError("`fixings` cannot be supplied as list, under 'ibor' `fixing_method`.")

        method = {
            "dfs": self._rate_ibor_from_df_curve,
            "values": self._rate_ibor_from_line_curve,
        }
        if isinstance(curve, NoInput):
            raise ValueError(
                "Must supply a `curve` to FloatPeriod.rate() for forecasting IBOR rates."
            )
        elif not isinstance(curve, dict):
            # TODO (low); this doesnt type well because of floating _base_type attribute.
            # consider wrapping to some NamedTuple record
            return method[curve._base_type](curve)
        else:
            if not self.stub:
                curve = _get_ibor_curve_from_dict(self.freq_months, curve)
                return method[curve._base_type](curve)
            else:
                return self._rate_ibor_interpolated_ibor_from_dict(curve)

    def _rate_ibor_from_df_curve(self, curve: Curve) -> DualTypes:
        if self.stub:
            r = curve._rate_with_raise(self.start, self.end) + self.float_spread / 100
        else:
            r = curve._rate_with_raise(self.start, f"{self.freq_months}m") + self.float_spread / 100
        return r

    def _rate_ibor_from_line_curve(self, curve: Curve) -> DualTypes:
        fixing_date = curve.calendar.lag(self.start, -self.method_param, False)
        return curve[fixing_date] + self.float_spread / 100

    def _rate_ibor_interpolated_ibor_from_dict(self, curve: dict[str, Curve]) -> DualTypes:
        """
        Get the rate on all available curves in dict and then determine the ones to interpolate.
        """
        calendar = next(iter(curve.values())).calendar  # note: ASSUMES all curve calendars are same
        fixing_date = add_tenor(self.start, f"-{self.method_param}B", "NONE", calendar)

        def _rate(c: Curve, tenor: str) -> DualTypes:
            if c._base_type == "dfs":
                return c._rate_with_raise(self.start, tenor)
            else:  # values
                return c._rate_with_raise(fixing_date, tenor)  # tenor is not used on LineCurve

        values = {add_tenor(self.start, k, "MF", calendar): _rate(v, k) for k, v in curve.items()}
        values = dict(sorted(values.items()))
        dates, rates = list(values.keys()), list(values.values())
        if self.end > dates[-1]:
            warnings.warn(
                "Interpolated stub period has a length longer than the provided "
                "IBOR curve tenors: using the longest IBOR value.",
                UserWarning,
            )
            return rates[-1]
        elif self.end < dates[0]:
            warnings.warn(
                "Interpolated stub period has a length shorter than the provided "
                "IBOR curve tenors: using the shortest IBOR value.",
                UserWarning,
            )
            return rates[0]
        else:
            i = index_left(dates, len(dates), self.end)
            _: DualTypes = rates[i] + (rates[i + 1] - rates[i]) * (
                (self.end - dates[i]) / (dates[i + 1] - dates[i])
            )
            return _

    def _rate_rfr(self, curve: Curve | NoInput) -> DualTypes:
        if isinstance(self.fixings, float | Dual | Dual2):
            # if fixings is a single value then return that value (curve unused can be NoInput)
            if (
                self.spread_compound_method in ["isda_compounding", "isda_flat_compounding"]
                and self.float_spread != 0
            ):
                warnings.warn(
                    "Unless the RFR period is a 1-day tenor, "
                    "a single scalar fixing will not be compounded correctly"
                    "with the given `spread_compound_method`, and "
                    "`float_spread`",
                    UserWarning,
                )

            # this ignores spread_compound_type
            return self.fixings + self.float_spread / 100

            # else next calculations made based on fixings in (None, list, Series)
        elif isinstance(self.fixings, Series | list) or isinstance(curve, NoInput):
            # try to calculate rate purely from the fixings
            return self._rfr_rate_from_individual_fixings(curve)
        else:
            method = {
                "dfs": self._rate_rfr_from_df_curve,
                "values": self._rate_rfr_from_line_curve,
            }
            return method[curve._base_type](curve)

    def _rate_rfr_from_df_curve(self, curve: Curve) -> DualTypes:
        if isinstance(curve, NoInput):
            # then attempt to get rate from fixings
            return self._rfr_rate_from_individual_fixings(curve)
        elif self.start < curve.node_dates[0]:
            # then likely fixing are required and curve does not have available data
            return self._rfr_rate_from_individual_fixings(curve)
        elif len(curve.node_dates) == 0:
            # TODO zero len curve is generated by pseudo curve in FloatRateNote.
            # This is bad construct
            return self._rfr_rate_from_individual_fixings(curve)
        elif self.fixing_method == "rfr_payment_delay" and not self._is_inefficient:
            return curve._rate_with_raise(self.start, self.end) + self.float_spread / 100
        elif self.fixing_method == "rfr_observation_shift" and not self._is_inefficient:
            start = curve.calendar.lag(self.start, -self.method_param, settlement=False)
            end = curve.calendar.lag(self.end, -self.method_param, settlement=False)
            return curve._rate_with_raise(start, end) + self.float_spread / 100
            # TODO: (low:perf) semi-efficient method for lockout under certain conditions
        else:
            # return inefficient calculation
            # this is also the path for all averaging methods
            return self._rfr_rate_from_individual_fixings(curve)

    def _rate_rfr_from_line_curve(self, curve: Curve) -> DualTypes:
        return self._rfr_rate_from_individual_fixings(curve)

    def _rate_rfr_avg_with_spread(
        self,
        rates: np.ndarray[tuple[int], np.dtype[np.object_]],
        dcf_vals: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> DualTypes:
        """
        Calculate all in rate with float spread under averaging.

        Parameters
        ----------
        rates : Series
            The rates which are expected for each daily period.
        dcf_vals : Series
            The weightings which are used for each rate in the compounding formula.

        Returns
        -------
        float, Dual, Dual2
        """
        # dcf_vals = dcf_vals.set_axis(rates.index)
        if self.spread_compound_method != "none_simple":
            raise ValueError(
                "`spread_compound` method must be 'none_simple' in an RFR averaging " "period.",
            )
        else:
            _: DualTypes = (dcf_vals * rates).sum() / dcf_vals.sum() + self.float_spread / 100
            return _

    def _rate_rfr_isda_compounded_with_spread(
        self,
        rates: np.ndarray[tuple[int], np.dtype[np.object_]],
        dcf_vals: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> DualTypes:
        """
        Calculate all in rates with float spread under different compounding methods.

        Parameters
        ----------
        rates : Series
            The rates which are expected for each daily period.
        dcf_vals : Series
            The weightings which are used for each rate in the compounding formula.

        Returns
        -------
        float, Dual, Dual2
        """
        # dcf_vals = dcf_vals.set_axis(rates.index)
        if self.float_spread == 0 or self.spread_compound_method == "none_simple":
            _: DualTypes = (
                (1 + dcf_vals * rates / 100).prod() - 1
            ) * 100 / dcf_vals.sum() + self.float_spread / 100
            return _
        elif self.spread_compound_method == "isda_compounding":
            _ = (
                ((1 + dcf_vals * (rates / 100 + self.float_spread / 10000)).prod() - 1)
                * 100
                / dcf_vals.sum()
            )
            return _
        elif self.spread_compound_method == "isda_flat_compounding":
            sub_cashflows = (rates / 100 + self.float_spread / 10000) * dcf_vals
            C_i = 0.0
            for i in range(1, len(sub_cashflows)):
                C_i += sub_cashflows[i - 1]
                sub_cashflows[i] += C_i * rates[i] / 100 * dcf_vals[i]
            _ = sub_cashflows.sum() * 100 / dcf_vals.sum()
            return _
        else:
            # this path not generally hit due to validation at initialisation
            raise ValueError(
                "`spread_compound_method` must be in {'none_simple', "
                "'isda_compounding', 'isda_flat_compounding'}.",
            )

    def _rfr_rate_from_individual_fixings(self, curve: Curve | NoInput) -> DualTypes:
        cal_, conv_ = self._maybe_get_cal_and_conv_from_curve(curve)

        data = self._rfr_get_individual_fixings_data(cal_, conv_, curve)
        if "avg" in self.fixing_method:
            rate = self._rate_rfr_avg_with_spread(
                data["rates"].to_numpy(), data["dcf_vals"].to_numpy()
            )
        else:
            rate = self._rate_rfr_isda_compounded_with_spread(
                data["rates"].to_numpy(), data["dcf_vals"].to_numpy()
            )
        return rate

    def _rfr_get_series_with_populated_fixings(
        self, obs_dates: Series[datetime]
    ) -> Series[DualTypes | None]:  # type: ignore[type-var]
        """
        Gets relevant DCF values and populates all the individual RFR fixings either known or
        from a curve, for latter calculations, either to derive a period rate or perform
        fixings table analysis.

        """
        rates = Series(NA, index=obs_dates[:-1])
        if not isinstance(self.fixings, NoInput):
            # then fixings will be a list or Series, scalars are already processed.
            assert not isinstance(self.fixings, float | Dual | Dual2 | Variable)  # noqa: S101

            if isinstance(self.fixings, list):
                rates.iloc[: len(self.fixings)] = self.fixings
            elif isinstance(self.fixings, Series):
                if not self.fixings.index.is_monotonic_increasing:
                    raise ValueError(
                        "`fixings` as a Series must have a monotonically increasing "
                        "datetimeindex.",
                    )
                # [-2] is used because the last rfr fixing is 1 day before the end
                fixing_rates = self.fixings.loc[obs_dates.iloc[0] : obs_dates.iloc[-2]]  # type: ignore[misc]

                try:
                    rates.loc[fixing_rates.index] = fixing_rates
                except KeyError as e:
                    raise ValueError(
                        "The supplied `fixings` contain more fixings than were "
                        "expected by the holiday calendar of the `curve`.\nThe "
                        "additional fixing dates are shown in the underlying "
                        "KeyError message below.\nIf the Series you are providing "
                        "contains valid fixings the fault probably lies with "
                        "Rateslib calendar definitions and should be reported.\n"
                        "This error can avoided by excluding these fixings using "
                        f"Series.pop().\n{e}",
                    )

                # basic error checking for missing fixings and provide warning.
                try:
                    first_forecast_date = rates[isna(rates)].index[0]
                    if rates[~isna(rates)].index[-1] > first_forecast_date:
                        # then a missing fixing exists
                        warnings.warn(
                            "`fixings` has missed a calendar value "
                            f"({first_forecast_date}) which "
                            "may be set to zero on a LineCurve or error on a Curve. "
                            "Subsequent fixings have been detected",
                            UserWarning,
                        )
                except (KeyError, IndexError):
                    pass
            else:
                raise TypeError(
                    "`fixings` should be of type scalar, None, list or Series.",
                )  # pragma: no cover
        return rates

    def _rfr_get_individual_fixings_data(
        self, calendar: CalTypes, convention: str, curve: Curve | NoInput, allow_na: bool = False
    ) -> dict[str, Any]:
        """
        Gets relevant DCF values and populates all the individual RFR fixings either known or
        from a curve, for latter calculations, either to derive a period rate or perform
        fixings table analysis.

        `allow_na` controls error handling. By default if any value is missing this will raise.
        """
        obs_dates, dcf_dates, dcf_vals, obs_vals = self._get_method_dcf_markers(
            calendar, convention, True
        )
        rates = self._rfr_get_series_with_populated_fixings(obs_dates)
        # TODO (low) the next few lines could probably be vectorised and made more efficient.
        fixed = (~isna(rates)).to_numpy()
        if not np.all(fixed):
            # then some fixings are missing... try from curve
            if isinstance(curve, NoInput):
                raise ValueError(
                    "Must supply a `curve` to FloatPeriod.rate() to forecast missing fixings.\n"
                    "Missing data is shown below for this period:\n"
                    f"{rates}"
                )
            rates = Series(  # type: ignore[assignment]
                {k: v if notna(v) else curve.rate(k, "1b", "F") for k, v in rates.items()}  # type: ignore
            )
            # Alternative solution to PR 172.
            # rates = Series({
            #     k: v
            #     if notna(v)
            #     else (curve.rate(k, "1b", "F") if k >= curve.node_dates[0] else None)
            #     for k, v in rates.items()
            # })

        if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
            # adjust the final rates values of the lockout arrays according to param
            try:
                rates.iloc[-self.method_param :] = rates.iloc[-self.method_param - 1]
            except IndexError:
                raise ValueError("period has too few dates for `rfr_lockout` param to function.")

        if rates.isna().any() and not allow_na:
            raise ValueError(
                "RFRs could not be calculated, have you missed providing `fixings` or "
                "does the `Curve` begin after the start of a `FloatPeriod` including "
                "the `method_param` adjustment?\n"
                "For further info see: Documentation > Cookbook > Working with fixings.",
            )

        return {
            "rates": rates,
            "fixed": fixed,
            "obs_dates": obs_dates,
            "dcf_dates": dcf_dates,
            "dcf_vals": dcf_vals,
            "obs_vals": obs_vals,
        }

    def fixings_table(
        self,
        curve: Curve | dict[str, Curve],
        approximate: bool = False,
        disc_curve: Curve | NoInput = NoInput(0),
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures.

        Parameters
        ----------
        curve : Curve, LineCurve, dict of such
            The forecast needed to calculate rates which affect compounding and
            dependent notional exposure.
        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        disc_curve : Curve
            A curve to make appropriate DF scalings. If *None* and ``curve`` contains
            DFs that will be used instead, otherwise errors are raised.
        right : datetime, optional
            Ignore fixing risks beyond this date.

        Returns
        -------
        DataFrame

        Notes
        -----
        **IBOR** and **RFR** ``fixing_method`` have different representations under
        this method.

        For *"ibor"* based floating rates the fixing exposures are indexed by
        **publication date** and not by reference value date. IBOR fixings tend to
        occur either in advance, or the same day.

        For *"rfr"* based floating rates the fixing exposures are indexed by the
        **reference value date** and not by publication date. RFR fixings tend to
        publish in arrears, usually at 9am the following business day. Central banks
        tend to publish data aligning the fixing rate with the reference value date
        and not by the publication date which is why this format is chosen. It also
        has practical application when constructing curves.

        Examples
        --------
        .. ipython:: python

           rfr_curve = Curve(
               nodes={dt(2022, 1, 1): 1.00, dt(2022, 1, 13): 0.9995},
               calendar="bus",
               id="rfr"
           )

        A regular `rfr_payment_delay` period.

        .. ipython:: python

           constants = {
               "start": dt(2022, 1, 5),
               "end": dt(2022, 1, 11),
               "payment": dt(2022, 1, 11),
               "frequency": "Q",
               "notional": -1000000,
               "currency": "gbp",
           }
           period = FloatPeriod(**{
               **constants,
               "fixing_method": "rfr_payment_delay"
           })
           period.fixings_table(rfr_curve)

        A 2 business day `rfr_observation_shift` period. Notice how the above had
        4 fixings spanning 6 calendar days, but the observation shift here attributes
        4 fixings spanning 4 calendar days so the notional exposure to those dates
        is increased by effectively 6/4.

        .. ipython:: python

           period = FloatPeriod(**{
               **constants,
               "fixing_method": "rfr_observation_shift",
               "method_param": 2,
            })
           period.fixings_table(rfr_curve)

        A 2 business day `rfr_lookback` period. Notice how the lookback period adjusts
        the weightings on the 6th January fixing by 3, and thus increases the notional
        exposure.

        .. ipython:: python

           period = FloatPeriod(**{
               **constants,
               "fixing_method": "rfr_lookback",
               "method_param": 2,
            })
           period.fixings_table(rfr_curve)

        A 2 business day `rfr_lockout` period. Notice how the exposure to the final
        fixing which then spans multiple days is increased.

        .. ipython:: python

           period = FloatPeriod(**{
               **constants,
               "fixing_method": "rfr_lockout",
               "method_param": 2,
           })
           period.fixings_table(rfr_curve)

        An IBOR fixing table

        .. ipython:: python

            ibor_3m = Curve(
               nodes={dt(2022, 1, 1): 1.00, dt(2023, 1, 1): 0.99},
               calendar="bus",
               id="ibor3m",
           )
           ibor_1m = Curve(
               nodes={dt(2022, 1, 1): 1.00, dt(2023, 1, 1): 0.995},
               calendar="bus",
               id="ibor1m",
           )
           period = FloatPeriod(
               start=dt(2022, 1, 5),
               end=dt(2022, 3, 7),
               payment=dt(2022, 3, 7),
               frequency="Q",
               notional=-1000000,
               currency="gbp",
               stub=True,
               fixing_method="ibor",
               method_param=2
           )
           period.fixings_table({"1m": ibor_1m, "3m": ibor_3m}, disc_curve=ibor_1m)
        """
        if isinstance(disc_curve, NoInput):
            if isinstance(curve, dict):
                raise ValueError("Cannot infer `disc_curve` from a dict of curves.")
            else:  # not isinstance(curve, dict):
                if curve._base_type == "dfs":
                    disc_curve_: Curve = curve
                else:
                    raise ValueError("Must supply a discount factor based `disc_curve`.")
        else:
            disc_curve_ = disc_curve

        if approximate:
            if not isinstance(self.fixings, NoInput):
                warnings.warn(
                    "Cannot approximate a fixings table when some published fixings "
                    f"are given within the period {self.start.strftime('%d-%b-%Y')}->"
                    f"{self.end.strftime('%d-%b-%Y')}. Switching to exact mode for this "
                    f"period.",
                    UserWarning,
                )
            else:
                try:
                    return self._fixings_table_fast(curve, disc_curve_, right=right)
                except ValueError:
                    # then probably a math domain error related to dates before the curve start
                    warnings.warn(
                        "Errored approximating a fixings table.\n Possibly this is due "
                        f"to period dates: {self.start.strftime('%d-%b-%Y')}->"
                        f"{self.end.strftime('%d-%b-%Y')},\n and the curve initial date."
                        f"Switching to exact mode for this period.",
                        UserWarning,
                    )
                    return self.fixings_table(
                        curve, approximate=False, disc_curve=disc_curve_, right=right
                    )

        if "rfr" in self.fixing_method:
            curve_: Curve | NoInput = _maybe_get_rfr_curve_from_dict(curve)
            cal_, conv_ = self._maybe_get_cal_and_conv_from_curve(curve_)
            _d = self._rfr_get_individual_fixings_data(cal_, conv_, curve_, allow_na=True)

            if not isinstance(right, NoInput) and _d["obs_dates"][0] > right:
                # then all fixings are out of scope, so perform no calculations
                df = DataFrame(
                    {
                        "obs_dates": [],
                        "notional": [],
                        "risk": [],
                        "dcf": [],
                        "rates": [],
                    },
                ).set_index("obs_dates")
            elif _d["rates"].isna().any():
                if (
                    isinstance(curve_, NoInput)
                    or not _d["obs_dates"].iloc[-1] <= curve_.node_dates[0]
                ):
                    raise ValueError(
                        "RFRs could not be calculated, have you missed providing `fixings` or "
                        "`curve`, or does the `curve` begin after the start of a `FloatPeriod` "
                        "including the `method_param` adjustment?\n"
                        "For further info see: Documentation > Cookbook > Working with fixings.",
                    )
                else:
                    # period exists before the curve and fixings are not supplied.
                    # Exposures are overwritten to zero.
                    df = DataFrame(
                        {
                            "obs_dates": _d["obs_dates"].iloc[:-1],
                            "notional": [0.0] * len(_d["rates"]),
                            "risk": [0.0] * len(_d["rates"]),
                            "dcf": _d["dcf_vals"],
                            "rates": _d["rates"].astype(float).reset_index(drop=True),
                        },
                    ).set_index("obs_dates")
            else:
                rate, table = self._rfr_fixings_array(_d, disc_curve_)
                table = table.iloc[:-1]
                df = table[["obs_dates", "notional", "risk", "dcf", "rates"]].set_index("obs_dates")

            if not isinstance(curve_, NoInput):
                id_: str = curve_.id
            else:
                id_ = "fixed"
            df.columns = MultiIndex.from_tuples(
                [(id_, "notional"), (id_, "risk"), (id_, "dcf"), (id_, "rates")]
            )
            return _trim_df_by_index(df, NoInput(0), right)
        else:  # "ibor" in self.fixing_method:
            return self._ibor_fixings_table(curve, disc_curve_, right)

    def _fixings_table_fast(
        self, curve: Curve | dict[str, Curve], disc_curve: Curve, right: NoInput | datetime
    ) -> DataFrame:
        """
        Return a DataFrame of **approximate** fixing exposures.

        For arguments see :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
        """
        if "rfr" in self.fixing_method:
            curve_: Curve = _maybe_get_rfr_curve_from_dict(curve)  # type: ignore[assignment]
            # Depending upon method get the observation dates and dcf dates
            obs_dates, dcf_dates, dcf_vals, obs_vals = self._get_method_dcf_markers(
                curve_.calendar, curve_.convention, True
            )

            if not isinstance(right, NoInput) and obs_dates[0] > right:
                # then all fixings are out of scope, so perform no calculations
                df = DataFrame(
                    [],
                    columns=MultiIndex.from_tuples(
                        [
                            (curve_.id, "notional"),
                            (curve_.id, "risk"),
                            (curve_.id, "dcf"),
                            (curve_.id, "rates"),
                        ]
                    ),
                    index=Index([], name="obs_dates", dtype=float),
                )
                return df
            # approximate DFs
            v_vals = Series(np.nan, index=obs_dates.iloc[1:])
            v_vals.iloc[0] = log(_dual_float(disc_curve[obs_dates.iloc[1]]))
            v_vals.iloc[-1] = log(_dual_float(disc_curve[obs_dates.iloc[-1]]))
            v_vals = v_vals.interpolate(method="time")
            v_vals = Series(np.exp(v_vals.to_numpy()), index=obs_vals.index)

            scalar = dcf_vals.to_numpy() / obs_vals.to_numpy()
            if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
                scalar[-self.method_param :] = 0.0
                scalar[-(self.method_param + 1)] = (
                    obs_vals.iloc[-(self.method_param + 1) :].sum()
                    / obs_vals.iloc[-(self.method_param + 1)]
                )
            # perform an efficient rate approximation
            rate = curve_._rate_with_raise(
                effective=obs_dates.iloc[0],
                termination=obs_dates.iloc[-1],
            )
            r_bar, d, n = average_rate(
                obs_dates.iloc[0],
                obs_dates.iloc[-1],
                curve_.convention,
                rate,
            )
            # approximate sensitivity to each fixing
            z = self.float_spread / 10000
            if "avg" in self.fixing_method:
                drdri: DualTypes = 1 / n
            elif self.spread_compound_method == "none_simple":
                drdri = (1 / n) * (1 + (r_bar / 100) * d) ** (n - 1)
            elif self.spread_compound_method == "isda_compounding":
                drdri = (1 / n) * (1 + (r_bar / 100 + z) * d) ** (n - 1)
            elif self.spread_compound_method == "isda_flat_compounding":
                dr = d * r_bar / 100
                drdri = (1 / n) * (
                    ((1 / n) * (comb(n, 1) + comb(n, 2) * dr + comb(n, 3) * dr**2))
                    + ((r_bar / 100 + z) / n) * (comb(n, 2) * d + 2 * comb(n, 3) * dr * d)
                )

            v = _dual_float(disc_curve[self.payment])
            notional_exposure = Series(
                (-self.notional * self.dcf * _dual_float(drdri) * v / d * scalar) / v_vals,
                index=obs_vals.index,
            )

            table = DataFrame(
                {
                    "obs_dates": obs_dates,
                    "obs_dcf": obs_vals,
                    "dcf_dates": dcf_dates,
                    "dcf": dcf_vals,
                    "notional": notional_exposure,
                    "risk": notional_exposure * v_vals * obs_vals * 0.0001,
                    "rates": Series(rate, index=obs_dates.index).astype(  # type: ignore[arg-type]
                        float,
                    ),  # .apply(float, convert_dtype=float),
                },
            )

            table = table.iloc[:-1]
            df = table[["obs_dates", "notional", "risk", "dcf", "rates"]].set_index("obs_dates")
            df.columns = MultiIndex.from_tuples(
                [
                    (curve_.id, "notional"),
                    (curve_.id, "risk"),
                    (curve_.id, "dcf"),
                    (curve_.id, "rates"),
                ]
            )
            return _trim_df_by_index(df, NoInput(0), right)
        else:  # "ibor" in self.fixing_method:
            return self._ibor_fixings_table(curve, disc_curve, right=right)

    def _ibor_fixings_table(
        self,
        curve: Curve | dict[str, Curve],
        disc_curve: Curve,
        right: datetime | NoInput,
        risk: DualTypes | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Calculate a fixings_table under an IBOR based methodology.

        Parameters
        ----------
        curve: Curve or Dict
            Dict may be relevant if the period is a stub.
        risk: float, optional
            This is the known financial exposure to the movement of the period IBOR fixing.
            Expressed per 1 in percentage rate, i.e. risk per bp * 10000

        Returns
        -------
        DataFrame
        """
        if isinstance(curve, dict):
            if self.stub:
                # then must perform an interpolated calculation
                return self._ibor_stub_fixings_table(curve, disc_curve, right, risk)
            else:  # not self.stub:
                # then extract the one relevant curve from dict
                curve_: Curve = _get_ibor_curve_from_dict(self.freq_months, curve)
        else:
            curve_ = curve

        return self._ibor_single_tenor_fixings_table(
            curve_, disc_curve, f"{self.freq_months}m", right, risk
        )

    def _ibor_single_tenor_fixings_table(
        self,
        curve: Curve,
        disc_curve: Curve,
        tenor: str,
        right: datetime | NoInput,
        risk: DualTypes | NoInput = NoInput(0),
    ) -> DataFrame:
        calendar = curve.calendar
        fixing_dt = calendar.lag(self.start, -self.method_param, False)
        if not isinstance(right, NoInput) and fixing_dt > right:
            # fixing not in scope, perform no calculations
            df = DataFrame(
                {
                    "obs_dates": [],
                    "notional": [],
                    "risk": [],
                    "dcf": [],
                    "rates": [],
                },
            ).set_index("obs_dates")
        else:
            reg_end_dt = add_tenor(self.start, tenor, curve.modifier, calendar)
            reg_dcf = dcf(self.start, reg_end_dt, curve.convention, reg_end_dt)

            if not isinstance(self.fixings, NoInput) or fixing_dt < curve.node_dates[0]:
                # then fixing is set so return zero exposure.
                _rate = NA if isinstance(self.fixings, NoInput) else _dual_float(self.rate(curve))
                df = DataFrame(
                    {
                        "obs_dates": [fixing_dt],
                        "notional": 0.0,
                        "risk": 0.0,
                        "dcf": [reg_dcf],
                        "rates": [_rate],
                    },
                ).set_index("obs_dates")
            else:
                risk = (
                    -self.notional * self.dcf * disc_curve[self.payment]
                    if isinstance(risk, NoInput)
                    else risk
                )
                df = DataFrame(
                    {
                        "obs_dates": [fixing_dt],
                        "notional": _dual_float(risk / (reg_dcf * disc_curve[reg_end_dt])),
                        "risk": _dual_float(risk) * 0.0001,  # scale to bp
                        "dcf": [reg_dcf],
                        "rates": [self.rate(curve)],
                    },
                ).set_index("obs_dates")

        df.columns = MultiIndex.from_tuples(
            [(curve.id, "notional"), (curve.id, "risk"), (curve.id, "dcf"), (curve.id, "rates")]
        )
        return df

    def _ibor_stub_fixings_table(
        self,
        curve: dict[str, Curve],
        disc_curve: Curve,
        right: datetime | NoInput,
        risk: DualTypes | NoInput = NoInput(0),
    ) -> DataFrame:
        calendar = next(iter(curve.values())).calendar  # note: ASSUMES all curve calendars are same
        values = {add_tenor(self.start, k, "MF", calendar): k for k, v in curve.items()}
        values = dict(sorted(values.items()))
        reg_end_dts = list(values.keys())

        if self.end > reg_end_dts[-1]:
            warnings.warn(
                "Interpolated stub period has a length longer than the provided "
                "IBOR curve tenors: using the longest IBOR value.",
                UserWarning,
            )
            a1, a2 = 0.0, 1.0
            i = len(reg_end_dts) - 2
        elif self.end < reg_end_dts[0]:
            warnings.warn(
                "Interpolated stub period has a length shorter than the provided "
                "IBOR curve tenors: using the shortest IBOR value.",
                UserWarning,
            )
            a1, a2 = 1.0, 0.0
            i = 0
        else:
            i = index_left(reg_end_dts, len(reg_end_dts), self.end)
            a2 = (self.end - reg_end_dts[i]) / (reg_end_dts[i + 1] - reg_end_dts[i])
            a1 = 1.0 - a2

        risk = (
            -self.notional * self.dcf * disc_curve[self.payment]
            if isinstance(risk, NoInput)
            else risk
        )
        tenor1, tenor2 = list(values.values())[i], list(values.values())[i + 1]
        df1 = self._ibor_single_tenor_fixings_table(
            curve[tenor1], disc_curve, tenor1, right, risk * a1
        )
        # df1[(curve[tenor1].id, "notional")] = df1[(curve[tenor1].id, "notional")] * a1
        # df1[(curve[tenor1].id, "risk")] = df1[(curve[tenor1].id, "risk")] * a1
        df2 = self._ibor_single_tenor_fixings_table(
            curve[tenor2], disc_curve, tenor2, right, risk * a2
        )
        # df2[(curve[tenor2].id, "notional")] = df2[(curve[tenor2].id, "notional")] * a2
        # df2[(curve[tenor2].id, "risk")] = df2[(curve[tenor2].id, "risk")] * a2
        df = concat([df1, df2], axis=1)
        return df

    def _rfr_fixings_array(
        self,
        d: dict[str, Any],
        disc_curve: Curve,
    ) -> tuple[DualTypes, DataFrame]:
        """
        Calculate the rate of a period via extraction and combination of every fixing.

        This method of calculation is inefficient and used when either:

        - known fixings needs to be combined with unknown fixings,
        - the fixing_method is of a type that needs individual fixing data,
        - the spread compound method is of a type that needs individual fixing data.

        Parameters
        ----------
        d: dict
            Data passed from function `get_rfr_individual_fixing_data`.
        disc_curve : Curve
            The discount curve used in scaling factors and calculations.

        Returns
        -------
        tuple
            The compounded rate, DataFrame of the calculation data.

        Notes
        -----
        ``start_obs`` and ``end_obs`` define the observation period for fixing rates.
        ``start_dcf`` and ``end_dcf`` define the period for day count fractions.
        Unless *"lookback"* is used which mis-aligns the obs and dcf periods these
        will be aligned.

        The ``fixing_exposure_approx`` is available only for ``spread_compound_method``
        that is either *"none_simple"* or *"isda_compounding"*.
        """
        # then perform additional calculations to return fixings table
        dcf_of_r = d["obs_vals"]  # these are the 1-d DCFs associated with each published fixing
        v_with_r = Series(
            [disc_curve[d["obs_dates"][i]] for i in range(1, len(d["dcf_dates"].index))]
        )
        # these are zero-lag discount factors associated with each published fixing
        rates_dual = Series(
            [
                Dual(_dual_float(r), [f"fixing_{i}"], [])
                for i, (k, r) in enumerate(d["rates"].items())
            ],
            index=d["rates"].index,
        )
        if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
            rates_dual.iloc[-self.method_param :] = rates_dual.iloc[-self.method_param - 1]
        if "avg" in self.fixing_method:
            rate: Dual = self._rate_rfr_avg_with_spread(
                rates_dual.to_numpy(), d["dcf_vals"].to_numpy()
            )  # type: ignore[assignment]
        else:
            rate = self._rate_rfr_isda_compounded_with_spread(  # type: ignore[assignment]
                rates_dual.to_numpy(), d["dcf_vals"].to_numpy()
            )

        dr_drj = Series(
            [gradient(rate, [f"fixing_{i}"])[0] for i in range(len(d["dcf_dates"].index) - 1)],
        ).astype(float)
        v = disc_curve[self.payment]

        risk = -_dual_float(self.notional) * self.dcf * _dual_float(v) * dr_drj
        risk[d["fixed"]] = 0.0

        notional_exposure = Series(0.0, index=range(len(dr_drj.index)))
        notional_exposure[~d["fixed"]] = risk[~d["fixed"]] / (
            dcf_of_r[~d["fixed"]] * v_with_r[~d["fixed"]].astype(float)
        )
        extra_cols = {
            "obs_dcf": dcf_of_r,
            "notional": notional_exposure.astype(float),  # apply(float, convert_dtype=float),
            "dr_drj": dr_drj,
            "risk": risk * 0.0001,
        }

        return rate, DataFrame(
            {
                "obs_dates": d["obs_dates"],
                "dcf_dates": d["dcf_dates"],
                "dcf": d["dcf_vals"],
                "rates": d["rates"].astype(float).reset_index(drop=True),
                **extra_cols,
            },
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @property
    def _is_inefficient(self) -> bool:
        """
        An inefficient float period is one which is RFR based and for which each individual
        RFR fixing is required is order to calculate correctly. This occurs in the
        following cases:

        1) The ``fixing_method`` is lookback - since fixing have arbitrary weightings
           misaligned with their standard weightings due to arbitrary shifts.
        2) The ``spread_compound_method`` is not *"none_simple"* - this is because the
           float spread is compounded alongside the rates so there is a non-linear
           relationship. Note if spread is zero this is negated and can be ignored.
        3) The ``fixing_method`` is lockout - technically this could be made semi
           efficient by splitting calculations into two parts. As of now it
           remains within the inefficient section.
        4) ``fixings`` are given which need to be incorporated into the calculation
        """
        if self.fixing_method in ["rfr_payment_delay", "rfr_observation_shift"]:
            if not isinstance(self.fixings, NoInput):
                return True
            else:
                return not (self.float_spread == 0 or self.spread_compound_method == "none_simple")
        elif self.fixing_method == "ibor":
            return False
        # else fixing method in ["rfr_lookback", "rfr_lockout"]
        return True

    def _get_method_dcf_endpoints(
        self, calendar: CalTypes
    ) -> tuple[datetime, datetime, datetime, datetime]:
        """
        For RFR periods return the relevant DCF markers for different aspects of calculation.

        `start_obs` and `end_obs` are the dates between which RFR fixings are observed.

        `start_dcf` and `end_dcf` are the dates between which the DCFs for each observed fixing
        are compounded or averaged with to determine the ultimate rate.

        For all methods except 'lookback', these dates will align with each other.
        For 'lookback' the observed RFRs are applied over different DCFs that do not naturally
        align.
        """
        # Depending upon method get the observation dates and dcf dates
        if self.fixing_method in [
            "rfr_payment_delay",
            "rfr_lockout",
            "rfr_payment_delay_avg",
            "rfr_lockout_avg",
        ]:
            start_obs, end_obs = self.start, self.end
            start_dcf, end_dcf = self.start, self.end
        elif self.fixing_method in [
            "rfr_observation_shift",
            "rfr_observation_shift_avg",
        ]:
            start_obs = calendar.lag(self.start, -self.method_param, settlement=False)
            end_obs = calendar.lag(self.end, -self.method_param, settlement=False)
            start_dcf, end_dcf = start_obs, end_obs
        elif self.fixing_method in ["rfr_lookback", "rfr_lookback_avg"]:
            start_obs = calendar.lag(self.start, -self.method_param, settlement=False)
            end_obs = calendar.lag(self.end, -self.method_param, settlement=False)
            start_dcf, end_dcf = self.start, self.end
        else:
            raise NotImplementedError(
                "`fixing_method` should be in {'rfr_payment_delay', 'rfr_lockout', "
                "'rfr_lookback', 'rfr_observation_shift'} or the same with '_avg' as "
                "a suffix for averaging methods.",
            )

        return start_obs, end_obs, start_dcf, end_dcf

    def _get_method_dcf_markers(
        self, calendar: CalTypes, convention: str, exposure: bool = False
    ) -> tuple[
        Series[datetime],
        Series[datetime],
        Series[float],
        Series[float],
    ]:
        """
        Use conventions from the given `curve` and the data attached to self to derive
        relevant DCF calculations for the Period.

        calendar: calendar derived from the rate index Curve.
        convention: the day count convention derived from the rate index Curve.
        exposure: bool / If needed for exposure generation additional values returned.
        """
        start_obs, end_obs, start_dcf, end_dcf = self._get_method_dcf_endpoints(calendar)
        # dates of the fixing observation period
        obs_dates = Series(calendar.bus_date_range(start=start_obs, end=end_obs))
        # TODO (low) if start_obs and end_obs are not business days this may raise. But cases may
        # arise if using unadjusted schedules. Then an improvement to use a `lag` adjustment
        # may be needed but this also needs careful thought for consequences.

        # dates for the dcf weight for each observation towards the calculation
        msg = (
            "RFR Observation and Accrual DCF dates do not align.\n"
            "This is usually the result of a 'rfr_lookback' Period which does "
            "not adhere to the holiday calendar of the `curve`.\n"
            f"start date: {self.start.strftime('%d-%m-%Y')} is curve holiday? "
            f"{calendar.is_non_bus_day(self.start)}\n"
            f"end date: {self.end.strftime('%d-%m-%Y')} is curve holiday? "
            f"{calendar.is_non_bus_day(self.end)}\n"
        )
        try:
            dcf_dates = Series(calendar.bus_date_range(start=start_dcf, end=end_dcf))
        except ValueError:
            raise ValueError(msg)
        else:
            if len(dcf_dates) != len(obs_dates):
                # this might only be true with lookback when obs dates are adjusted
                # but DCF dates are not, and if starting on holiday causes problems.
                raise ValueError(msg)

        # TODO (low) this calculation could be vectorised by a 360 or 365 multiplier
        dcf_vals = Series(
            [  # calculate the dcf values from the dcf dates
                dcf(dcf_dates[i], dcf_dates[i + 1], convention)
                for i in range(len(dcf_dates.index) - 1)
            ],
        )

        obs_vals = None
        if exposure:
            # for calculating fixing notional exposure the DCF of the actual fixings is
            # required for comparison
            # need to calculate the dcfs associated with the rates (unshifted)
            if self.fixing_method in [
                "rfr_payment_delay",
                "rfr_observation_shift",
                "rfr_lockout",
                "rfr_payment_delay_avg",
                "rfr_observation_shift_avg",
                "rfr_lockout_avg",
            ]:  # for all these methods there is no shift
                obs_vals = dcf_vals.copy()
            elif self.fixing_method in ["rfr_lookback", "rfr_lookback_avg"]:
                obs_vals = Series(
                    [
                        dcf(obs_dates[i], obs_dates[i + 1], convention)
                        for i in range(len(dcf_dates.index) - 1)
                    ],
                )

        return obs_dates, dcf_dates, dcf_vals, obs_vals  # type: ignore[return-value]

    def _get_analytic_delta_quadratic_coeffs(
        self, fore_curve: Curve, disc_curve: Curve
    ) -> tuple[DualTypes, DualTypes]:
        """
        For use in the Leg._spread calculation get the 'a' and 'b' coefficients
        """
        os, oe, _, _ = self._get_method_dcf_endpoints(fore_curve.calendar)
        rate = fore_curve._rate_with_raise(
            effective=os,
            termination=oe,
            float_spread=0.0,
            spread_compound_method=self.spread_compound_method,
        )
        r, d, n = average_rate(os, oe, fore_curve.convention, rate)
        # approximate sensitivity to each fixing
        z = 0.0 if self.float_spread is None else self.float_spread
        if self.spread_compound_method == "isda_compounding":
            d2rdz2: Number = d * (n - 1) * (1 + (r / 100 + z / 10000) * d) ** (n - 2) / 1e8
            drdz: Number = (1 + (r / 100 + z / 10000) * d) ** (n - 1) / 1e4
            Nvd = -self.notional * disc_curve[self.payment] * self.dcf
            a: DualTypes = 0.5 * Nvd * d2rdz2
            b: DualTypes = Nvd * drdz
        elif self.spread_compound_method == "isda_flat_compounding":
            # d2rdz2 = 0.0
            drdz = (1 + comb(n, 2) / n * r / 100 * d + comb(n, 3) / n * (r / 100 * d) ** 2) / 1e4
            Nvd = -self.notional * disc_curve[self.payment] * self.dcf
            a, b = 0.0, Nvd * drdz

        return a, b


class CreditPremiumPeriod(BasePeriod):
    """
    Create a credit premium period defined by a credit spread.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    fixed_rate : float or None, optional
        The rate applied to determine the cashflow. If `None`, can be set later,
        typically after a mid-market rate for all periods has been calculated.
        Entered in percentage points, e.g 50bps is 0.50.
    premium_accrued : bool, optional
        Whether the premium is accrued within the period to default.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = -NdS

    The NPV of the full cashflow is defined as;

    .. math::

       P_c = Cv(m_{payment})Q(m_{end})

    If ``premium_accrued`` is permitted then an additional component equivalent to the following
    is calculated using an approximation of the inter-period default rate,

    .. math::

       P_a = Cv(m_{payment}) \\left ( Q(m_{start}) - Q(m_{end}) \\right ) \\frac{(n+r)}{2n}

    where *r* is the number of days after the *start* that *today* is for an on-going period, zero otherwise, and
    :math:`Q(m_{start})` is equal to one for an on-going period.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = P_c + I_{pa} P_a

    where :math:`I_{pa}` is an indicator function if the *Period* allows ``premium_accrued`` or not.

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial S} = Ndv(m) \\left ( Q(m_{end}) + I_{pa} (Q(m_{start}) - Q(m_{end}) \\frac{(n+r)}{2n}  \\right )
    """  # noqa: E501

    def __init__(
        self,
        *args: Any,
        fixed_rate: float | NoInput = NoInput(0),
        premium_accrued: bool | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.premium_accrued = _drb(defaults.cds_premium_accrued, premium_accrued)
        self.fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)

    @property
    def cashflow(self) -> float | None:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional.
        """
        if isinstance(self.fixed_rate, NoInput):
            return None
        else:
            _: float = -self.notional * self.dcf * self.fixed_rate * 0.01
            return _

    def accrued(self, settlement: datetime) -> float | None:
        """
        Calculate the amount of premium accrued until a specific date within the *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float
        """
        if self.cashflow is None:  # self.fixed_rate is NoInput
            return None
        else:
            if settlement <= self.start or settlement >= self.end:
                return 0.0
            return self.cashflow * (settlement - self.start).days / (self.end - self.start).days

    def npv(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditPremiumPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        if not isinstance(disc_curve, Curve) and isinstance(disc_curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")
        if not isinstance(curve, Curve) and isinstance(curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")
        if isinstance(self.fixed_rate, NoInput):
            raise ValueError("`fixed_rate` must be set as a value to return a valid NPV.")
        v_payment = disc_curve[self.payment]
        q_end = curve[self.end]
        _ = 0.0
        if self.premium_accrued:
            v_end = disc_curve[self.end]
            n = _dual_float((self.end - self.start).days)

            if self.start < curve.node_dates[0]:
                # then mid-period valuation
                r: float = _dual_float((curve.node_dates[0] - self.start).days)
                q_start: DualTypes = 1.0
                _v_start: DualTypes = 1.0
            else:
                r, q_start, _v_start = 0.0, curve[self.start], disc_curve[self.start]

            # method 1:
            _ = 0.5 * (1 + r / n)
            _ *= q_start - q_end
            _ *= v_end

            # # method 4 EXACT
            # _ = 0.0
            # for i in range(1, int(s)):
            #     m_i, m_i2 = m_today + timedelta(days=i-1), m_today + timedelta(days=i)
            #     _ += (
            #     (i + r) / n * disc_curve[m_today + timedelta(days=i)] * (curve[m_i] - curve[m_i2])
            #     )

        return _maybe_local(self.cashflow * (q_end * v_payment + _), local, self.currency, fx, base)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *CreditPremiumPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        if not isinstance(disc_curve, Curve) and isinstance(disc_curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")
        if not isinstance(curve, Curve) and isinstance(curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")

        v_payment = disc_curve[self.payment]
        q_end = curve[self.end]
        _ = 0.0
        if self.premium_accrued:
            v_end = disc_curve[self.end]
            n = _dual_float((self.end - self.start).days)

            if self.start < curve.node_dates[0]:
                # then mid-period valuation
                r: float = _dual_float((curve.node_dates[0] - self.start).days)
                q_start: DualTypes = 1.0
                _v_start: DualTypes = 1.0
            else:
                r = 0.0
                q_start = curve[self.start]
                _v_start = disc_curve[self.start]

            # method 1:
            _ = 0.5 * (1 + r / n)
            _ *= q_start - q_end
            _ *= v_end

        return _maybe_fx_converted(
            0.0001 * self.notional * self.dcf * (q_end * v_payment + _),
            self.currency,
            fx,
            base,
        )

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),  # type: ignore[override]
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *CreditPremiumPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if not isinstance(curve, NoInput) and not isinstance(disc_curve, NoInput):
            npv_: DualTypes = self.npv(curve, disc_curve)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx)
            survival = _dual_float(curve[self.end])
        else:
            npv, npv_fx, survival = None, None, None

        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["rate"]: None
            if isinstance(self.fixed_rate, NoInput)
            else _dual_float(self.fixed_rate),
            defaults.headers["survival"]: survival,
            defaults.headers["cashflow"]: None
            if self.cashflow is None
            else _dual_float(self.cashflow),
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }


class CreditProtectionPeriod(BasePeriod):
    """
    Create a credit protection period defined by a recovery rate.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    recovery_rate : float, Dual, Dual2, optional
        The assumed recovery rate that defines payment on credit default. Set by ``defaults``.
    discretization : int, optional
        The number of days to discretize the numerical integration over possible credit defaults.
        Set by ``defaults``.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow``, paid on a credit event, is defined as follows;

    .. math::

       C = -N(1-R)

    where *R* is the recovery rate.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as a discretized sum of inter-period blocks whose
    probability of default and protection payment sum to give an expected payment;

    .. math::

       j &= [n/discretization] \\\\
       P &= C \\sum_{i=1}^{j} \\frac{1}{2} \\left ( v(m_{i-1}) + v_(m_{i}) \\right ) \\left ( Q(m_{i-1}) - Q(m_{i}) \\right ) \\\\

    The *start* and *end* of the period are restricted by the *Curve* if the *Period* is current (i.e. *today* is
    later than *start*)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0
    """  # noqa: E501

    def __init__(
        self,
        *args: Any,
        recovery_rate: DualTypes | NoInput = NoInput(0),
        discretization: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.recovery_rate: DualTypes = _drb(defaults.cds_recovery_rate, recovery_rate)
        if self.recovery_rate < 0.0 and self.recovery_rate > 1.0:
            raise ValueError("`recovery_rate` value must be in [0.0, 1.0]")
        self.discretization: int = _drb(defaults.cds_protection_discretization, discretization)
        super().__init__(*args, **kwargs)

    @property
    def cashflow(self) -> DualTypes:
        """
        float, Dual or Dual2 : The calculated protection amount determined from notional
        and recovery rate.
        """
        return -self.notional * (1 - self.recovery_rate)

    def npv(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditProtectionPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        if not isinstance(disc_curve, Curve) and isinstance(disc_curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")
        if not isinstance(curve, Curve) and isinstance(curve, NoInput):
            raise TypeError("`curves` have not been supplied correctly.")

        if self.start < curve.node_dates[0]:
            s2 = curve.node_dates[0]
        else:
            s2 = self.start

        value: DualTypes = 0.0
        q2: DualTypes = curve[s2]
        v2: DualTypes = disc_curve[s2]
        while s2 < self.end:
            q1, v1 = q2, v2
            s2 = s2 + timedelta(days=self.discretization)
            if s2 > self.end:
                s2 = self.end
            q2, v2 = curve[s2], disc_curve[s2]
            value += 0.5 * (v1 + v2) * (q1 - q2)
            # value += v2 * (q1 - q2)

        value *= self.cashflow
        return _maybe_local(value, local, self.currency, fx, base)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *CreditProtectionPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),  # type: ignore[override]
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *CreditProtectionPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if not isinstance(curve, NoInput) and not isinstance(disc_curve, NoInput):
            npv_: DualTypes = self.npv(curve, disc_curve)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx)
            survival = _dual_float(curve[self.end])
        else:
            npv, npv_fx, survival = None, None, None

        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["recovery"]: _dual_float(self.recovery_rate),
            defaults.headers["survival"]: survival,
            defaults.headers["cashflow"]: _dual_float(self.cashflow),
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def analytic_rec_risk(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> float:
        """
        Calculate the exposure of the NPV to a change in recovery rate.

        For parameters see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`

        Returns
        -------
        float
        """
        rr = self.recovery_rate
        if isinstance(rr, Dual | Dual2 | Variable):
            self.recovery_rate = Variable(rr.real, ["__recovery_rate__"])
        else:
            self.recovery_rate = Variable(_dual_float(rr), ["__recovery_rate__"])
        pv: Dual | Dual2 | Variable = self.npv(curve, disc_curve, fx, base, False)  # type: ignore[assignment]
        self.recovery_rate = rr
        _: float = _dual_float(gradient(pv, ["__recovery_rate__"], order=1)[0])
        return _ * 0.01


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
        notional: float,
        payment: datetime,
        currency: str | NoInput = NoInput(0),
        stub_type: str | NoInput = NoInput(0),
        rate: float | NoInput = NoInput(0),
    ):
        self.notional, self.payment = notional, payment
        self.currency = _drb(defaults.base_currency, currency).lower()
        self.stub_type = stub_type
        self._rate: float | NoInput = rate if isinstance(rate, NoInput) else _dual_float(rate)

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def rate(self) -> float | None:
        """
        Return the associated rate initialised with the *Cashflow*. Not used for calculations.
        """
        return None if isinstance(self._rate, NoInput) else self._rate

    def npv(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
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
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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
    def cashflow(self) -> float:
        return -self.notional

    def analytic_delta(
        self,
        curve: Curve | None = None,
        disc_curve: Curve | None = None,
        fx: float | FXRates | FXForwards | None = None,
        base: str | None = None,
    ) -> DualTypes:
        """
        Return the analytic delta of the *Cashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class IndexMixin(metaclass=ABCMeta):
    """
    Abstract base class to include methods and properties related to indexed *Periods*.
    """

    index_base: DualTypes | NoInput
    index_method: str
    index_fixings: DualTypes | Series[DualTypes] | NoInput  # type: ignore[type-var]
    index_lag: int
    index_only: bool = False
    payment: datetime
    end: datetime
    currency: str

    @property
    @abstractmethod
    def real_cashflow(self) -> DualTypes | None:
        pass  # pragma: no cover

    def cashflow(self, curve: Curve | NoInput = NoInput(0)) -> DualTypes | None:
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
        self, curve: Curve | NoInput = NoInput(0)
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
        denominator = self._index_value(
            i_fixings=self.index_base,
            i_date=i_date_base,
            i_curve=curve,
            i_lag=self.index_lag,
            i_method=self.index_method,
        )
        numerator = self._index_value(
            i_fixings=self.index_fixings,
            i_date=self.end,
            i_curve=curve,
            i_lag=self.index_lag,
            i_method=self.index_method,
        )
        if numerator is None or denominator is None:
            return None, numerator, denominator
        else:
            return numerator / denominator, numerator, denominator

    @staticmethod
    def _index_value_from_curve(
        i_date: datetime,
        i_curve: Curve | NoInput,
        i_lag: int,
        i_method: str,
    ) -> DualTypes | None:
        if isinstance(i_curve, NoInput):
            return None
        elif i_lag != i_curve.index_lag:
            warnings.warn(
                f"The `index_lag` of the Period ({i_lag}) does not match "
                f"the `index_lag` of the Curve ({i_curve.index_lag}): returning None."
            )
            return None  # TODO decide if RolledCurve to correct index lag be attempted
        else:
            return i_curve.index_value(i_date, i_method)

    @staticmethod
    def _index_value(
        i_fixings: DualTypes | Series[DualTypes] | NoInput,  # type: ignore[type-var]
        i_date: datetime | NoInput,
        i_curve: Curve | NoInput,
        i_lag: int,
        i_method: str,
    ) -> DualTypes | None:
        """
        Project an index rate, or lookup from provided fixings, for a given date.

        If ``index_fixings`` are set on the period this will be used instead of
        the ``curve``.

        Parameters
        ----------
        curve : Curve

        Returns
        -------
        float, Dual, Dual2, Variable or None
        """
        if isinstance(i_date, NoInput):
            if not isinstance(i_fixings, Series | NoInput):
                # i_fixings is a given value, probably aligned with an ``index_base``
                return i_fixings
            else:
                # internal method so this line should never be hit
                raise ValueError(
                    "Must supply an `i_date` from which to forecast."
                )  # pragma: no cover
        else:
            if isinstance(i_fixings, NoInput):
                return IndexMixin._index_value_from_curve(i_date, i_curve, i_lag, i_method)
            elif isinstance(i_fixings, Series):
                if i_method == "daily":
                    adj_date = i_date
                    unavailable_date: datetime = i_fixings.index[-1]  # type: ignore[attr-defined]
                else:
                    adj_date = datetime(i_date.year, i_date.month, 1)
                    _: datetime = i_fixings.index[-1]  # type: ignore[attr-defined]
                    unavailable_date = _get_eom(_.month, _.year)

                if i_date > unavailable_date:
                    if isinstance(i_curve, NoInput):
                        return None  # NoInput(0)
                    else:
                        return IndexMixin._index_value_from_curve(i_date, i_curve, i_lag, i_method)
                    # raise ValueError(
                    #     "`index_fixings` cannot forecast the index value. "
                    #     f"There are no fixings available after date: {unavailable_date}"
                    # )
                else:
                    try:
                        ret: DualTypes | None = i_fixings[adj_date]  # type: ignore[index]
                        return ret
                    except KeyError:
                        s = i_fixings.copy()  # type: ignore[attr-defined]
                        s.loc[adj_date] = np.nan
                        ret = s.sort_index().interpolate("time")[adj_date]
                        return ret
            else:
                # i_fixings is a given value
                return i_fixings

    def npv(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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


class IndexFixedPeriod(IndexMixin, FixedPeriod):
    """
    Create a period defined with a real rate adjusted by an index.

    When used with an inflation index this defines a real coupon period with a
    cashflow adjusted upwards by the inflation index.

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
        fixings that are available in that object, using linear interpolation if
        necessary.
    index_method : str, optional
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
        # if index_base is None:
        #     raise ValueError("`index_base` cannot be None.")
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = (
            defaults.index_method if isinstance(index_method, NoInput) else index_method.lower()
        )
        self.index_lag = defaults.index_lag if isinstance(index_lag, NoInput) else index_lag
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super(IndexMixin, self).__init__(*args, **kwargs)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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
        curve: Curve | NoInput = NoInput(0),  # type: ignore[override]
        disc_curve: Curve | NoInput = NoInput(0),
        fx: DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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

    Parameters
    ----------
    args : dict
        Required positional args to :class:`Cashflow`.
    index_base : float or None, optional
        The base index to determine the cashflow.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the whole
        period. If a datetime indexed ``Series`` will use the
        fixings that are available in that object, and derive the rest from the
        ``curve``.
    index_method : str
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
        index_base: float,
        index_fixings: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        index_only: bool = False,
        end: datetime | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = (
            defaults.index_method if isinstance(index_method, NoInput) else index_method.lower()
        )
        self.index_lag = defaults.index_lag if isinstance(index_lag, NoInput) else index_lag
        self.index_only = index_only
        super(IndexMixin, self).__init__(*args, **kwargs)
        self.end = self.payment if isinstance(end, NoInput) else end

    @property
    def real_cashflow(self) -> DualTypes:
        return -self.notional

    def cashflows(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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


class FXOptionPeriod(metaclass=ABCMeta):
    """
    Abstract base class for constructing volatility components of FXOptions.

    Pricing model uses Black 76 log-normal volatility calculations.

    Parameters
    -----------
    pair: str
        The currency pair for the FX rate which the option is settled. 3-digit code, e.g. "eurusd".
    expiry: datetime
        The expiry of the option: when the fixing and moneyness is determined.
    delivery: datetime
        The delivery date of the underlying FX pair. E.g. typically this would be **spot** as
        measured from the expiry date.
    payment: datetime
        The payment date of the premium associated with the option.
    strike: float, Dual, Dual2
        The strike value of the option.
    notional: float
        The amount in ccy1 (LHS) on which the option is based.
    option_fixing: float, optional
        If an option has already expired this argument is used to fix the price determined at
        expiry.
    delta_type: str in {"forward", "spot", "forward_pa", "spot_pa"}
        When deriving strike from a delta percentage the method used to associate the sensitivity
        to either a spot rate or a forward rate, possibly also premium adjusted.
    metric: str in {"pips", "percent"}, optional
        The pricing metric for the rate of the options.
    """

    # https://www.researchgate.net/publication/275905055_A_Guide_to_FX_Options_Quoting_Conventions/
    style: str = "european"
    kind: str
    phi: float

    @abstractmethod
    def __init__(
        self,
        pair: str,
        expiry: datetime,
        delivery: datetime,
        payment: datetime,
        strike: DualTypes | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        option_fixing: float | NoInput = NoInput(0),
        delta_type: str | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
    ) -> None:
        self.pair: str = pair.lower()
        self.currency: str = self.pair[3:]
        self.domestic: str = self.pair[:3]
        self.notional: float = defaults.notional if isinstance(notional, NoInput) else notional
        self.strike: DualTypes | NoInput = strike
        self.payment: datetime = payment
        self.delivery: datetime = delivery
        self.expiry: datetime = expiry
        self.option_fixing: float | NoInput = option_fixing
        self.delta_type: str = _drb(defaults.fx_delta_type, delta_type).lower()
        self.metric: str | NoInput = metric

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def cashflows(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: DualTypes | FXVols | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the properties of the period used in calculating cashflows.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            The base currency in which to express the NPV.
        local: bool,
            Whether to display NPV in a currency local to the object.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
            The percentage log-normal volatility to price the option.

        Returns
        -------
        dict
        """
        fx_, base = _get_fx_and_base(self.currency, fx, base)
        df, collateral = _dual_float(disc_curve_ccy2[self.payment]), disc_curve_ccy2.collateral
        npv_: dict[str, DualTypes] = self.npv(disc_curve, disc_curve_ccy2, fx, base, local=True, vol=vol)  # type: ignore[assignment]
        npv: float = _dual_float(npv_[self.currency])

        # TODO: (low-perf) get_vol is called twice for same value, once in npv and once for output
        # This method should not be called to get values used in later calculations becuase it
        # is not efficient. Prefer other ways to get values, i.e. by direct calculation calls.
        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: "Optionality",
            defaults.headers["pair"]: self.pair,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["expiry"]: self.expiry,
            defaults.headers["t_e"]: _dual_float(self._t_to_expiry(disc_curve_ccy2.node_dates[0])),
            defaults.headers["delivery"]: self.delivery,
            defaults.headers["rate"]: _dual_float(fx.rate(self.pair, self.delivery)),
            defaults.headers["strike"]: self.strike,
            defaults.headers["vol"]: _dual_float(self._get_vol_maybe_from_obj(vol, fx, disc_curve)),
            defaults.headers["model"]: "Black76",
            defaults.headers["payment"]: self.payment,
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["cashflow"]: npv / df,
            defaults.headers["df"]: df,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: npv * _dual_float(fx_),
            defaults.headers["collateral"]: collateral,
        }

    def npv(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: DualTypes | FXVols | NoInput = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *FXOption*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            The base currency in which to express the NPV.
        local: bool,
            Whether to display NPV in a currency local to the object.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
            The percentage log-normal volatility to price the option.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        if self.payment < disc_curve_ccy2.node_dates[0]:
            # payment date is in the past avoid issues with fixings or rates
            return _maybe_local(0.0, local, self.currency, NoInput(0), NoInput(0))

        if not isinstance(self.option_fixing, NoInput):
            if self.kind == "call" and self.strike < self.option_fixing:
                value = (self.option_fixing - self.strike) * self.notional
            elif self.kind == "put" and self.strike > self.option_fixing:
                value = (self.strike - self.option_fixing) * self.notional
            else:
                return _maybe_local(0.0, local, self.currency, NoInput(0), NoInput(0))
            value *= disc_curve_ccy2[self.payment]

        else:
            # value is expressed in currency (i.e. pair[3:])
            vol_ = self._get_vol_maybe_from_obj(vol, fx, disc_curve)

            value = _black76(
                F=fx.rate(self.pair, self.delivery),
                K=self.strike,
                t_e=self._t_to_expiry(disc_curve_ccy2.node_dates[0]),
                v1=None,  # not required: disc_curve[self.expiry],
                v2=disc_curve_ccy2[self.delivery],
                vol=vol_ / 100.0,
                phi=self.phi,  # controls calls or put price
            )
            value *= self.notional

        return _maybe_local(value, local, self.currency, fx, base)

    def rate(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the pricing metric of the *FXOption*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency. (Not used).
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            Not used by `rate`.
        local: bool,
            Not used by `rate`.
        vol: float, Dual, Dual2
            The percentage log-normal volatility to price the option.
        metric: str in {"pips", "percent"}
            The metric to return. If "pips" assumes the premium is in foreign (rhs)
            currency. If "percent", the premium is assumed to be domestic (lhs).

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        npv = self.npv(
            disc_curve,
            disc_curve_ccy2,
            fx,
            self.currency,
            False,
            vol,
        )

        if not isinstance(metric, NoInput):
            metric_ = metric.lower()
        elif not isinstance(self.metric, NoInput):
            metric_ = self.metric.lower()
        else:
            metric_ = defaults.fx_option_metric

        if metric_ == "pips":
            points_premium = (npv / disc_curve_ccy2[self.payment]) / self.notional
            return points_premium * 10000.0
        elif metric_ == "percent":
            currency_premium = (npv / disc_curve_ccy2[self.payment]) / fx.rate(
                self.pair,
                self.payment,
            )
            return currency_premium / self.notional * 100
        else:
            raise ValueError("`metric` must be in {'pips', 'percent'}")

    def implied_vol(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        premium: DualTypes | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
    ):
        """
        Calculate the implied volatility of the FX option.

        Parameters
        ----------
        disc_curve: Curve
            Not used by `implied_vol`.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            Not used by `implied_vol`.
        local: bool,
            Not used by `implied_vol`.
        premium: float, Dual, Dual2
            The premium value of the option paid at the appropriate payment date. Expressed
            either in *'pips'* or *'percent'* of notional. Must align with ``metric``.
        metric: str in {"pips", "percent"}, optional
            The manner in which the premium is expressed.

        Returns
        -------
        float, Dual or Dual2
        """
        # This function uses newton_1d and is AD safe.

        # convert the premium to a standardised immediate pips value.
        if metric == "percent":
            # convert premium to pips form
            premium = premium * fx.rate(self.pair, self.payment) * 100.0
        # convert to immediate pips form
        imm_premium = premium * disc_curve_ccy2[self.payment]

        t_e = self._t_to_expiry(disc_curve_ccy2.node_dates[0])
        v2 = disc_curve_ccy2[self.delivery]
        f_d = fx.rate(self.pair, self.delivery)

        def root(vol, f_d, k, t_e, v2, phi):
            f0 = _black76(f_d, k, t_e, None, v2, vol, phi) * 10000.0 - imm_premium
            sqrt_t = t_e**0.5
            d_plus = _d_plus_min_u(k / f_d, vol * sqrt_t, 0.5)
            f1 = v2 * dual_norm_pdf(phi * d_plus) * f_d * sqrt_t * 10000.0
            return f0, f1

        result = newton_1dim(root, 0.10, args=(f_d, self.strike, t_e, v2, self.phi))
        return result["g"] * 100.0

    def analytic_greeks(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float | NoInput = NoInput(0),
        premium: DualTypes | NoInput = NoInput(0),  # expressed in the payment currency
    ):
        r"""
        Return the different greeks for the *FX Option*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by `analytic_greeks`.
        local: bool,
            Not used by `analytic_greeks`.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
            The volatility used in calculation.
        premium: float, Dual, Dual2, optional
            The premium value of the option paid at the appropriate payment date.
            Premium should be expressed in domestic currency.
            If not given calculates and assumes a mid-market premium.

        Returns
        -------
        dict

        Notes
        -----
        **Delta** :math:`\Delta`

        This is the percentage value of the domestic notional in either the *forward* or *spot*
        FX rate. The choice of which is defined by the option's ``delta_type``.

        Delta is also expressed in nominal domestic currency amount.

        **Gamma** :math:`\Gamma`

        This defines by how much *delta* will change for a 1.0 increase in either the *forward*
        or *spot* FX rate. Which rate is determined by the option's ``delta_type``.

        Gamma is also expressed in nominal domestic currency amount for a +1% change in FX rates.

        **Vanna** :math:`\Delta_{\nu}`

        This defines by how much *delta* will change for a 1.0 increase (i.e. 100 log-vols) in
        volatility. The additional

        **Vega** :math:`\nu`

        This defines by how much the PnL of the option will change for a 1.0 increase in
        volatility for a nominal of 1 unit of domestic currency.

        Vega is also expressed in foreign currency for a 0.01 (i.e. 1 log-vol) move higher in vol.

        **Vomma (Volga)** :math:`\nu_{\nu}`

        This defines by how much *vega* will change for a 1.0 increase in volatility.

        These values can be used to estimate PnL for a change in the *forward* or
        *spot* FX rate and the volatility according to,

        .. math::

           \delta P \approx v_{deli} N^{dom} \left ( \Delta \delta f + \frac{1}{2} \Gamma \delta f^2 + \Delta_{\nu} \delta f \delta \sigma \right ) + N^{dom} \left ( \nu \delta \sigma + \frac{1}{2} \nu_{\nu} \delta \sigma^2 \right )

        where :math:`v_{deli}` is the date of FX settlement for *forward* or *spot* rate.

        **Kappa** :math:`\kappa`

        This defines by how much the PnL of the option will change for a 1.0 increase in
        strike for a nominal of 1 unit of domestic currency.

        **Kega** :math:`\left . \frac{dK}{d\sigma} \right|_{\Delta}`

        This defines the rate of change of strike with respect to volatility for a constant delta.

        Raises
        ------
        ValueError: if the ``strike`` is not set on the *Option*.
        """  # noqa: E501
        spot = fx.pairs_settlement[self.pair]
        w_spot = disc_curve[spot]
        w_deli = disc_curve[self.delivery]
        if self.delivery != self.payment:
            w_payment = disc_curve[self.payment]
        else:
            w_payment = w_deli
        v_deli = disc_curve_ccy2[self.delivery]
        v_spot = disc_curve_ccy2[spot]
        f_d = fx.rate(self.pair, self.delivery)
        f_t = fx.rate(self.pair, spot)
        u = self.strike / f_d
        sqrt_t = self._t_to_expiry(disc_curve.node_dates[0]) ** 0.5

        if isinstance(vol, FXVolObj):
            delta_idx, vol_, _ = vol.get_from_strike(self.strike, f_d, w_deli, w_spot, self.expiry)
        else:
            delta_idx, vol_ = None, vol
        vol_ /= 100.0
        vol_sqrt_t = vol_ * sqrt_t
        eta, z_w, z_u = _delta_type_constants(self.delta_type, w_deli / w_spot, u)
        d_eta = _d_plus_min_u(u, vol_sqrt_t, eta)
        d_plus = _d_plus_min_u(u, vol_sqrt_t, 0.5)
        d_min = _d_plus_min_u(u, vol_sqrt_t, -0.5)
        _is_spot = "spot" in self.delta_type

        _ = dict()
        _["delta"] = self._analytic_delta(
            premium,
            "_pa" in self.delta_type,
            z_u,
            z_w,
            d_eta,
            self.phi,
            d_plus,
            w_payment,
            w_spot,
            self.notional,
        )
        _[f"delta_{self.pair[:3]}"] = abs(self.notional) * _["delta"]
        _["gamma"] = self._analytic_gamma(
            _is_spot,
            v_deli,
            v_spot,
            z_w,
            self.phi,
            d_plus,
            f_d,
            vol_sqrt_t,
        )
        _[f"gamma_{self.pair[:3]}_1%"] = (
            _["gamma"] * abs(self.notional) * (f_t if _is_spot else f_d) * 0.01
        )
        _["vega"] = self._analytic_vega(v_deli, f_d, sqrt_t, self.phi, d_plus)
        _[f"vega_{self.pair[3:]}"] = _["vega"] * abs(self.notional) * 0.01
        _["vomma"] = self._analytic_vomma(_["vega"], d_plus, d_min, vol_)
        _["vanna"] = self._analytic_vanna(z_w, self.phi, d_plus, d_min, vol_)
        # _["vanna"] = self._analytic_vanna(_["vega"], _is_spot, f_t, f_d, d_plus, vol_sqrt_t)

        _["_kega"] = self._analytic_kega(
            z_u,
            z_w,
            eta,
            vol_,
            sqrt_t,
            f_d,
            self.phi,
            self.strike,
            d_eta,
        )
        _["_kappa"] = self._analytic_kappa(v_deli, self.phi, d_min)

        _["_delta_index"] = delta_idx
        _["__delta_type"] = self.delta_type
        _["__vol"] = vol_
        _["__strike"] = self.strike
        _["__forward"] = f_d
        _["__sqrt_t"] = sqrt_t
        _["__bs76"] = self._analytic_bs76(self.phi, v_deli, f_d, d_plus, self.strike, d_min)
        _["__notional"] = self.notional
        if self.phi > 0:
            _["__class"] = "FXCallPeriod"
        else:
            _["__class"] = "FXPutPeriod"
        return _

    @staticmethod
    def _analytic_vega(v_deli, f_d, sqrt_t, phi, d_plus):
        return v_deli * f_d * sqrt_t * dual_norm_pdf(phi * d_plus)

    @staticmethod
    def _analytic_vomma(vega, d_plus, d_min, vol):
        return vega * d_plus * d_min / vol

    @staticmethod
    def _analytic_gamma(spot, v_deli, v_spot, z_w, phi, d_plus, f_d, vol_sqrt_t):
        _ = z_w * dual_norm_pdf(phi * d_plus) / (f_d * vol_sqrt_t)
        if spot:
            return _ * z_w * v_spot / v_deli
        return _

    @staticmethod
    def _analytic_delta(premium, adjusted, z_u, z_w, d_eta, phi, d_plus, w_payment, w_spot, N_dom):
        if not adjusted or isinstance(premium, NoInput):
            # returns unadjusted delta or mid-market premium adjusted delta
            return z_u * z_w * phi * dual_norm_cdf(phi * d_eta)
        else:
            # returns adjusted delta with set premium in domestic (LHS) currency.
            # ASSUMES: if premium adjusted the premium is expressed in LHS currency.
            return z_w * phi * dual_norm_cdf(phi * d_plus) - w_payment / w_spot * premium / N_dom

    @staticmethod
    def _analytic_vanna(z_w, phi, d_plus, d_min, vol):
        return -z_w * dual_norm_pdf(phi * d_plus) * d_min / vol

    # @staticmethod
    # def _analytic_vanna(vega, spot, f_t, f_d, d_plus, vol_sqrt_t):  # Alternative monetary def.
    #     if spot:
    #         return vega / f_t * (1 - d_plus / vol_sqrt_t)
    #     else:
    #         return vega / f_d * (1 - d_plus / vol_sqrt_t)

    @staticmethod
    def _analytic_kega(z_u, z_w, eta, vol, sqrt_t, f_d, phi, k, d_eta):
        if eta < 0:
            # dz_u_du = 1.0
            x = vol * phi * dual_norm_cdf(phi * d_eta) / (f_d * z_u * dual_norm_pdf(phi * d_eta))
        else:
            x = 0.0

        _ = (d_eta - 2.0 * eta * sqrt_t * vol) / (-1 / (k * sqrt_t) + x)
        return _

    @staticmethod
    def _analytic_kappa(v_deli, phi, d_min):
        return -v_deli * phi * dual_norm_cdf(phi * d_min)

    @staticmethod
    def _analytic_bs76(phi, v_deli, f_d, d_plus, k, d_min):
        return phi * v_deli * (f_d * dual_norm_cdf(phi * d_plus) - k * dual_norm_cdf(phi * d_min))

    def _strike_and_index_from_atm(
        self,
        delta_type: str,
        vol: DualTypes | FXVols,
        w_deli,
        w_spot,
        f,
        t_e,
    ):
        if not isinstance(vol, FXVolObj):
            vol_delta_type = delta_type  # set delta types as being equal if the vol is a constant.
        else:
            if isinstance(vol, FXDeltaVolSurface):
                # convert a Surface to Smile for simplified calculations below.
                vol = vol.get_smile(self.expiry)
            vol_delta_type = vol.delta_type

        z_w = w_deli / w_spot
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, None)
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, None)

        # u, delta_idx, delta =
        # self._moneyness_from_delta_three_dimensional(delta_type, vol, t_e, z_w)

        if eta_0 == 0.5:  # then delta type is unadjusted
            if eta_1 == 0.5:  # then smile delta type matches: closed form eqn available
                if isinstance(vol, FXDeltaVolSmile):
                    delta_idx = z_w_1 / 2.0
                    vol = vol[delta_idx]
                else:
                    delta_idx = None
                u = self._moneyness_from_atm_delta_closed_form(vol, t_e)
            else:  # then smile delta type unmatched: 2-d solver required
                delta = z_w_0 * self.phi / 2.0
                u, delta_idx = self._moneyness_from_delta_two_dimensional(
                    delta,
                    delta_type,
                    vol,
                    t_e,
                    z_w,
                )
        else:  # then delta type is adjusted,
            if eta_1 == -0.5:  # then smile type matches: use 1-d solver
                u = self._moneyness_from_atm_delta_one_dimensional(
                    delta_type,
                    vol_delta_type,
                    vol,
                    t_e,
                    z_w,
                )
                delta_idx = z_w_1 * u * 0.5
            else:  # smile delta type unmatched: 2-d solver required
                u, delta_idx = self._moneyness_from_atm_delta_two_dimensional(
                    delta_type,
                    vol,
                    t_e,
                    z_w,
                )

        return u * f, delta_idx

    def _strike_and_index_from_delta(
        self,
        delta: float,
        delta_type: str,
        vol: DualTypes | FXVols,
        w_deli,
        w_spot,
        f,
        t_e,
    ):
        if not isinstance(vol, FXVolObj):
            vol_delta_type = delta_type  # set delta types as being equal if the vol is a constant.
        else:
            if isinstance(vol, FXDeltaVolSurface):
                vol = vol.get_smile(self.expiry)
            vol_delta_type = vol.delta_type

        z_w = w_deli / w_spot
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, None)
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, None)

        # then delta types are both unadjusted, used closed form.
        if eta_0 == eta_1 and eta_0 == 0.5:
            if isinstance(vol, FXDeltaVolSmile):
                delta_idx = (-z_w_1 / z_w_0) * (delta - 0.5 * z_w_0 * (self.phi + 1.0))
                vol = vol[delta_idx]
            else:
                delta_idx = None
            u = self._moneyness_from_delta_closed_form(delta, vol, t_e, z_w_0)

        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = self._moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                vol_delta_type,
                vol,
                t_e,
                z_w,
            )
            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * u * (self.phi + 1.0) * 0.5)
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = self._moneyness_from_delta_two_dimensional(
                delta,
                delta_type,
                vol,
                t_e,
                z_w,
            )

        return u * f, delta_idx

    def _moneyness_from_atm_delta_closed_form(self, vol: DualTypes, t_e: DualTypes):
        """
        Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

        This function preserves AD.

        Parameters
        -----------
        vol: float, Dual, Dual2
            The volatility (in %, e.g. 10.0) to use in calculations.
        t_e: float, Dual, Dual2
            The time to expiry.

        Returns
        -------
        float, Dual or Dual2
        """
        _ = dual_exp((vol / 100.0) ** 2 * t_e / 2.0)
        return _

    def _moneyness_from_delta_closed_form(
        self,
        delta: float,
        vol: DualTypes,
        t_e: DualTypes,
        z_w_0: DualTypes,
    ):
        """
        Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

        This function preserves AD.

        Parameters
        -----------
        delta: float
            The input unadjusted delta for which to determine the moneyness for.
        vol: float, Dual, Dual2
            The volatility (in %, e.g. 10.0) to use in calculations.
        t_e: float, Dual, Dual2
            The time to expiry.
        z_w_0: float, Dual, Dual2
            The scalar for 'spot' or 'forward' delta types.
            If 'forward', this should equal 1.0.
            If 'spot', this should be :math:`w_deli / w_spot`.

        Returns
        -------
        float, Dual or Dual2
        """
        vol_sqrt_t = vol * t_e**0.5 / 100.0
        _ = dual_inv_norm_cdf(self.phi * delta / z_w_0)
        _ = dual_exp(vol_sqrt_t * (0.5 * vol_sqrt_t - self.phi * _))
        return _

    def _moneyness_from_atm_delta_one_dimensional(
        self,
        delta_type: str,
        vol_delta_type: str,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ):
        def root1d(g, delta_type, vol_delta_type, phi, sqrt_t_e, z_w, ad):
            u = g

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0

            delta_idx = z_w_1 * z_u_0 / 2.0
            if isinstance(vol, FXDeltaVolSmile):
                vol_ = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

            # Calculate derivative values
            ddelta_idx_du = dz_u_0_du * z_w_1 * 0.5

            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd_du = (
                -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du
            )

            nd0 = dual_norm_pdf(phi * d0)
            f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

            return f0, f1

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        else:
            avg_vol = vol
        g01 = self.phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_1dim(
            root1d,
            g00,
            args=(delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u = root_solver["g"]
        return u

    def _moneyness_from_delta_one_dimensional(
        self,
        delta,
        delta_type: str,
        vol_delta_type: str,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ):
        def root1d(g, delta, delta_type, vol_delta_type, phi, sqrt_t_e, z_w, ad):
            u = g

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0

            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * z_u_0 * (phi + 1.0) * 0.5)
            if isinstance(vol, FXDeltaVolSmile):
                vol_ = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0 = delta - z_w_0 * z_u_0 * phi * _phi0

            # Calculate derivative values
            ddelta_idx_du = dz_u_0_du * z_w_1 * (phi + 1.0) * 0.5

            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd_du = (
                -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du
            )

            nd0 = dual_norm_pdf(phi * d0)
            f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

            return f0, f1

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        else:
            avg_vol = vol
        g01 = delta if self.phi > 0 else max(delta, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        msg = (
            f"If the delta, {delta:.1f}, is premium adjusted for a call option is it infeasible?"
            if self.phi > 0
            else ""
        )
        try:
            root_solver = newton_1dim(
                root1d,
                g00,
                args=(delta, delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
                pre_args=(0,),
                final_args=(1,),
            )
        except ValueError as e:
            raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

        if root_solver["state"] == -1:
            raise ValueError(
                f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
            )

        u = root_solver["g"]
        return u

    def _moneyness_from_delta_two_dimensional(
        self,
        delta,
        delta_type,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ):
        def root2d(g, delta, delta_type, vol_delta_type, phi, sqrt_t_e, z_w, ad):
            u, delta_idx = g[0], g[1]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            vol_ = vol[delta_idx] / 100.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0 = delta - z_w_0 * z_u_0 * phi * _phi0

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

            # Calculate Jacobian values
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00 = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du
            f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
            f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
            f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

            return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

        avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        g01 = delta if self.phi > 0 else max(delta, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        msg = (
            f"If the delta, {_dual_float(delta):.1f}, is premium adjusted for a "
            "call option is it infeasible?"
            if self.phi > 0
            else ""
        )
        try:
            root_solver = newton_ndim(
                root2d,
                [g00, abs(g01)],
                args=(delta, delta_type, vol.delta_type, self.phi, t_e**0.5, z_w),
                pre_args=(0,),
                final_args=(1,),
                raise_on_fail=False,
            )
        except ValueError as e:
            raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

        if root_solver["state"] == -1:
            raise ValueError(
                f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
            )
        u, delta_idx = root_solver["g"][0], root_solver["g"][1]
        return u, delta_idx

    def _moneyness_from_atm_delta_two_dimensional(
        self,
        delta_type,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ):
        def root2d(g, delta_type, vol_delta_type, phi, sqrt_t_e, z_w, ad):
            u, delta_idx = g[0], g[1]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            vol_ = vol[delta_idx] / 100.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

            # Calculate Jacobian values
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00 = phi * z_w_0 * dz_u_0_du * (0.5 - _phi0) - z_w_0 * z_u_0 * nd0 * dd_du
            f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
            f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
            f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

            return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

        avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        g01 = self.phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_ndim(
            root2d,
            [g00, abs(g01)],
            args=(delta_type, vol.delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u, delta_idx = root_solver["g"][0], root_solver["g"][1]
        return u, delta_idx

    def _moneyness_from_delta_three_dimensional(
        self,
        delta_type,
        vol: float | FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ):
        """
        Solve the ATM delta problem where delta is not explicit.
        """

        def root3d(g, delta_type, vol_delta_type, phi, sqrt_t_e, z_w, ad):
            u, delta_idx, delta = g[0], g[1], g[2]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            if isinstance(vol, FXDeltaVolSmile):
                vol_ = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0 = delta - z_w_0 * z_u_0 * phi * _phi0

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

            f0_2 = delta - phi * z_u_0 * z_w_0 / 2.0

            # Calculate Jacobian values
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00 = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du  # dh0/du
            f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du  # dh1/du
            f1_20 = -phi * z_w_0 * dz_u_0_du / 2.0  # dh2/du
            f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx  # dh0/ddidx
            f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx  # dh1/ddidx
            f1_21 = 0.0  # dh2/ddidx
            f1_02 = 1.0  # dh0/ddelta
            f1_12 = 0.0  # dh1/ddelta
            f1_22 = 1.0  # dh2/ddelta

            return [f0_0, f0_1, f0_2], [
                [f1_00, f1_01, f1_02],
                [f1_10, f1_11, f1_12],
                [f1_20, f1_21, f1_22],
            ]

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
            vol_delta_type = vol.delta_type
        else:
            avg_vol = vol
            vol_delta_type = self.delta_type
        g02 = 0.5 * self.phi * (z_w if "spot" in delta_type else 1.0)
        g01 = g02 if self.phi > 0 else max(g02, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_ndim(
            root3d,
            [g00, abs(g01), g02],
            args=(delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u, delta_idx, delta = root_solver["g"][0], root_solver["g"][1], root_solver["g"][1]
        return u, delta_idx, delta

    def _get_vol_maybe_from_obj(
        self,
        vol: FXVols | DualTypes,
        fx: FXForwards,
        disc_curve: Curve,
    ) -> DualTypes:
        """Return a volatility for the option from a given Smile."""
        if isinstance(vol, FXVolObj):
            spot = fx.pairs_settlement[self.pair]
            f = fx.rate(self.pair, self.delivery)
            _, vol_, _ = vol.get_from_strike(
                self.strike,
                f,
                disc_curve[self.delivery],
                disc_curve[spot],
                self.expiry,
            )
        else:
            vol_ = vol

        return vol_

    def _t_to_expiry(self, now: datetime):
        # TODO make this a dual, associated with theta
        return (self.expiry - now).days / 365.0

    def _payoff_at_expiry(self, range: list[float] | NoInput = NoInput(0)):
        if isinstance(self.strike, NoInput):
            raise ValueError(
                "Cannot return payoff for option without a specified `strike`.",
            )  # pragma: no cover
        if isinstance(range, NoInput):
            x = np.linspace(0, 20, 1001)
        else:
            x = np.linspace(range[0], range[1], 1001)
        _ = (x - self.strike) * self.phi
        __ = np.zeros(1001)
        if self.phi > 0:  # call
            y = np.where(x < self.strike, __, _) * self.notional
        else:  # put
            y = np.where(x > self.strike, __, _) * self.notional
        return x, y


class FXCallPeriod(FXOptionPeriod):
    """
    Create an FXCallPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    kind = "call"
    phi = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class FXPutPeriod(FXOptionPeriod):
    """
    Create an FXPutPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    kind = "put"
    phi = -1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def _float_or_none(val: DualTypes | None) -> float | None:
    if val is None:
        return None
    else:
        return _dual_float(val)


def _get_ibor_curve_from_dict(months: int, d: dict[str, Curve]) -> Curve:
    try:
        return d[f"{months}m"]
    except KeyError:
        try:
            return d[f"{months}M"]
        except KeyError:
            raise ValueError(
                "If supplying `curve` as dict must provide a tenor mapping key and curve for"
                f"the frequency of the given Period. The missing mapping is '{months}m'."
            )


def _maybe_get_rfr_curve_from_dict(curve: Curve | dict[str, Curve] | NoInput) -> Curve | NoInput:
    if isinstance(curve, dict):
        return _get_rfr_curve_from_dict(curve)
    else:
        return curve


def _get_rfr_curve_from_dict(d: dict[str, Curve]) -> Curve:
    for s in ["rfr", "RFR", "Rfr"]:
        try:
            ret: Curve = d[s]
        except KeyError:
            continue
        else:
            return ret
    raise ValueError(
        "A `curve` supplied as dict to an RFR based period must contain a key entry 'rfr'."
    )


def _trim_df_by_index(
    df: DataFrame, left: datetime | NoInput, right: datetime | NoInput
) -> DataFrame:
    """
    Used by fixings_tables to constrict the view to a left and right bound
    """
    if len(df.index) == 0 or (isinstance(left, NoInput) and isinstance(right, NoInput)):
        return df
    elif isinstance(left, NoInput):
        return df[:right]  # type: ignore[misc]
    elif isinstance(right, NoInput):
        return df[left:]  # type: ignore[misc]
    else:
        return df[left:right]  # type: ignore[misc]


# def _validate_broad_delta_bounds(phi, delta, delta_type):
#     if phi < 0 and "_pa" in delta_type:
#         assert delta <= 0.0
#     elif phi < 0:
#         assert -1.0 <= delta <= 0.0
#     elif
