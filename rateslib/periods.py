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

from abc import abstractmethod, ABCMeta
from datetime import datetime
from typing import Optional, Union
import warnings
from math import comb, log

import numpy as np

# from pandas.tseries.offsets import CustomBusinessDay
from pandas import DataFrame, date_range, Series, NA, isna, notna

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.calendars import add_tenor, dcf, _get_eom, _is_holiday, CalInput
from rateslib.curves import (
    Curve,
    LineCurve,
    IndexCurve,
    average_rate,
    CompositeCurve,
    index_left,
)
from rateslib.dual import Dual, Dual2, DualTypes, gradient
from rateslib.fx import FXForwards, FXRates


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_fx_and_base(
    currency: str,
    fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
    base: Union[str, NoInput] = NoInput(0),
):
    if fx is None:
        raise NotImplementedError("TraceBack for NoInput")
    if base is None:
        raise NotImplementedError("TraceBack for NoInput")

    if isinstance(fx, (FXRates, FXForwards)):
        base = fx.base if base is NoInput.blank else base.lower()
        if base == currency:
            fx = 1.0
        else:
            fx = fx.rate(pair=f"{currency}{base}")
    elif base is not NoInput.blank:  # and fx is then a float or None
        if fx is NoInput.blank:
            if base.lower() != currency.lower():
                raise ValueError(
                    f"`base` ({base}) cannot be requested without supplying `fx` as a "
                    "valid FXRates or FXForwards object to convert to "
                    f"currency ({currency}).\n"
                    "If you are using a `Solver` with multi-currency instruments have you "
                    "forgotten to attach the FXForwards in the solver's `fx` argument?"
                )
            fx = 1.0
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
            fx = fx
    else:  # base is None and fx is float or None.
        if fx is NoInput.blank:
            fx = 1.0
        else:
            if abs(fx - 1.0) < 1e-10:
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
            fx = fx

    return fx, base


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
        notional: Union[float, NoInput] = NoInput(0),
        currency: Union[str, NoInput] = NoInput(0),
        convention: Union[str, NoInput] = NoInput(0),
        termination: Union[datetime, NoInput] = NoInput(0),
        stub: bool = False,
        roll: Union[int, str, NoInput] = NoInput(0),
        calendar: CalInput = NoInput(0),
    ):
        if end < start:
            raise ValueError("`end` cannot be before `start`.")
        self.start, self.end, self.payment = start, end, payment
        self.frequency = frequency.upper()
        self.notional = defaults.notional if notional is NoInput.blank else notional
        self.currency = defaults.base_currency if currency is NoInput.blank else currency.lower()
        self.convention = defaults.convention if convention is NoInput.blank else convention
        self.termination = termination
        self.freq_months = defaults.frequency_months[self.frequency]
        self.stub = stub
        self.roll = roll
        self.calendar = calendar

    def __repr__(self):
        return (
            f"<{type(self).__name__}: {self.start.strftime('%Y-%m-%d')}->"
            f"{self.end.strftime('%Y-%m-%d')},{self.notional},{self.convention}>"
        )

    @property
    def dcf(self):
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
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
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
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)
        _ = fx * self.notional * self.dcf * disc_curve_[self.payment] / 10000
        return _

    @abstractmethod
    def cashflows(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ) -> dict:
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
        disc_curve: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        if disc_curve is NoInput.blank:
            df, collateral = None, None
        else:
            df, collateral = float(disc_curve[self.payment]), disc_curve.collateral

        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: "Stub" if self.stub else "Regular",
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["a_acc_start"]: self.start,
            defaults.headers["a_acc_end"]: self.end,
            defaults.headers["payment"]: self.payment,
            defaults.headers["convention"]: self.convention,
            defaults.headers["dcf"]: self.dcf,
            defaults.headers["notional"]: float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["collateral"]: collateral,
        }

    @abstractmethod
    def npv(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ) -> Union[DualTypes, dict[str, DualTypes]]:
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

    def __init__(self, *args, fixed_rate: Union[float, NoInput] = NoInput(0), **kwargs):
        self.fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs) -> DualTypes:
        """
        Return the analytic delta of the *FixedPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return super().analytic_delta(*args, **kwargs)

    @property
    def cashflow(self) -> Union[float, None]:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional.
        """
        if self.fixed_rate is NoInput.blank:
            return None
        else:
            return -self.notional * self.dcf * self.fixed_rate / 100

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def npv(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ) -> DualTypes:
        """
        Return the NPV of the *FixedPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_from_curve(curve, disc_curve)
        if not isinstance(disc_curve, Curve) and curve is NoInput.blank:
            raise TypeError("`curves` have not been supplied correctly.")
        value = self.cashflow * disc_curve_[self.payment]
        if local:
            return {self.currency: value}
        else:
            fx, _ = _get_fx_and_base(self.currency, fx, base)
            return fx * value

    def cashflows(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ) -> dict:
        """
        Return the cashflows of the *FixedPeriod*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve_ is NoInput.blank or self.fixed_rate is NoInput.blank:
            npv = None
            npv_fx = None
        else:
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)

        cashflow = None if self.cashflow is None else float(self.cashflow)
        return {
            **super().cashflows(curve, disc_curve_, fx, base),
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }


def _validate_float_args(
    fixing_method: Union[str, NoInput],
    method_param: Union[int, NoInput],
    spread_compound_method: Union[str, NoInput],
):
    """
    Validate the argument input to float periods.

    Returns
    -------
    tuple
    """
    if fixing_method is NoInput.blank:
        fixing_method_: str = defaults.fixing_method
    else:
        fixing_method_ = fixing_method.lower()
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
            f"got '{fixing_method_}'."
        )

    if method_param is NoInput.blank:
        method_param_: int = defaults.fixing_method_param[fixing_method_]
    else:
        method_param_ = method_param
    if method_param_ != 0 and fixing_method_ == "rfr_payment_delay":
        raise ValueError(
            "`method_param` should not be used (or a value other than 0) when "
            f"using a `fixing_method` of 'rfr_payment_delay', got {method_param_}. "
            f"Configure the `payment_lag` option instead to have the "
            f"appropriate effect."
        )
    elif fixing_method_ == "rfr_lockout" and method_param_ < 1:
        raise ValueError(
            f'`method_param` must be >0 for "rfr_lockout" `fixing_method`, ' f"got {method_param_}"
        )

    if spread_compound_method is NoInput.blank:
        spread_compound_method_: str = defaults.spread_compound_method
    else:
        spread_compound_method_ = spread_compound_method.lower()
    if spread_compound_method_ not in [
        "none_simple",
        "isda_compounding",
        "isda_flat_compounding",
    ]:
        raise ValueError(
            "`spread_compound_method` must be in {'none_simple', "
            "'isda_compounding', 'isda_flat_compounding'}, "
            f"got {spread_compound_method_}"
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
        *args,
        float_spread: Union[float, NoInput] = NoInput(0),
        fixings: Union[float, list, Series, NoInput] = NoInput(0),
        fixing_method: Union[str, NoInput] = NoInput(0),
        method_param: Union[int, NoInput] = NoInput(0),
        spread_compound_method: Union[str, NoInput] = NoInput(0),
        **kwargs,
    ):
        self.float_spread = 0.0 if float_spread is NoInput.blank else float_spread

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
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the analytic delta of the *FloatPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        if self.spread_compound_method == "none_simple" or self.float_spread == 0:
            # then analytic_delta is not impacted by float_spread compounding
            dr_dz = 1.0
        elif isinstance(curve, Curve):
            _ = self.float_spread
            DualType = Dual if curve.ad in [0, 1] else Dual2
            DualArgs = ([],) if curve.ad in [0, 1] else ([], [])
            self.float_spread = DualType(float(_), ["z_float_spread"], *DualArgs)
            rate = self.rate(curve)
            dr_dz = gradient(rate, ["z_float_spread"])[0] * 100
            self.float_spread = _
        else:
            raise TypeError("`curve` must be supplied for given `spread_compound_method`")

        return dr_dz * super().analytic_delta(curve, disc_curve, fx, base)

    def cashflows(
        self,
        curve: Union[Curve, dict, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the cashflows of the *FloatPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx, base = _get_fx_and_base(self.currency, fx, base)
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)

        if curve is not NoInput.blank:
            cashflow = float(self.cashflow(curve))
            rate = float(100 * cashflow / (-self.notional * self.dcf))
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)
        else:
            cashflow, rate, npv, npv_fx = None, None, None, None

        return {
            **super().cashflows(curve, disc_curve_, fx, base),
            defaults.headers["rate"]: rate,
            defaults.headers["spread"]: float(self.float_spread),
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def npv(
        self,
        curve: Union[Curve, dict, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the cashflows of the *FloatPeriod*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        if not isinstance(disc_curve_, Curve) or curve is NoInput.blank:
            raise TypeError("`curves` have not been supplied correctly.")
        if self.payment < disc_curve_.node_dates[0]:
            if local:
                return {self.currency: 0.0}
            else:
                return 0.0  # payment date is in the past avoid issues with fixings or rates
        value = self.rate(curve) / 100 * self.dcf * disc_curve_[self.payment] * -self.notional
        if local:
            return {self.currency: value}
        else:
            fx, _ = _get_fx_and_base(self.currency, fx, base)
            return fx * value

    def cashflow(self, curve: Union[Curve, LineCurve, dict]) -> Union[None, DualTypes]:
        if curve is None:
            return None
        else:
            rate = None if curve is None else self.rate(curve)
            _ = -self.notional * self.dcf * rate / 100
            return _

    def rate(self, curve: Union[Curve, LineCurve, dict]):
        """
        Calculating the floating rate for the period.

        Parameters
        ----------
        curve : Curve, LineCurve, IndexCurve, dict of curves
            The forecasting curve object.

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
        if isinstance(self.fixings, (float, Dual, Dual2)):
            # if fixings is a single value then return that value (curve unused)
            if (
                self.spread_compound_method in ["isda_compounding", "isda_flat_compounding"]
                and self.float_spread != 0
                and "rfr" in self.fixing_method
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

        if "rfr" in self.fixing_method:
            method = {
                "dfs": self._rfr_rate_from_df_curve,
                "values": self._rfr_rate_from_line_curve,
            }
            try:
                return method[curve._base_type](curve)
            except AttributeError:
                raise ValueError(
                    "Must supply a valid curve for forecasting.\n"
                    "Do not supply a dict of curves for RFR based methods."
                )
        elif "ibor" in self.fixing_method:
            method = {
                "dfs": self._ibor_rate_from_df_curve,
                "values": self._ibor_rate_from_line_curve,
            }
            if not isinstance(curve, dict):
                return method[curve._base_type](curve)
            else:
                if not self.stub:
                    curve = curve[f"{self.freq_months}m"]
                    return method[curve._base_type](curve)
                else:
                    return self._interpolated_ibor_from_curve_dict(curve)
        else:
            raise ValueError("`fixing_method` not valid for the FloatPeriod.")  # pragma: no cover

    def _interpolated_ibor_from_curve_dict(self, curve: dict):
        """
        Get the rate on all available curves in dict and then determine the ones to interpolate.
        """
        calendar = next(iter(curve.values())).calendar  # note: ASSUMES all curve calendars are same
        fixing_date = add_tenor(self.start, f"-{self.method_param}B", NoInput(0), calendar)

        def _rate(c: Union[Curve, LineCurve, IndexCurve], tenor):
            if c._base_type == "dfs":
                return c.rate(self.start, tenor)
            else:  # values
                return c.rate(fixing_date)

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
            _ = rates[i] + (rates[i + 1] - rates[i]) * (
                (self.end - dates[i]) / (dates[i + 1] - dates[i])
            )
            return _

    def _ibor_rate_from_df_curve(self, curve: Curve):
        # the compounding method has no effect on single rate (ibor) fixings.
        if isinstance(self.fixings, Series):
            # check if we return published IBOR rate
            fixing_date = add_tenor(self.start, f"-{self.method_param}B", None, curve.calendar)
            try:
                return self.fixings[fixing_date] + self.float_spread / 100
            except KeyError:
                # TODO warn if Series contains close dates but cannot find a value for exact date.
                # fixing not available: use curve
                pass
        elif isinstance(self.fixings, list):  # this is also validated in __init__
            raise ValueError("`fixings` cannot be supplied as list, under 'ibor' `fixing_method`.")

        if self.stub:
            r = curve.rate(self.start, self.end) + self.float_spread / 100
        else:
            r = curve.rate(self.start, f"{self.freq_months}m") + self.float_spread / 100
        return r

    def _ibor_rate_from_line_curve(self, curve: LineCurve):
        # the compounding method has no effect on single rate (ibor) fixings.
        fixing_date = add_tenor(self.start, f"-{self.method_param}B", NoInput(0), curve.calendar)
        if isinstance(self.fixings, Series):
            try:
                return self.fixings[fixing_date] + self.float_spread / 100
            except KeyError:
                # TODO warn if Series contains close dates but cannot find a value for exact date.
                # fixing not available: use curve
                pass
        elif isinstance(self.fixings, list):  # this is also validated in __init__
            raise ValueError("`fixings` cannot be supplied as list, under 'ibor' `fixing_method`.")

        return curve[fixing_date] + self.float_spread / 100

    def _rfr_rate_from_df_curve(self, curve: Curve):
        if self.fixing_method == "rfr_payment_delay" and not self._is_inefficient:
            return curve.rate(self.start, self.end) + self.float_spread / 100

        elif self.fixing_method == "rfr_observation_shift" and not self._is_inefficient:
            start = add_tenor(self.start, f"-{self.method_param}b", "P", curve.calendar)
            end = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            return curve.rate(start, end) + self.float_spread / 100
            # TODO: (low:perf) semi-efficient method for lockout under certain conditions
        else:
            # return inefficient calculation
            # this is also the path for all averaging methods
            return self._rfr_fixings_array(curve, fixing_exposure=False)[0]

    def _rfr_rate_from_line_curve(self, curve: LineCurve):
        return self._rfr_fixings_array(curve, fixing_exposure=False)[0]

    def _avg_rate_with_spread(self, rates, dcf_vals):
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
        dcf_vals = dcf_vals.set_axis(rates.index)
        if self.spread_compound_method != "none_simple":
            raise ValueError(
                "`spread_compound` method must be 'none_simple' in an RFR averaging " "period."
            )
        else:
            return (dcf_vals * rates).sum() / dcf_vals.sum() + self.float_spread / 100

    def _isda_compounded_rate_with_spread(self, rates, dcf_vals):
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
        dcf_vals = dcf_vals.set_axis(rates.index)
        if self.float_spread == 0 or self.spread_compound_method == "none_simple":
            return (
                (1 + dcf_vals * rates / 100).prod() - 1
            ) * 100 / dcf_vals.sum() + self.float_spread / 100
        elif self.spread_compound_method == "isda_compounding":
            return (
                ((1 + dcf_vals * (rates / 100 + self.float_spread / 10000)).prod() - 1)
                * 100
                / dcf_vals.sum()
            )
        elif self.spread_compound_method == "isda_flat_compounding":
            sub_cashflows = (rates / 100 + self.float_spread / 10000) * dcf_vals
            C_i = 0.0
            for i in range(1, len(sub_cashflows)):
                C_i += sub_cashflows.iloc[i - 1]
                sub_cashflows.iloc[i] += C_i * rates.iloc[i] / 100 * dcf_vals.iloc[i]
            total_cashflow = sub_cashflows.sum()
            return total_cashflow * 100 / dcf_vals.sum()
        else:
            # this path not generally hit due to validation at initialisation
            raise ValueError(
                "`spread_compound_method` must be in {'none_simple', "
                "'isda_compounding', 'isda_flat_compounding'}."
            )

    def fixings_table(
        self,
        curve: Union[Curve, LineCurve, dict],
        approximate: bool = False,
        disc_curve: Curve = NoInput(0),
    ):
        """
        Return a DataFrame of fixing exposures.

        Parameters
        ----------
        curve : Curve, LineCurve, IndexCurve dict of such
            The forecast needed to calculate rates which affect compounding and
            dependent notional exposure.
        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        disc_curve : Curve
            A curve to make appropriate DF scalings. If *None* and ``curve`` contains
            DFs that will be used instead, otherwise errors are raised.

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
               calendar="bus"
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

            ibor_curve = Curve(
               nodes={dt(2022, 1, 1): 1.00, dt(2023, 1, 1): 0.99},
               calendar="bus",
           )
           period = FloatPeriod(**{
               **constants,
               "fixing_method": "ibor",
               "method_param": 2,
            })
           period.fixings_table(ibor_curve)
        """
        if disc_curve is NoInput.blank and isinstance(curve, dict):
            raise ValueError("Cannot infer `disc_curve` from a dict of curves.")
        elif disc_curve is NoInput.blank and curve._base_type == "dfs":
            disc_curve = curve

        if approximate:
            if self.fixings is not NoInput.blank:
                warnings.warn(
                    "Cannot approximate a fixings table when some published fixings "
                    f"are given within the period {self.start.strftime('%d-%b-%Y')}->"
                    f"{self.end.strftime('%d-%b-%Y')}. Switching to exact mode for this "
                    f"period.",
                    UserWarning,
                )
            else:
                return self._fixings_table_fast(curve, disc_curve)

        if "rfr" in self.fixing_method:
            rate, table = self._rfr_fixings_array(
                curve,
                fixing_exposure=True,
                disc_curve=disc_curve,
            )
            table = table.iloc[:-1]
            return table[["obs_dates", "notional", "dcf", "rates"]].set_index("obs_dates")
        elif "ibor" in self.fixing_method:
            if isinstance(curve, dict):
                calendar = next(iter(curve.values())).calendar
            else:
                calendar = curve.calendar
            fixing_date = add_tenor(self.start, f"-{self.method_param}b", "P", calendar)
            return DataFrame(
                {
                    "obs_dates": [fixing_date],
                    "notional": -self.notional,
                    "dcf": [None],
                    "rates": [self.rate(curve)],
                }
            ).set_index("obs_dates")

    def _rfr_fixings_array(
        self,
        curve: Union[Curve, LineCurve],
        fixing_exposure: bool = False,
        disc_curve: Curve = None,
    ):
        """
        Calculate the rate of a period via extraction and combination of every fixing.

        This method of calculation is inefficient and used when either:

        - known fixings needs to be combined with unknown fixings,
        - the fixing_method is of a type that needs individual fixing data,
        - the spread compound method is of a type that needs individual fixing data.

        Parameters
        ----------
        curve : Curve or LineCurve
            The forecasting curve used to extract the fixing data.
        fixing_exposure : bool
            Whether to calculate sensitivities to the fixings additionally.
        fixing_exposure_approx : bool
            Whether to use an approximation, if available, for fixing exposure calcs.

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

        obs_dates, dcf_dates = self._get_method_dcf_markers(curve)

        dcf_vals = Series(
            [  # calculate the dcf values from the dcf dates
                dcf(dcf_dates[i], dcf_dates[i + 1], curve.convention)
                for i in range(len(dcf_dates.index) - 1)
            ]
        )

        rates = Series(NA, index=obs_dates[:-1])
        if self.fixings is not NoInput.blank:
            # then fixings will be a list or Series, scalars are already processed.
            if isinstance(self.fixings, list):
                rates.iloc[: len(self.fixings)] = self.fixings
            elif isinstance(self.fixings, Series):
                if not self.fixings.index.is_monotonic_increasing:
                    raise ValueError(
                        "`fixings` as a Series must have a monotonically increasing "
                        "datetimeindex."
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
                        f"Series.pop().\n{e}"
                    )

                # basic error checking for missing fixings and provide warning.
                try:
                    first_forecast_date = rates[isna(rates)].index[0]
                    if rates[~isna(rates)].index[-1] > first_forecast_date:
                        # then a missing fixing exists
                        warnings.warn(
                            f"`fixings` has missed a calendar value "
                            f"({first_forecast_date}) which "
                            "has been inferred from the curve. Subsequent "
                            "fixings have been detected",
                            UserWarning,
                        )
                except (KeyError, IndexError):
                    pass
            else:
                raise TypeError(
                    "`fixings` should be of type scalar, None, list or Series."
                )  # pragma: no cover

        # reindex the rates series getting missing values from the curves
        # TODO (low) the next two lines could probably be vectorised and made more efficient.
        fixed = ~isna(rates)
        rates = Series({k: v if notna(v) else curve.rate(k, "1b", "F") for k, v in rates.items()})

        if fixing_exposure:
            # need to calculate the dcfs associated with the rates (unshifted)
            if self.fixing_method in [
                "rfr_payment_delay",
                "rfr_observation_shift",
                "rfr_lockout",
                "rfr_payment_delay_avg",
                "rfr_observation_shift_avg",
                "rfr_lockout_avg",
            ]:  # for all these methods there is no shift
                dcf_of_r = dcf_vals.copy()
            elif self.fixing_method in ["rfr_lookback", "rfr_lookback_avg"]:
                dcf_of_r = Series(
                    [
                        dcf(obs_dates[i], obs_dates[i + 1], curve.convention)
                        for i in range(len(dcf_dates.index) - 1)
                    ]
                )
            v_with_r = Series([disc_curve[obs_dates[i]] for i in range(1, len(dcf_dates.index))])

        if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
            # adjust the final rates values of the lockout arrays according to param
            try:
                rates.iloc[-self.method_param :] = rates.iloc[-self.method_param - 1]
            except IndexError:
                raise ValueError("period has too few dates for `rfr_lockout` param to function.")

        if fixing_exposure:
            rates_dual = Series(
                [Dual(float(r), [f"fixing_{i}"], []) for i, (k, r) in enumerate(rates.items())],
                index=rates.index,
            )
            if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
                rates_dual.iloc[-self.method_param :] = rates_dual.iloc[-self.method_param - 1]
            if "avg" in self.fixing_method:
                rate = self._avg_rate_with_spread(rates_dual, dcf_vals)
            else:
                rate = self._isda_compounded_rate_with_spread(rates_dual, dcf_vals)
            notional_exposure = Series(
                [gradient(rate, [f"fixing_{i}"])[0] for i in range(len(dcf_dates.index) - 1)]
            ).astype(float)
            v = disc_curve[self.payment]
            mask = ~fixed.to_numpy()  # exclude fixings that are already fixed

            notional_exposure[mask] *= -self.notional * (self.dcf / dcf_of_r[mask]) * float(v)
            notional_exposure[mask] /= v_with_r[mask].astype(float)
            # notional_exposure[mask] *= (-self.notional * (self.dcf / dcf_of_r[mask]) * v / v_with_r[mask])
            # notional_exposure[fixed.drop_index(drop=True)] = 0.0
            notional_exposure[fixed.to_numpy()] = 0.0
            extra_cols = {
                "obs_dcf": dcf_of_r,
                "notional": notional_exposure.astype(float),  # apply(float, convert_dtype=float),
            }
        else:
            if "avg" in self.fixing_method:
                rate = self._avg_rate_with_spread(rates, dcf_vals)
            else:
                rate = self._isda_compounded_rate_with_spread(rates, dcf_vals)
            extra_cols = {}

        if rates.isna().any():
            raise ValueError(
                "RFRs could not be calculated, have you missed providing `fixings` or "
                "does the `Curve` begin after the start of a `FloatPeriod` including"
                "the `method_param` adjustment?"
            )

        return rate, DataFrame(
            {
                "obs_dates": obs_dates,
                "dcf_dates": dcf_dates,
                "dcf": dcf_vals,
                "rates": rates.astype(float).reset_index(drop=True),
                **extra_cols,
            }
        )

    def _fixings_table_fast(self, curve: Union[Curve, LineCurve], disc_curve: Curve):
        """
        Return a DataFrame of **approximate** fixing exposures.

        For arguments see :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
        """
        if "rfr" in self.fixing_method:
            # Depending upon method get the observation dates and dcf dates
            obs_dates, dcf_dates = self._get_method_dcf_markers(curve)

            # TODO (low) this calculation could be vectorised by a 360 or 365 multiplier
            dcf_vals = Series(
                [  # calculate the dcf values from the dcf dates
                    dcf(dcf_dates[i], dcf_dates[i + 1], curve.convention)
                    for i in range(len(dcf_dates.index) - 1)
                ]
            )
            obs_vals = Series(
                [  # calculate the dcf values from the dcf dates
                    dcf(obs_dates[i], obs_dates[i + 1], curve.convention)
                    for i in range(len(obs_dates.index) - 1)
                ]
            )

            # approximate DFs
            v_vals = Series(np.nan, index=obs_dates.iloc[1:])
            v_vals.iloc[0] = log(float(disc_curve[obs_dates.iloc[1]]))
            v_vals.iloc[-1] = log(float(disc_curve[obs_dates.iloc[-1]]))
            v_vals = v_vals.interpolate(method="time")
            v_vals = Series(np.exp(v_vals.to_numpy()), index=obs_vals.index)

            scalar = dcf_vals.values / obs_vals.values
            if self.fixing_method in ["rfr_lockout", "rfr_lockout_avg"]:
                scalar[-self.method_param :] = 0.0
                scalar[-(self.method_param + 1)] = (
                    obs_vals.iloc[-(self.method_param + 1) :].sum()
                    / obs_vals.iloc[-(self.method_param + 1)]
                )
            # perform an efficient rate approximation
            rate = curve.rate(
                effective=obs_dates.iloc[0],
                termination=obs_dates.iloc[-1],
            )
            r_bar, d, n = average_rate(
                obs_dates.iloc[0], obs_dates.iloc[-1], curve.convention, rate
            )
            # approximate sensitivity to each fixing
            z = self.float_spread / 10000
            if "avg" in self.fixing_method:
                drdri = 1 / n
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

            v = float(disc_curve[self.payment])
            v_vals /= v
            notional_exposure = Series(
                (-self.notional * self.dcf * float(drdri) / d * scalar) / v_vals,
                index=obs_vals.index,
            )

            table = DataFrame(
                {
                    "obs_dates": obs_dates,
                    "obs_dcf": obs_vals,
                    "dcf_dates": dcf_dates,
                    "dcf": dcf_vals,
                    "notional": notional_exposure,
                    "rates": Series(rate, index=obs_dates.index).astype(
                        float
                    ),  # .apply(float, convert_dtype=float),
                }
            )

            table = table.iloc[:-1]
            return table[["obs_dates", "notional", "dcf", "rates"]].set_index("obs_dates")
        elif "ibor" in self.fixing_method:
            fixing_date = add_tenor(self.start, f"-{self.method_param}b", "P", curve.calendar)
            return DataFrame(
                {
                    "obs_dates": [fixing_date],
                    "notional": -self.notional,
                    "dcf": [None],
                    "rates": [self.rate(curve)],
                }
            ).set_index("obs_dates")

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @property
    def _is_inefficient(self):
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
            if self.fixings is not NoInput.blank:
                return True
            elif self.float_spread == 0 or self.spread_compound_method == "none_simple":
                return False
            else:
                return True
        elif self.fixing_method == "ibor":
            return False
        # else fixing method in ["rfr_lookback", "rfr_lockout"]
        return True

    def _get_method_dcf_markers(self, curve: Union[Curve, LineCurve], endpoints=False):
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
            start_obs = add_tenor(self.start, f"-{self.method_param}b", "P", curve.calendar)
            end_obs = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            start_dcf, end_dcf = start_obs, end_obs
        elif self.fixing_method in ["rfr_lookback", "rfr_lookback_avg"]:
            start_obs = add_tenor(self.start, f"-{self.method_param}b", "P", curve.calendar)
            end_obs = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            start_dcf, end_dcf = self.start, self.end
        else:
            raise NotImplementedError(
                "`fixing_method` should be in {'rfr_payment_delay', 'rfr_lockout', "
                "'rfr_lookback', 'rfr_observation_shift'} or the same with '_avg' as "
                "a suffix for averaging methods."
            )

        if endpoints:
            # return just the edges without the computation of creating Series
            return start_obs, end_obs, start_dcf, end_dcf

        # dates of the fixing observation period
        obs_dates = Series(date_range(start=start_obs, end=end_obs, freq=curve.calendar))
        # dates for the dcf weight for each observation towards the calculation
        dcf_dates = Series(date_range(start=start_dcf, end=end_dcf, freq=curve.calendar))
        if len(dcf_dates) != len(obs_dates):
            # this might only be true with lookback when obs dates are adjusted
            # but DCF dates are not, and if starting on holiday causes problems.
            raise ValueError(
                "RFR Observation and Accrual DCF dates do not align.\n"
                "This is usually the result of a 'rfr_lookback' Period which does "
                "not adhere to the holiday calendar of the `curve`.\n"
                f"start date: {self.start.strftime('%d-%m-%Y')} is curve holiday? "
                f"{_is_holiday(self.start, curve.calendar)}\n"
                f"end date: {self.end.strftime('%d-%m-%Y')} is curve holiday? "
                f"{_is_holiday(self.end, curve.calendar)}\n"
            )

        return obs_dates, dcf_dates

    def _get_analytic_delta_quadratic_coeffs(self, fore_curve, disc_curve):
        """
        For use in the Leg._spread calculation get the 'a' and 'b' coefficients
        """
        os, oe, _, _ = self._get_method_dcf_markers(fore_curve, True)
        rate = fore_curve.rate(
            effective=os,
            termination=oe,
            float_spread=0.0,
            spread_compound_method=self.spread_compound_method,
        )
        r, d, n = average_rate(os, oe, fore_curve.convention, rate)
        # approximate sensitivity to each fixing
        z = 0.0 if self.float_spread is None else self.float_spread
        if self.spread_compound_method == "isda_compounding":
            d2rdz2 = d * (n - 1) * (1 + (r / 100 + z / 10000) * d) ** (n - 2) / 1e8
            drdz = (1 + (r / 100 + z / 10000) * d) ** (n - 1) / 1e4
            Nvd = -self.notional * disc_curve[self.payment] * self.dcf
            a, b = 0.5 * Nvd * d2rdz2, Nvd * drdz
        elif self.spread_compound_method == "isda_flat_compounding":
            # d2rdz2 = 0.0
            drdz = (1 + comb(n, 2) / n * r / 100 * d + comb(n, 3) / n * (r / 100 * d) ** 2) / 1e4
            Nvd = -self.notional * disc_curve[self.payment] * self.dcf
            a, b = 0.0, Nvd * drdz

        return a, b


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
        currency: Union[str, NoInput] = NoInput(0),
        stub_type: Union[str, NoInput] = NoInput(0),
        rate: Union[float, NoInput] = NoInput(0),
    ):
        self.notional, self.payment = notional, payment
        self.currency = defaults.base_currency if currency is NoInput.blank else currency.lower()
        self.stub_type = stub_type
        self._rate = rate if rate is NoInput.blank else float(rate)

    def rate(self):
        """
        Return the associated rate initialised with the *Cashflow*. Not used for calculations.
        """
        return self._rate

    def npv(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the *Cashflow*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        if not isinstance(disc_curve, Curve) and curve is NoInput.blank:
            raise TypeError("`curves` have not been supplied correctly.")
        value = self.cashflow * disc_curve_[self.payment]
        if local:
            return {self.currency: value}
        else:
            fx, _ = _get_fx_and_base(self.currency, fx, base)
            return fx * value

    def cashflows(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ) -> dict:
        """
        Return the cashflows of the *Cashflow*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve_ is NoInput.blank:
            npv, npv_fx, df, collateral = None, None, None, None
        else:
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)
            df, collateral = float(disc_curve_[self.payment]), disc_curve_.collateral

        try:
            cashflow_ = float(self.cashflow)
        except TypeError:  # cashflow in superclass not a property
            cashflow_ = None

        rate = None if self.rate() is NoInput.blank else self.rate()
        stub_type = None if self.stub_type is NoInput.blank else self.stub_type
        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: stub_type,
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["a_acc_start"]: None,
            defaults.headers["a_acc_end"]: None,
            defaults.headers["payment"]: self.payment,
            defaults.headers["convention"]: None,
            defaults.headers["dcf"]: None,
            defaults.headers["notional"]: float(self.notional),
            defaults.headers["df"]: df,
            defaults.headers["rate"]: rate,
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow_,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
            defaults.headers["collateral"]: collateral,
        }

    @property
    def cashflow(self):
        return -self.notional

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        """
        Return the analytic delta of the *Cashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class IndexMixin(metaclass=ABCMeta):
    """
    Abstract base class to include methods and properties related to indexed *Periods*.
    """

    index_base: Union[float, Series, NoInput] = NoInput(0)
    index_method: str = ""
    index_fixings: Union[float, Series, NoInput] = NoInput(0)
    index_lag: Union[int, NoInput] = NoInput(0)
    payment: datetime = datetime(1990, 1, 1)
    currency: str = ""

    def cashflow(self, curve: Union[IndexCurve, NoInput] = NoInput(0)) -> Optional[DualTypes]:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional,
        adjusted for the index.
        """
        if self.real_cashflow is None:
            return None
        index_ratio, _, _ = self.index_ratio(curve)
        if index_ratio is None:
            return None
        else:
            if self.index_only:
                _ = -1.0
            else:
                _ = 0.0
            _ = self.real_cashflow * (index_ratio + _)
        return _

    def index_ratio(self, curve: Union[IndexCurve, NoInput] = NoInput(0)) -> tuple:
        """
        Calculate the index ratio for the end date of the *IndexPeriod*.

        .. math::

           I(m) = \\frac{I_{val}(m)}{I_{base}}

        Parameters
        ----------
        curve : IndexCurve
            The index curve from which index values are forecast.

        Returns
        -------
        float, Dual, Dual2
        """
        denominator = self._index_value(
            i_fixings=self.index_base,
            i_date=getattr(self, "start", None),  # IndexCashflow has no start
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
        i_curve: Union[IndexCurve, NoInput],
        i_lag: int,
        i_method: str,
    ) -> Optional[DualTypes]:
        if i_curve is NoInput.blank:
            return None
        elif (not isinstance(i_curve, IndexCurve) and not isinstance(i_curve, CompositeCurve)) or (
            isinstance(i_curve, CompositeCurve) and not isinstance(i_curve.curves[0], IndexCurve)
        ):
            raise TypeError("`index_value` must be forecast from an `IndexCurve`.")
        elif i_lag != i_curve.index_lag:
            return None  # TODO decide if RolledCurve to correct index lag be attempted
        else:
            return i_curve.index_value(i_date, i_method)

    @staticmethod
    def _index_value(
        i_fixings: Union[float, Series, NoInput],
        i_date: datetime,
        i_curve: Union[IndexCurve, NoInput],
        i_lag: int,
        i_method: str,
    ) -> Union[DualTypes, NoInput]:
        """
        Project an index rate, or lookup from provided fixings, for a given date.

        If ``index_fixings`` are set on the period this will be used instead of
        the ``curve``.

        Parameters
        ----------
        curve : IndexCurve

        Returns
        -------
        float, Dual, Dual2
        """
        if i_fixings is NoInput.blank:
            return IndexMixin._index_value_from_curve(i_date, i_curve, i_lag, i_method)
        else:
            if isinstance(i_fixings, Series):
                if i_method == "daily":
                    adj_date = i_date
                    unavailable_date = i_fixings.index[-1]
                else:
                    adj_date = datetime(i_date.year, i_date.month, 1)
                    _ = i_fixings.index[-1]
                    unavailable_date = _get_eom(_.month, _.year)

                if i_date > unavailable_date:
                    if i_curve is NoInput.blank:
                        return NoInput(0)
                    else:
                        return IndexMixin._index_value_from_curve(i_date, i_curve, i_lag, i_method)
                    # raise ValueError(
                    #     "`index_fixings` cannot forecast the index value. "
                    #     f"There are no fixings available after date: {unavailable_date}"
                    # )
                else:
                    try:
                        return i_fixings[adj_date]
                    except KeyError:
                        s = i_fixings.copy()
                        s.loc[adj_date] = np.NaN  # type: ignore[call-overload]
                        _ = s.sort_index().interpolate("time")[adj_date]
                        return _
            else:
                return i_fixings

    def npv(
        self,
        curve: Union[IndexCurve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
        local: bool = False,
    ):
        """
        Return the cashflows of the *IndexPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_from_curve(curve, disc_curve)
        if not isinstance(disc_curve, Curve) and curve is NoInput.blank:
            raise TypeError("`curves` have not been supplied correctly.")
        value = self.cashflow(curve) * disc_curve_[self.payment]
        if local:
            return {self.currency: value}
        else:
            fx, _ = _get_fx_and_base(self.currency, fx, base)
            return fx * value

    @property
    @abstractmethod
    def real_cashflow(self):
        pass  # pragma: no cover


class IndexFixedPeriod(IndexMixin, FixedPeriod):  # type: ignore[misc]
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
           curve=IndexCurve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.99}, index_base=100.0, index_lag=2),
           disc_curve=Curve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.98})
       )
    """

    def __init__(
        self,
        *args,
        index_base: Union[float, Series, NoInput] = NoInput(0),
        index_fixings: Union[float, Series, NoInput] = NoInput(0),
        index_method: Union[str, NoInput] = NoInput(0),
        index_lag: Union[int, NoInput] = NoInput(0),
        **kwargs,
    ):
        # if index_base is None:
        #     raise ValueError("`index_base` cannot be None.")
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_only = False
        self.index_method = (
            defaults.index_method if index_method is NoInput.blank else index_method.lower()
        )
        self.index_lag = defaults.index_lag if index_lag is NoInput.blank else index_lag
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super(IndexMixin, self).__init__(*args, **kwargs)

    def analytic_delta(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the analytic delta of the *IndexFixedPeriod*.
        See :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        real_a_delta = super().analytic_delta(curve, disc_curve, fx, base)
        index_ratio, _, _ = self.index_ratio(curve)
        _ = None if index_ratio is None else real_a_delta * index_ratio
        return _

    @property
    def real_cashflow(self):
        """
        float, Dual or Dual2 : The calculated real value from rate, dcf and notional.
        """
        if self.fixed_rate is NoInput.blank:
            return None
        else:
            return -self.notional * self.dcf * self.fixed_rate / 100

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere

    def cashflows(
        self,
        curve: Union[IndexCurve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Return the cashflows of the *IndexFixedPeriod*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Union[Curve, NoInput] = _disc_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve_ is NoInput.blank or self.fixed_rate is NoInput.blank:
            npv = None
            npv_fx = None
        else:
            npv = float(self.npv(curve, disc_curve_))
            npv_fx = npv * float(fx)

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
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def npv(self, *args, **kwargs):
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
           curve=IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.99}, index_base=100.0),
           disc_curve=Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 31): 0.98}),
       )
    """

    def __init__(
        self,
        *args,
        index_base: float,
        index_fixings: Union[float, Series, NoInput] = NoInput(0),
        index_method: Union[str, NoInput] = NoInput(0),
        index_lag: Union[int, NoInput] = NoInput(0),
        index_only: bool = False,
        end: Union[datetime, NoInput] = NoInput(0),
        **kwargs,
    ):
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = (
            defaults.index_method if index_method is NoInput.blank else index_method.lower()
        )
        self.index_lag = defaults.index_lag if index_lag is NoInput.blank else index_lag
        self.index_only = index_only
        super(IndexMixin, self).__init__(*args, **kwargs)
        self.end = self.payment if end is NoInput.blank else end

    @property
    def real_cashflow(self):
        return -self.notional

    def cashflows(
        self,
        curve: Union[Curve, NoInput] = NoInput(0),
        disc_curve: Union[Curve, NoInput] = NoInput(0),
        fx: Union[float, FXRates, FXForwards, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ) -> dict:
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

    def npv(self, *args, **kwargs):
        """
        Return the NPV of the *IndexCashflow*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        return super().npv(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs):
        """
        Return the analytic delta of the *IndexCashflow*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0


def _float_or_none(val):
    if val is None:
        return None
    else:
        return float(val)


def _disc_from_curve(curve: Curve, disc_curve: Union[Curve, NoInput]) -> Curve:
    if disc_curve is NoInput.blank:
        _: Curve = curve
    else:
        _ = disc_curve
    return _


def _disc_maybe_from_curve(
    curve: Union[Curve, NoInput, dict], disc_curve: Union[Curve, NoInput]
) -> Union[Curve, NoInput]:
    if disc_curve is NoInput.blank:
        if isinstance(curve, dict):
            raise ValueError("`disc_curve` cannot be inferred from a dictionary of curves.")
        _: Curve = curve
    else:
        _ = disc_curve
    return _
