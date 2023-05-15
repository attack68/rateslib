# This module is a dependent of legs.py

# TODO float spread averaging

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

import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas import DataFrame, date_range, Series, NA, isna

from rateslib import defaults
from rateslib.calendars import add_tenor, get_calendar, dcf
from rateslib.curves import Curve, LineCurve, IndexCurve
from rateslib.dual import Dual, Dual2, DualTypes
from rateslib.fx import FXForwards, FXRates


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _get_fx_and_base(
    currency: str,
    fx: Optional[Union[float, FXRates, FXForwards]] = None,
    base: Optional[str] = None,
):
    if isinstance(fx, (FXRates, FXForwards)):
        base = fx.base if base is None else base.lower()
        if base == currency:
            fx = 1.0
        else:
            fx = fx.rate(pair=f"{currency}{base}")
    elif fx is None:
        fx = 1.0
    return fx, base


class BasePeriod(metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``Period`` subclasses.

    Parameters
    ----------
    start : Datetime
        The adjusted start date of the calculation period.
    end: Datetime
        The adjusted end date of the calculation period.
    payment : Datetime
        The adjusted payment date of the period.
    frequency : str, optional
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

    See Also
    --------
    FixedPeriod : Create a period defined with a fixed rate.
    FloatPeriod : Create a period defined with a floating rate index.
    Cashflow : Create a period defined by a single cashflow.
    """

    @abstractmethod
    def __init__(
        self,
        start: datetime,
        end: datetime,
        payment: datetime,
        frequency: str,
        notional: Optional[float] = None,
        currency: Optional[str] = None,
        convention: Optional[str] = None,
        termination: Optional[datetime] = None,
        stub: bool = False,
    ):
        if end < start:
            raise ValueError("`end` cannot be before `start`.")
        self.start, self.end, self.payment = start, end, payment
        self.frequency = frequency.upper()
        self.notional = defaults.notional if notional is None else notional
        self.currency = defaults.base_currency if currency is None else currency.lower()
        self.convention = defaults.convention if convention is None else convention
        self.termination = termination
        self.freq_months = defaults.frequency_months[self.frequency]
        self.stub = stub

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
        )

    @abstractmethod
    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
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

        Notes
        -----
        For a :class:`FixedPeriod` this gives the sensitivity to the fixed rate.

        .. math::

           C = v N d R, \\quad A = \\frac{\\partial C}{\\partial R} = v N d

        For a :class:`FloatPeriod` this gives the sensitivity to the float spread, which
        under a ``spread_compound_method`` of *"none_simple"* (or if the
        ``float_spread`` is zero) is equivalent to :class:`FixedPeriod` analytic delta.
        If other compounding methods are applied the figure is usually slightly higher.

        .. math::

           C = v N d r(r_i, z), \\quad A = \\frac{\\partial C}{\\partial z} = v N d \\frac{\\partial r}{\\partial z}

        where,

        .. math::

           d =& \\text{DCF of period} \\\\
           v =& \\text{DF of period payment date}\\\\
           N =& \\text{Notional of period}\\\\
           R =& \\text{Fixed rate of period}\\\\
           r =& \\text{Float period rate as a function of fixings and spread}\\\\
           z =& \\text{Float period spread}\\\\

        The sign, or direction, of analytic delta is ignorant of whether a period is
        fixed rate or floating rate, and thus, dependent upon the context of calculating
        either fixed rates or floating spreads, requires different interpretation.

        For a **positive notional**, which assumes paying a cashflow, this method
        returns a **positive value**. The resultant usage of analytic delta should then
        assign a proper sign for its context in post-processing.

        The analytic delta of a :class:`Cashflow` is set to zero to be compatible with
        inherited :class:`~rateslib.instruments.BaseDerivative` methods.

        Examples
        --------
        .. ipython:: python

           curve = Curve({dt(2021,1,1): 1.00, dt(2025,1,1): 0.83}, "log_linear", id="SONIA")
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
        disc_curve_: Curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)
        _ = fx * self.notional * self.dcf * disc_curve_[self.payment] / 10000
        return _

    @abstractmethod
    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
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
        disc_curve = disc_curve or curve
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
            defaults.headers["df"]: None
            if disc_curve is None
            else float(disc_curve[self.payment]),
        }

    @abstractmethod
    def npv(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
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
    """

    def __init__(self, *args, fixed_rate: Union[float, None] = None, **kwargs):
        self.fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs):
        return super().analytic_delta(*args, **kwargs)

    @property
    def cashflow(self):
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional.
        """
        return (
            None
            if self.fixed_rate is None
            else -self.notional * self.dcf * self.fixed_rate / 100
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def npv(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        disc_curve = disc_curve or curve
        if disc_curve is None:
            raise TypeError(
                "`curves` have not been supplied correctly. NoneType has been detected."
            )
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = self.cashflow * disc_curve[self.payment]
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve is None or self.fixed_rate is None:
            npv = None
            npv_fx = None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)

        cashflow = None if self.cashflow is None else float(self.cashflow)
        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }


def _validate_float_args(
    fixing_method: Optional[str],
    method_param: Optional[int],
    spread_compound_method: Optional[str],
):
    """
    Validate the argument input to float periods.

    Returns
    -------
    tuple
    """
    fixing_method_: str = (
        defaults.fixing_method if fixing_method is None else fixing_method.lower()
    )
    if fixing_method_ not in [
        "ibor",
        "rfr_payment_delay",
        "rfr_observation_shift",
        "rfr_lockout",
        "rfr_lookback",
    ]:
        raise ValueError(
            "`fixing_method` must be in {'rfr_payment_delay', "
            "'rfr_observation_shift', 'rfr_lockout', 'rfr_lookback', 'ibor'}, "
            f"got '{fixing_method_}'."
        )

    method_param_ = (
        defaults.fixing_method_param[fixing_method_]
        if method_param is None
        else method_param
    )
    if method_param_ != 0 and fixing_method_ == "rfr_payment_delay":
        raise ValueError(
            "`method_param` should not be used (or a value other than 0) when "
            f"using a `fixing_method` of 'rfr_payment_delay', got {method_param_}. "
            f"Configure the `payment_lag` option instead to have the "
            f"appropriate effect."
        )
    elif fixing_method_ == "rfr_lockout" and method_param_ < 1:
        raise ValueError(
            f'`method_param` must be >0 for "rfr_lockout" `fixing_method`, '
            f"got {method_param_}"
        )

    if spread_compound_method is None:
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
        ``curve``.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----

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
    :download:`Swap Compounding Formulae<_static/SSRN-id3882163(compounding).pdf>` and
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
        float_spread: Optional[float] = None,
        fixings: Optional[Union[float, list, Series]] = None,
        fixing_method: Optional[str] = None,
        method_param: Optional[int] = None,
        spread_compound_method: Optional[str] = None,
        **kwargs,
    ):
        self.float_spread = 0 if float_spread is None else float_spread

        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        self.fixings = fixings
        if isinstance(self.fixings, list) and self.fixing_method == "ibor":
            raise ValueError(
                "`fixings` can only be a single value, not list, under 'ibor' method."
            )

        # self.calendar = get_calendar(calendar)
        # self.modifier = modifier
        super().__init__(*args, **kwargs)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        if self.spread_compound_method == "none_simple" or self.float_spread == 0:
            # then analytic_delta is not impacted by float_spread compounding
            dr_dz = 1.0
        elif isinstance(curve, Curve):
            _ = self.float_spread
            DualType = Dual if curve.ad in [0, 1] else Dual2
            self.float_spread = DualType(float(_), "z_float_spread")
            rate = self.rate(curve)
            dr_dz = rate.gradient("z_float_spread")[0] * 100
            self.float_spread = _
        else:
            raise TypeError(
                "`curve` must be supplied for given `spread_compound_method`"
            )

        return dr_dz * super().analytic_delta(curve, disc_curve, fx, base)

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)

        rate = None if curve is None else float(self.rate(curve))
        if disc_curve is None or rate is None:
            npv, npv_fx = None, None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)
        cashflow = (
            None if rate is None else -float(self.notional * self.dcf * rate / 100)
        )

        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["rate"]: rate,
            defaults.headers["spread"]: float(self.float_spread),
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def npv(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        disc_curve = disc_curve or curve
        if disc_curve is None or curve is None:
            raise TypeError(
                "`curves` have not been supplied correctly. NoneType has been detected."
            )
        if self.payment < disc_curve.node_dates[0]:
            return 0.0  # payment date is in the past avoid issues with fixings or rates
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = (
            self.rate(curve)
            / 100
            * self.dcf
            * disc_curve[self.payment]
            * -self.notional
        )
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def rate(self, curve: Union[Curve, LineCurve]):
        """
        Calculating the floating rate for the period.

        Parameters
        ----------
        curve : Curve, LineCurve
            The forecasting curve object.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        .. ipython:: python

           period.rate(curve)
        """
        if isinstance(self.fixings, (float, Dual, Dual2)):
            # if fixings is a single value then return that value (curve unused)
            if (
                self.spread_compound_method
                in ["isda_compounding", "isda_flat_compounding"]
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

        if type(curve) is Curve:
            if "rfr" in self.fixing_method:
                return self._rfr_rate_from_df_curve(curve)
            elif "ibor" in self.fixing_method:
                return self._ibor_rate_from_df_curve(curve)
        elif type(curve) is LineCurve:
            if "rfr" in self.fixing_method:
                return self._rfr_rate_from_line_curve(curve)
            elif "ibor" in self.fixing_method:
                return self._ibor_rate_from_line_curve(curve)
        else:
            raise TypeError("`curve` must be of type `Curve` or `LineCurve`.")

    def _rfr_rate_from_df_curve(self, curve: Curve):
        if self.fixing_method == "rfr_payment_delay" and not self._is_complex:
            return curve.rate(self.start, self.end) + self.float_spread / 100

        elif self.fixing_method == "rfr_observation_shift" and not self._is_complex:
            start = add_tenor(self.start, f"-{self.method_param}b", "P", curve.calendar)
            end = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            return curve.rate(start, end) + self.float_spread / 100

            # TODO: semi-efficient method for lockout under certain conditions
        else:
            # return complex calculation
            return self._rfr_fixings_array(curve, fixing_exposure=False)[0]

    def _ibor_rate_from_df_curve(self, curve: Curve):
        # the compounding method has no effect on single rate (ibor) fixings.
        if isinstance(self.fixings, Series):
            # check if we return published IBOR rate
            fixing_date = add_tenor(
                self.start, f"-{self.method_param}B", None, curve.calendar
            )
            try:
                return self.fixings[fixing_date] + self.float_spread / 100
            except KeyError:
                # fixing not available: use curve
                pass

        if self.stub:
            r = curve.rate(self.start, self.end) + self.float_spread / 100
        else:
            r = curve.rate(self.start, f"{self.freq_months}m") + self.float_spread / 100
        return r

    def _ibor_rate_from_line_curve(self, curve: LineCurve):
        # the compounding method has no effect on single rate (ibor) fixings.
        fixing_date = add_tenor(
            self.start, f"-{self.method_param}B", None, curve.calendar
        )
        if isinstance(self.fixings, Series):
            try:
                return self.fixings[fixing_date] + self.float_spread / 100
            except KeyError:
                # fixing not available: use curve
                pass
        return curve[fixing_date] + self.float_spread / 100

    def _rfr_rate_from_line_curve(self, curve: LineCurve):
        return self._rfr_fixings_array(curve, fixing_exposure=False)[0]

    def _rfr_fixings_array(
        self,
        curve: Union[Curve, LineCurve],
        fixing_exposure: bool = False,
    ):
        """
        Calculate the rate of a period via extraction and combination of every fixing.

        This method of calculation is used when either:

        - known fixings needs to be combined with unknown fixings,
        - the fixing_method is of a type that needs individual fixing data,
        - the spread compound method is of a type that needs individual fixing data.

        Parameters
        ----------
        curve : Curve or LineCurve
            The forecasting curve used to extract the fixing data.
        fixing_exposure : bool
            Whether to calculate sensitivities to the fixings additionally.

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
        """
        if self.fixing_method in ["rfr_payment_delay", "rfr_lockout"]:
            start_obs, end_obs = self.start, self.end
            start_dcf, end_dcf = self.start, self.end
        elif self.fixing_method == "rfr_observation_shift":
            start_obs = add_tenor(
                self.start, f"-{self.method_param}b", "P", curve.calendar
            )
            end_obs = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            start_dcf, end_dcf = start_obs, end_obs
        elif self.fixing_method == "rfr_lookback":
            start_obs = add_tenor(
                self.start, f"-{self.method_param}b", "P", curve.calendar
            )
            end_obs = add_tenor(self.end, f"-{self.method_param}b", "P", curve.calendar)
            start_dcf, end_dcf = self.start, self.end
        else:
            raise NotImplementedError(
                "`fixing_method` should be in {'rfr_payment_delay', 'rfr_lockout', "
                "'rfr_lookback', 'rfr_observation_shift'}"
            )

        obs_dates = Series(
            date_range(  # dates of the fixing observation period
                start=start_obs, end=end_obs, freq=curve.calendar
            )
        )
        dcf_dates = Series(
            date_range(  # dates for the dcf weights period
                start=start_dcf, end=end_dcf, freq=curve.calendar
            )
        )
        if len(dcf_dates) != len(obs_dates):
            # this should never be true since dates should be adjusted under the
            # curve calendar which is consistent in all above execution statements.
            raise ValueError(  # pragma: no cover
                "Observation and dcf dates do not align, possible `calendar` issue."
            )

        dcf_vals = Series(
            [  # calculate the dcf values from the dcf dates
                dcf(dcf_dates[i], dcf_dates[i + 1], curve.convention)
                for i in range(len(dcf_dates.index) - 1)
            ]
        )

        rates = Series(NA, index=obs_dates[:-1])
        if self.fixings is not None:
            # then fixings will be a list or Series, scalars are already processed.
            if isinstance(self.fixings, list):
                rates.iloc[: len(self.fixings)] = self.fixings
            elif isinstance(self.fixings, Series):
                if not self.fixings.index.is_monotonic_increasing:
                    raise ValueError(
                        "`fixings` as a Series must have a monotonically increasing "
                        "datetimeindex."
                    )
                fixing_rates = self.fixings.loc[obs_dates.iloc[0] : obs_dates.iloc[-2]]  # type: ignore[misc]
                rates.loc[fixing_rates.index] = fixing_rates

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
                raise TypeError(  # pragma: no cover
                    "`fixings` should be of type scalar, None, list or Series."
                )

        # reindex the rates series getting missing values from the curves
        rates = Series(
            {k: curve.rate(k, "1b") if isna(v) else v for k, v in rates.items()}
        )

        if fixing_exposure:
            # need to calculate the dcfs associated with the rates (unshifted)
            if self.fixing_method in [
                "rfr_payment_delay",
                "rfr_observation_shift",
                "rfr_lockout",
            ]:  # for all these methods there is no shift
                dcf_of_r = dcf_vals.copy()
            elif self.fixing_method == "rfr_lookback":
                dcf_of_r = Series(
                    [
                        dcf(obs_dates[i], obs_dates[i + 1], curve.convention)
                        for i in range(len(dcf_dates.index) - 1)
                    ]
                )

        if self.fixing_method == "rfr_lockout":
            # adjust the final rates values of the lockout arrays according to param
            try:
                rates.iloc[-self.method_param :] = rates.iloc[-self.method_param - 1]
            except IndexError:
                raise ValueError(
                    "period has too few dates for `rfr_lockout` param to function."
                )

        if fixing_exposure:
            rates_dual = Series(
                [
                    Dual(float(r), f"fixing_{i}")
                    for i, (k, r) in enumerate(rates.items())
                ],
                index=rates.index,
            )
            if self.fixing_method == "rfr_lockout":
                rates_dual.iloc[-self.method_param :] = rates_dual.iloc[
                    -self.method_param - 1
                ]
            rate = self._isda_compounded_rate_with_spread(rates_dual, dcf_vals)
            notional_exposure = Series(
                [
                    rate.gradient(f"fixing_{i}")[0]
                    for i in range(len(dcf_dates.index) - 1)
                ]
            )
            notional_exposure *= -self.notional * self.dcf / dcf_of_r
            extra_cols = {
                "obs_dcf": dcf_of_r,
                "notional": notional_exposure.apply(float, convert_dtype=float),
            }
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
                "rates": rates.apply(float, convert_dtype=float).reset_index(drop=True),
                **extra_cols,
            }
        )

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
            for i in range(1, len(sub_cashflows)):
                k, k_ = rates.index[i], rates.index[i - 1]
                sub_cashflows[k] += sub_cashflows[k_] * rates[k] / 100 * dcf_vals[k]
            total_cashflow = sub_cashflows.sum()
            return total_cashflow * 100 / dcf_vals.sum()
        else:
            # this path not generally hit due to validation at initialisation
            raise ValueError(
                "`spread_compound_method` must be in {'none_simple', "
                "'isda_compounding', 'isda_flat_compounding'}."
            )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @property
    def _is_complex(self):
        """
        A complex float leg is one which is RFR based and for which each individual
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
            if self.fixings is not None:
                return True
            elif self.float_spread == 0 or self.spread_compound_method == "none_simple":
                return False
            else:
                return True
        elif self.fixing_method == "ibor":
            return False
        return True

    def fixings_table(self, curve: Union[Curve, LineCurve]):
        """
        Return a DataFrame of fixing exposures.

        Parameters
        ----------
        curve : Curve, LineCurve
            The forecast needed to calculate rates which affect compounding and
            dependent notional exposure.

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
        if "rfr" in self.fixing_method:
            rate, table = self._rfr_fixings_array(curve, fixing_exposure=True)
            table = table.iloc[:-1]
            return table[["obs_dates", "notional", "dcf", "rates"]].set_index(
                "obs_dates"
            )
        elif "ibor" in self.fixing_method:
            fixing_date = add_tenor(
                self.start, f"-{self.method_param}b", "P", curve.calendar
            )
            return DataFrame(
                {
                    "obs_dates": [fixing_date],
                    "notional": -self.notional,
                    "dcf": [None],
                    "rates": [self.rate(curve)],
                }
            ).set_index("obs_dates")


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
    """

    def __init__(
        self,
        notional: float,
        payment: datetime,
        currency: Optional[str] = None,
        stub_type: Optional[str] = None,
        rate: Optional[float] = None,
    ):
        self.notional, self.payment = notional, payment
        self.currency = defaults.base_currency if currency is None else currency.lower()
        self.stub_type = stub_type
        self.rate_ = rate if rate is None else float(rate)

    def rate(self):
        """
        The rate of the cashflow (nominal only - not used in calculations)
        """
        return self.rate_

    def npv(
        self,
        curve: Curve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        disc_curve = disc_curve or curve
        if disc_curve is None:
            raise TypeError(
                "`curves` have not been supplied correctly. NoneType has been detected."
            )
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = self.cashflow * disc_curve[self.payment]
        if local:
            return {self.currency: value}
        else:
            return fx * value

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ) -> dict:
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve is None:
            npv, npv_fx = None, None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)

        try:
            cashflow_ = float(self.cashflow)
        except TypeError:  # cashflow in superclass not a property
            cashflow_ = None

        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: self.stub_type,
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["a_acc_start"]: None,
            defaults.headers["a_acc_end"]: None,
            defaults.headers["payment"]: self.payment,
            defaults.headers["convention"]: None,
            defaults.headers["dcf"]: None,
            defaults.headers["notional"]: float(self.notional),
            defaults.headers["df"]: None
            if disc_curve is None
            else float(disc_curve[self.payment]),
            defaults.headers["rate"]: self.rate(),
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow_,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
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
        return 0


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class IndexMixin(metaclass=ABCMeta):
    index_base: float = 0.0
    index_method: str = ""
    index_fixings: Optional[Union[float, Series]] = None
    payment: datetime = datetime(1990, 1, 1)
    currency: str = ""

    def cashflow(self, curve: Optional[IndexCurve] = None) -> Optional[DualTypes]:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional,
        adjusted for the index.
        """
        if self.real_cashflow is None:
            return None
        if self.rate(curve) is None:
            return None
        _ = self.real_cashflow * self.rate(curve) / self.index_base
        return _

    def rate(self, curve: Optional[IndexCurve] = None) -> Optional[DualTypes]:
        """
        Project an index rate for the cashflow payment date.

        Parameters
        ----------
        curve : IndexCurve

        Returns
        -------
        float, Dual, Dual2
        """
        if self.index_fixings is None:
            if curve is None:
                return None
            # forecast inflation index from curve
            return curve.index_value(self.payment, self.index_method)
        else:
            if isinstance(self.index_fixings, Series):
                if self.index_method == "daily":
                    adj_date = self.payment
                else:  # index_method == "monthly"
                    adj_date = datetime(self.payment.year, self.payment.month, 1)

                try:
                    return self.index_fixings[adj_date]
                except KeyError:
                    s = self.index_fixings.copy()
                    s.loc[adj_date] = np.NaN  # type: ignore[call-overload]
                    return s.sort_index().interpolate("linear")[adj_date]
            else:
                return self.index_fixings

    def npv(
        self,
        curve: IndexCurve,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
        local: bool = False,
    ):
        disc_curve = disc_curve or curve
        if disc_curve is None:
            raise TypeError(
                "`curves` have not been supplied correctly. NoneType has been detected."
            )
        fx, base = _get_fx_and_base(self.currency, fx, base)
        value = self.cashflow(curve) * disc_curve[self.payment]
        if local:
            return {self.currency: value}
        else:
            return fx * value

    @property
    @abstractmethod
    def real_cashflow(self):
        pass  # pragma: no cover


class IndexFixedPeriod(IndexMixin, FixedPeriod):  # type: ignore[misc]
    """
    Create a period defined with a real rate adjusted by an index.

    When used with inflation products this defines a real coupon period with a
    cashflow adjusted upwards by the inflation index.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`FixedPeriod`.
    index_base : float or None, optional
        The base index to determine the cashflow.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the whole
        period. If a datetime indexed ``Series`` will use the
        fixings that are available in that object, and derive the rest from the
        ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    kwargs : dict
        Required keyword arguments to :class:`FixedPeriod`.
    """

    def __init__(
        self,
        *args,
        index_base: float,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: str = "daily",
        **kwargs,
    ):
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = index_method.lower()
        if self.index_method not in ["daily", "monthly"]:
            raise ValueError("`index_method` must be in {'daily', 'monthly'}.")
        super(IndexMixin, self).__init__(*args, **kwargs)

    def analytic_delta(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Union[float, FXRates, FXForwards] = 1.0,
        base: Optional[str] = None,
    ):
        real_a_delta = super().analytic_delta(curve, disc_curve, fx, base)
        _ = real_a_delta * self.rate(curve) / self.index_base
        return _

    @property
    def real_cashflow(self):
        """
        float, Dual or Dual2 : The calculated real value from rate, dcf and notional.
        """
        return (
            None
            if self.fixed_rate is None
            else -self.notional * self.dcf * self.fixed_rate / 100
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere

    def cashflows(
        self,
        curve: Optional[IndexCurve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ):
        disc_curve = disc_curve or curve
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if disc_curve is None or self.fixed_rate is None:
            npv = None
            npv_fx = None
        else:
            npv = float(self.npv(curve, disc_curve))
            npv_fx = npv * float(fx)

        cashflow_ = self.cashflow(curve)
        cashflow_ = None if cashflow_ is None else float(cashflow_)

        index_ = self.rate(curve)
        if index_ is None:
            index_ratio_ = None
        else:
            index_ratio_ = index_ /self.index_base

        return {
            **super(FixedPeriod, self).cashflows(curve, disc_curve, fx, base),
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["real_cashflow"]: self.real_cashflow,
            defaults.headers["index_value"]: index_,
            defaults.headers["index_ratio"]: index_ratio_,
            defaults.headers["cashflow"]: cashflow_,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }


class IndexCashflow(IndexMixin, Cashflow):  # type: ignore[misc]
    """
    Create a cashflow defined with a real rate adjusted by an index.

    When used with inflation products this defines a real redemption with a
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
        monthly data taken from the first day of month.
    kwargs : dict
        Required keyword arguments to :class:`Cashflow`.
    """
    def __init__(
        self,
        *args,
        index_base: float,
        index_fixings: Optional[Union[float, Series]] = None,
        index_method: str = "daily",
        **kwargs,
    ):
        self.index_base = index_base
        self.index_fixings = index_fixings
        self.index_method = index_method.lower()
        super(IndexMixin, self).__init__(*args, **kwargs)

    @property
    def real_cashflow(self):
        return -self.notional

    def cashflows(
        self,
        curve: Optional[Curve] = None,
        disc_curve: Optional[Curve] = None,
        fx: Optional[Union[float, FXRates, FXForwards]] = None,
        base: Optional[str] = None,
    ) -> dict:

        index_ = self.rate(curve)
        if index_ is None:
            index_ratio_ = None
        else:
            index_ratio_ = index_ / self.index_base

        return {
            **super(IndexMixin, self).cashflows(curve, disc_curve, fx, base),
            defaults.headers["real_cashflow"]: self.real_cashflow,
            defaults.headers["index_value"]: index_,
            defaults.headers["index_ratio"]: index_ratio_,
            defaults.headers["cashflow"]: self.cashflow(curve),
        }
