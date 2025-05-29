from __future__ import annotations

import warnings
from math import comb, log
from typing import TYPE_CHECKING

import numpy as np
from pandas import NA, DataFrame, Index, MultiIndex, Series, concat, isna, notna

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf, get_calendar
from rateslib.curves import Curve, average_rate, index_left
from rateslib.curves._parsers import _disc_maybe_from_curve, _disc_required_maybe_from_curve
from rateslib.curves.utils import _CurveType
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.periods.base import BasePeriod
from rateslib.periods.utils import (
    _float_or_none,
    _get_fx_and_base,
    _get_ibor_curve_from_dict,
    _get_rfr_curve_from_dict,
    _maybe_get_rfr_curve_from_dict,
    _maybe_local,
    _trim_df_by_index,
    _validate_float_args,
    _validate_fx_as_forwards,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        CalTypes,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Number,
        datetime,
        datetime_,
        str_,
    )


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

    def __init__(self, *args: Any, fixed_rate: DualTypes_ = NoInput(0), **kwargs: Any) -> None:
        self.fixed_rate: DualTypes | NoInput = fixed_rate
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
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
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
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
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
        fixings: DualTypes | list[DualTypes] | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
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
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
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
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
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
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> dict[str, DualTypes] | DualTypes:
        """
        Return the NPV of the *FloatPeriod*.
        See
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        if self.payment < disc_curve_.nodes.initial:
            if local:
                return {self.currency: 0.0}
            else:
                return 0.0  # payment date is in the past avoid issues with fixings or rates
        value = self.rate(curve) / 100 * self.dcf * disc_curve_[self.payment] * -self.notional

        return _maybe_local(value, local, self.currency, fx, base)

    def cashflow(self, curve: CurveOption_ = NoInput(0)) -> DualTypes | None:
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

    def _maybe_get_cal_and_conv_from_curve(self, curve: CurveOption_) -> tuple[CalTypes, str]:
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
                cal_ = list(curve.values())[0].meta.calendar
                conv_ = list(curve.values())[0].meta.convention
            else:
                cal_ = curve.meta.calendar
                conv_ = curve.meta.convention
        return cal_, conv_

    def rate(self, curve: CurveOption_ = NoInput(0)) -> DualTypes:
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
                fixing: DualTypes = self.fixings[fixing_date] + self.float_spread / 100
                return fixing
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
            _CurveType.dfs: self._rate_ibor_from_df_curve,
            _CurveType.values: self._rate_ibor_from_line_curve,
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
        fixing_date = curve.meta.calendar.lag(self.start, -self.method_param, False)
        return curve[fixing_date] + self.float_spread / 100

    def _rate_ibor_interpolated_ibor_from_dict(self, curve: dict[str, Curve]) -> DualTypes:
        """
        Get the rate on all available curves in dict and then determine the ones to interpolate.
        """
        calendar = next(
            iter(curve.values())
        ).meta.calendar  # note: ASSUMES all curve calendars are same
        fixing_date = add_tenor(self.start, f"-{self.method_param}B", "NONE", calendar)

        def _rate(c: Curve, tenor: str) -> DualTypes:
            if c._base_type == _CurveType.dfs:
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
            if curve._base_type == _CurveType.dfs:
                return self._rate_rfr_from_df_curve(curve)
            else:  # curve._base_type == _CurveType.values:
                return self._rate_rfr_from_line_curve(curve)

    def _rate_rfr_from_df_curve(self, curve: Curve) -> DualTypes:
        if isinstance(curve, NoInput):
            # then attempt to get rate from fixings
            return self._rfr_rate_from_individual_fixings(curve)
        elif self.start < curve.nodes.initial:
            # then likely fixing are required and curve does not have available data
            return self._rfr_rate_from_individual_fixings(curve)
        elif curve.nodes.n == 0:
            # TODO zero len curve is generated by pseudo curve in FloatRateNote.
            # This is bad construct
            return self._rfr_rate_from_individual_fixings(curve)
        elif self.fixing_method == "rfr_payment_delay" and not self._is_inefficient:
            return curve._rate_with_raise(self.start, self.end) + self.float_spread / 100
        elif self.fixing_method == "rfr_observation_shift" and not self._is_inefficient:
            start = curve.meta.calendar.lag(self.start, -self.method_param, settlement=False)
            end = curve.meta.calendar.lag(self.end, -self.method_param, settlement=False)
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
                "`spread_compound` method must be 'none_simple' in an RFR averaging period.",
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
                        "`fixings` as a Series must have a monotonically increasing datetimeindex.",
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
            #     else (curve.rate(k, "1b", "F") if k >= curve.nodes.initial else None)
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
        curve: CurveOption_,
        approximate: bool = False,
        disc_curve: CurveOption_ = NoInput(0),
        right: datetime_ = NoInput(0),
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
        disc_curve_ = _disc_required_maybe_from_curve(curve, disc_curve)

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
                    or not _d["obs_dates"].iloc[-1] <= curve_.nodes.initial
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
        self, curve: CurveOption_, disc_curve: Curve, right: NoInput | datetime
    ) -> DataFrame:
        """
        Return a DataFrame of **approximate** fixing exposures.

        For arguments see :meth:`~rateslib.periods.FloatPeriod.fixings_table`.
        """
        if "rfr" in self.fixing_method:
            curve_: Curve = _maybe_get_rfr_curve_from_dict(curve)  # type: ignore[assignment]
            # Depending upon method get the observation dates and dcf dates
            obs_dates, dcf_dates, dcf_vals, obs_vals = self._get_method_dcf_markers(
                curve_.meta.calendar, curve_.meta.convention, True
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
                obs_dates.iloc[0], obs_dates.iloc[-1], curve_.meta.convention, rate, dcf_vals.sum()
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
                    ((1 / n) * (comb(int(n), 1) + comb(int(n), 2) * dr + comb(int(n), 3) * dr**2))
                    + ((r_bar / 100 + z) / n) * (comb(int(n), 2) * d + 2 * comb(int(n), 3) * dr * d)
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
        curve: CurveOption_,
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
        elif isinstance(curve, NoInput):
            raise ValueError("`curve` must be supplied as Curve or dict for `ibor_fiixngs_table`.")
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
        calendar = curve.meta.calendar
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
            reg_end_dt = add_tenor(self.start, tenor, curve.meta.modifier, calendar)
            reg_dcf = dcf(self.start, reg_end_dt, curve.meta.convention, reg_end_dt)

            if not isinstance(self.fixings, NoInput) or fixing_dt < curve.nodes.initial:
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
        calendar = next(
            iter(curve.values())
        ).meta.calendar  # note: ASSUMES all curve calendars are same
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
        os, oe, _, _ = self._get_method_dcf_endpoints(fore_curve.meta.calendar)
        rate = fore_curve._rate_with_raise(
            effective=os,
            termination=oe,
            float_spread=0.0,
            spread_compound_method=self.spread_compound_method,
        )
        r, d, n = average_rate(os, oe, fore_curve.meta.convention, rate, self.dcf)
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
            drdz = (
                1 + comb(int(n), 2) / n * r / 100 * d + comb(int(n), 3) / n * (r / 100 * d) ** 2
            ) / 1e4
            Nvd = -self.notional * disc_curve[self.payment] * self.dcf
            a, b = 0.0, Nvd * drdz

        return a, b


class NonDeliverableFixedPeriod(FixedPeriod):
    """
    Create a *FixedPeriod* whose non-deliverable cashflow is converted to a ``settlement_currency``.

    Parameters
    ----------
    args:
        Positional arguments to :class:`~rateslib.periods.FixedPeriod`.
    settlement_currency:
        The currency used for settlement of the non-deliverable cashflow.
    fx_fixing: float, Dual, Dual2, optional
        The FX fixing to determine the settlement amount.
        The ``currency`` should be the left hand side, and ``settlement_currency`` as RHS,
        e.g. BRLUSD, unless ``reversed``, in which case it should be, e.g. USDBRL.
    fx_fixing_date: datetime
        Date on which the FX fixing for settlement is determined.
    reversed: bool, optional
        If *True* reverses the FX rate for settlement fixing, as shown above.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = -NdRf

    where *f* is the FX rate at settlement to convert ``currency`` into ``settlement_currency``.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = Cv = -NdRfv(m)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial R} = Ndfv(m)

    Examples
    --------
    .. ipython:: python

       fp = NonDeliverableFixedPeriod(
           start=dt(2022, 2, 1),
           end=dt(2022, 8, 1),
           payment=dt(2022, 8, 2),
           frequency="S",
           notional=1e6,
           currency="brl",
           convention="30e360",
           fixed_rate=5.0,
           settlement_currency="usd",
           fx_fixing=5.0,
           fx_fixing_date=dt(2022, 7, 30),
           reversed=True,
       )
       fp.cashflows(curve=Curve({dt(2022, 1, 1):1.0, dt(2022, 12, 31): 0.98}))
    """

    def __init__(
        self,
        *args: Any,
        settlement_currency: str,
        fx_fixing: DualTypes_ = NoInput(0),
        fx_fixing_date: datetime_ = NoInput(0),
        reversed: bool = False,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        self.settlement_currency = settlement_currency.lower()
        self.fx_fixing = fx_fixing
        self.fx_fixing_date = fx_fixing_date
        self.reversed = reversed
        super().__init__(*args, **kwargs)
        if self.reversed:
            self.pair = f"{self.settlement_currency}{self.currency}"
        else:
            self.pair = f"{self.currency}{self.settlement_currency}"

    def _get_fx_fixing(self, fx: FX_) -> DualTypes:
        if isinstance(self.fx_fixing, NoInput):
            fx_ = _validate_fx_as_forwards(fx)
            fx_fixing: DualTypes = fx_.rate(self.pair, self.payment)
        else:
            fx_fixing = self.fx_fixing
        return fx_fixing

    def cashflow(self, fx: FX_) -> DualTypes | None:  # type: ignore[override]
        """
        Determine the cashflow amount, expressed in the ``settlement_currency``.

        Parameters
        ----------
        fx: FXForwards, optional
            Required to forecast the FX rate at settlement, if an ``fx_fixing`` is not known.

        Returns
        -------
        float, Dual, Dual2
        """
        if isinstance(self.fixed_rate, NoInput):
            return None
        else:
            fx_fixing: DualTypes = self._get_fx_fixing(fx)

            d_value: DualTypes = -self.notional * self.dcf * self.fixed_rate / 100
            if self.reversed:
                d_value /= fx_fixing
            else:
                d_value *= fx_fixing
            return d_value

    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *NonDeliverableFixedPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`

        Value is expressed in units of ``settlement_currency`` unless ``base`` is directly
        specified.
        """
        reference_ccy_value = super().analytic_delta(
            curve=curve, disc_curve=disc_curve, fx=NoInput(0), base=self.currency
        )
        fx_fixing: DualTypes = self._get_fx_fixing(fx)
        if self.reversed:
            settlement_ccy_value = reference_ccy_value / fx_fixing
        else:
            settlement_ccy_value = reference_ccy_value * fx_fixing
        fx_, _ = _get_fx_and_base(self.settlement_currency, fx, base)
        return settlement_ccy_value * fx_

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> dict[str, DualTypes] | DualTypes:
        """
        Return the NPV of the *NonDeliverableFixedPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`

        Value is expressed in units of ``settlement_currency`` unless ``base`` is directly
        specified.
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        try:
            value: DualTypes = self.cashflow(fx) * disc_curve_[self.payment]  # type: ignore[operator]
        except TypeError as e:
            # either fixed rate is None
            if isinstance(self.fixed_rate, NoInput):
                raise TypeError("`fixed_rate` must be set on the Period for an `npv`.")
            else:
                raise e
        return _maybe_local(value, local, self.settlement_currency, fx, base)

    def cashflows(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *FixedPeriod*.
        See :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        fx_, base_ = _get_fx_and_base(self.settlement_currency, fx, base)

        if isinstance(disc_curve_, NoInput) or isinstance(self.fixed_rate, NoInput):
            npv = None
            npv_fx = None
        else:
            npv_dual: DualTypes = self.npv(curve, disc_curve_, fx=fx, local=False)  # type: ignore[assignment]
            npv = _dual_float(npv_dual)
            npv_fx = npv * _dual_float(fx_)

        cashflow = _float_or_none(self.cashflow(fx))
        return {
            **BasePeriod.cashflows(self, curve, disc_curve_, fx_, base_),
            defaults.headers["pair"]: self.pair,
            defaults.headers["currency"]: self.settlement_currency,
            defaults.headers["rate"]: self.fixed_rate,
            defaults.headers["spread"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["index_value"]: _float_or_none(self._get_fx_fixing(fx)),
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: npv_fx,
        }
