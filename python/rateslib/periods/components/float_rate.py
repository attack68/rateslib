from __future__ import annotations

import warnings
from math import prod
from typing import TYPE_CHECKING

import numpy as np
from pandas import Series, isna

import rateslib.errors as err
from rateslib import NoInput, defaults
from rateslib.curves import _BaseCurve, index_left
from rateslib.curves.utils import _CurveType
from rateslib.dual import Dual, Dual2, Variable
from rateslib.enums.generics import Err, Ok, _drb
from rateslib.enums.parameters import (
    FloatFixingMethod,
    SpreadCompoundMethod,
    _get_float_fixing_method,
    _get_spread_compound_method,
)
from rateslib.fixings import FixingMissingForecasterError
from rateslib.periods.utils import _get_rfr_curve_from_dict
from rateslib.scheduling import add_tenor, dcf
from rateslib.scheduling.convention import Convention
from rateslib.scheduling.float_rate_index import FloatRateSeries, _get_float_rate_series_or_blank
from rateslib.scheduling.frequency import _get_frequency, _get_tenor_from_frequency

if TYPE_CHECKING:
    from rateslib.typing import (
        Arr1dF64,
        Arr1dObj,
        CalTypes,
        Convention,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Frequency,
        Result,
        Series,
        _BaseCurve_,
        datetime,
        str_,
    )

####################################################################################################

# Main function of this module


def rate_value(
    start: datetime,
    end: datetime,
    rate_curve: CurveOption_ = NoInput(0),
    *,
    rate_fixings: DualTypes_ | str = NoInput(0),
    frequency: Frequency | str_ = NoInput(0),
    rate_series: FloatRateSeries | str_ = NoInput(0),
    fixing_method: FloatFixingMethod | str = FloatFixingMethod.RFRPaymentDelay,
    method_param: int = 0,
    spread_compound_method: SpreadCompoundMethod | str = SpreadCompoundMethod.NoneSimple,
    float_spread: DualTypes = 0.0,
    stub: bool = False,
) -> DualTypes:
    return try_rate_value(
        start=start,
        end=end,
        rate_curve=rate_curve,
        rate_series=rate_series,
        frequency=frequency,
        rate_fixings=rate_fixings,
        fixing_method=fixing_method,
        method_param=method_param,
        spread_compound_method=spread_compound_method,
        float_spread=float_spread,
        stub=stub,
    ).unwrap()


def try_rate_value(
    start: datetime,
    end: datetime,
    rate_curve: CurveOption_ = NoInput(0),
    *,
    rate_series: FloatRateSeries | str_ = NoInput(0),
    frequency: Frequency | str_ = NoInput(0),
    rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    fixing_method: FloatFixingMethod | str = FloatFixingMethod.RFRPaymentDelay,
    method_param: int = 0,
    spread_compound_method: SpreadCompoundMethod | str = SpreadCompoundMethod.NoneSimple,
    float_spread: DualTypes = 0.0,
    stub: bool = False,
) -> Result[DualTypes]:
    """
    Derive a floating rate value from a combination of market inputs.


    """
    fm = _get_float_fixing_method(fixing_method)
    scm = _get_spread_compound_method(spread_compound_method)
    rs = _get_float_rate_series_or_blank(rate_series)
    if fm == FloatFixingMethod.IBOR:
        return _IBORRate._rate(
            start=start,
            end=end,
            rate_curve=rate_curve,
            rate_fixings=rate_fixings,
            method_param=method_param,
            float_spread=_drb(0.0, float_spread),
            stub=stub,
            rate_series=rs,
            frequency=_get_frequency(frequency, NoInput(0), NoInput(0)),
        )
    else:  #  RFR based
        if isinstance(rate_curve, dict):
            rate_curve_: _BaseCurve_ = _get_rfr_curve_from_dict(rate_curve)
        else:
            rate_curve_ = rate_curve
        r_result = _RFRRate._rate(
            start=start,
            end=end,
            rate_curve=rate_curve_,
            rate_fixings=rate_fixings,
            fixing_method=fm,
            method_param=method_param,
            spread_compound_method=scm,
            float_spread=float_spread,
            rate_series=rs,
        )
        if isinstance(r_result, Err):
            return r_result
        else:
            return Ok(r_result.unwrap()[0])


class _IBORRate:
    @staticmethod
    def _rate(
        *,
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        start: datetime,
        end: datetime,
        method_param: int,
        stub: bool,
        float_spread: DualTypes,
        rate_series: FloatRateSeries | NoInput,
        frequency: Frequency,
    ) -> Result[DualTypes]:
        rate_series_ = _maybe_get_rate_series_from_curve(
            rate_curve=rate_curve, rate_series=rate_series, method_param=method_param
        )
        fixing_date = rate_series_.calendar.lag_bus_days(start, -rate_series_.lag, settlement=False)
        if stub:
            # TODO: pass through tenor convention and modifier to the interpolated stub
            return _IBORRate._rate_interpolated_stub(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                float_spread=float_spread,
                rate_series=rate_series_,
            )
        else:
            return _IBORRate._rate_single_tenor(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_interpolated_stub(
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
        rate_series: FloatRateSeries,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, NoInput):
            # will attempt to forecast stub period from rate_curve
            if isinstance(rate_curve, dict):
                return _IBORRate._rate_interpolated_stub_from_curve_dict(
                    rate_curve=rate_curve,
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                )
            else:
                return _IBORRate._rate_stub_forecast_from_curve(
                    rate_curve=rate_curve,
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                )
        else:
            # will maybe find relevant fixing values in Series
            return _IBORRate._rate_interpolated_stub_maybe_from_fixings(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                rate_series=rate_series,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_interpolated_stub_maybe_from_fixings(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        rate_fixings: DualTypes | Series[DualTypes] | str,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
        rate_series: FloatRateSeries,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, str):
            tenors, dates, fixings = defaults.fixings.get_stub_ibor_fixings(
                value_start_date=start,
                value_end_date=end,
                fixing_calendar=rate_series.calendar,
                fixing_modifier=rate_series.modifier,
                fixing_identifier=rate_fixings,
                fixing_date=fixing_date,
            )
            if len(tenors) == 0:
                # nothing found
                return _IBORRate._rate_interpolated_stub(
                    rate_curve=rate_curve,
                    rate_fixings=NoInput(0),  # no fixings are found
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                    rate_series=rate_series,
                )
            elif len(tenors) == 1:
                if fixings[0] is None:
                    return _IBORRate._rate_interpolated_stub(
                        rate_curve=rate_curve,
                        rate_fixings=NoInput(0),  # no fixings are found
                        fixing_date=fixing_date,
                        start=start,
                        end=end,
                        float_spread=float_spread,
                        rate_series=rate_series,
                    )
                return Ok(fixings[0] + float_spread / 100.0)
            else:
                if fixings[0] is None or fixings[1] is None:
                    # missing data exists
                    return _IBORRate._rate_interpolated_stub(
                        rate_curve=rate_curve,
                        rate_fixings=NoInput(0),  # no fixings are found
                        fixing_date=fixing_date,
                        start=start,
                        end=end,
                        float_spread=float_spread,
                        rate_series=rate_series,
                    )
                return Ok(
                    _IBORRate._interpolated_stub_rate(
                        left_date=dates[0],
                        right_date=dates[1],
                        left_rate=fixings[0],
                        right_rate=fixings[1],
                        maturity_date=end,
                        float_spread=float_spread,
                    )
                )
        elif isinstance(rate_fixings, Series):
            raise ValueError(err.VE_FIXINGS_BAD_TYPE)
        else:
            return Ok(rate_fixings + float_spread / 100.0)

    @staticmethod
    def _rate_interpolated_stub_from_curve_dict(
        rate_curve: dict[str, _BaseCurve],
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Get the rate on all available curves in dict and then determine the ones to interpolate.
        """

        def _rate(c: _BaseCurve, tenor: str) -> DualTypes:
            if c._base_type == _CurveType.dfs:
                return c._rate_with_raise(start, tenor)
            else:  # values
                return c._rate_with_raise(fixing_date, tenor)  # tenor is not used on LineCurve

        try:
            values = {
                add_tenor(start, k, v.meta.modifier, v.meta.calendar): _rate(v, k)
                for k, v in rate_curve.items()
            }
        except Exception as e:
            return Err(e)
        values = dict(sorted(values.items()))
        dates, rates = list(values.keys()), list(values.values())
        if end > dates[-1]:
            warnings.warn(
                "Interpolated stub period has a length longer than the provided "
                "IBOR curve tenors: using the longest IBOR value.",
                UserWarning,
            )
            ret: DualTypes = rates[-1]
        elif end < dates[0]:
            warnings.warn(
                "Interpolated stub period has a length shorter than the provided "
                "IBOR curve tenors: using the shortest IBOR value.",
                UserWarning,
            )
            ret = rates[0]
        else:
            i = index_left(dates, len(dates), end)
            ret = rates[i] + (rates[i + 1] - rates[i]) * (
                (end - dates[i]).days / (dates[i + 1] - dates[i]).days
            )
        return Ok(ret + float_spread / 100.0)

    @staticmethod
    def _rate_single_tenor(
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, NoInput):
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            return _IBORRate._rate_tenor_maybe_from_fixings(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_tenor_maybe_from_fixings(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        rate_fixings: DualTypes | Series[DualTypes] | str,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, str | Series):
            if isinstance(rate_fixings, str):
                identifier = rate_fixings
                fixings = defaults.fixings[identifier][1]
            else:
                identifier = "<SERIES_OBJECT>"
                fixings = rate_fixings

            try:
                fixing = fixings.loc[fixing_date]
                return Ok(fixing + float_spread / 100.0)
            except KeyError:
                warnings.warn(
                    f"Fixings are provided in series: '{identifier}', but the value for required"
                    f" date: {fixing_date} is not found.\nAttempting to forecast from "
                    f"the `rate_curve`.",
                )
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            # is just a scalar value so return directly.
            return Ok(rate_fixings + float_spread / 100.0)

    @staticmethod
    def _rate_tenor_forecast_from_curve(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        tenor = _get_tenor_from_frequency(frequency)
        if isinstance(rate_curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_TENOR_IBOR))
        elif isinstance(rate_curve, dict):
            remapped_rate_curve = {k.lower(): v for k, v in rate_curve.items()}
            rate_curve_ = remapped_rate_curve[tenor.lower()]
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve_,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            if rate_curve._base_type == _CurveType.dfs:
                try:
                    r = rate_curve._rate_with_raise(start, tenor) + float_spread / 100.0
                except Exception as e:
                    return Err(e)
                else:
                    return Ok(r)
            else:
                try:
                    r = rate_curve._rate_with_raise(fixing_date, NoInput(0)) + float_spread / 100.0
                except Exception as e:
                    return Err(e)
                else:
                    return Ok(r)

    @staticmethod
    def _rate_stub_forecast_from_curve(
        rate_curve: _BaseCurve_,
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_STUB_IBOR))

        if rate_curve._base_type == _CurveType.dfs:
            try:
                r = rate_curve._rate_with_raise(start, end) + float_spread / 100.0
            except Exception as e:
                return Err(e)
            else:
                return Ok(r)
        else:
            try:
                r = rate_curve[fixing_date] + float_spread / 100.0
            except Exception as e:
                return Err(e)
            else:
                return Ok(r)

    @staticmethod
    def _interpolated_stub_rate(
        left_date: datetime,
        right_date: datetime,
        left_rate: DualTypes,
        right_rate: DualTypes,
        maturity_date: datetime,
        float_spread: DualTypes,
    ) -> DualTypes:
        return (
            left_rate
            + (maturity_date - left_date).days
            / (right_date - left_date).days
            * (right_rate - left_rate)
            + float_spread / 100.0
        )


class _RFRRate:
    """
    Class for maintaining methods related to calculating the period rate for an RFR compounded
    period. These periods have multiple branches depending upon;

    - which `fixing_method` has been selected.
    - which `spread_compound_method` has been selected (if the `float_spread` is non-zero).
    - whether there are any known fixings that must be populated to the calculation or unknown
      fixings must be forecast by some curve.

    """

    @staticmethod
    def _rate(
        start: datetime,
        end: datetime,
        rate_curve: _BaseCurve_,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        float_spread: DualTypes,
        rate_series: FloatRateSeries | NoInput,
    ) -> Result[  # type: ignore[type-var]
        tuple[
            DualTypes,
            tuple[datetime, datetime] | None,
            tuple[datetime, datetime] | None,
            Arr1dObj | None,
            Arr1dObj | None,
            Arr1dF64 | None,
            Arr1dF64 | None,
            Series[DualTypes] | None,
            Series[DualTypes] | None,
            Series[DualTypes] | None,
        ]
    ]:
        """
        To avoid repeated calculation, this function will pass back the data it calculates.
        In some short-circuited calculation not all data will have been calculated and returns
        None

        - 0: rate
        - 1: date_boundary_obs
        - 2: date_boundary_dcf
        - 3: dates_obs
        - 4: dates_dcf
        - 5: dcfs_obs
        - 6: dcfs_dcf
        - 7: fixing_rates
        - 8: populated
        - 9: unpopulated

        """

        if isinstance(rate_fixings, int | float | Dual | Dual2 | Variable):
            # a scalar value is assumed to have been pre-computed **including** the float spread
            # otherwise this information is of no use, since a computation including a
            # complicated float spread cannot be performed on just a compounded or average rate.
            return Ok((rate_fixings,) + (None,) * 9)

        rate_series_ = _maybe_get_rate_series_from_curve(
            rate_curve=rate_curve,
            rate_series=rate_series,
            method_param=method_param,
        )

        bounds_obs, bounds_dcf, is_matching = _RFRRate._adjust_dates(
            start=start,
            end=end,
            fixing_method=fixing_method,
            method_param=method_param,
            fixing_calendar=rate_series_.calendar,
        )

        # >>> short-circuit here before any complex calculation or date lookup is performed.
        # EFFICIENT CALCULATION:
        if _RFRRate._is_rfr_efficient(
            rate_curve=rate_curve,
            rate_fixings=rate_fixings,
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            fixing_method=fixing_method,
        ):
            r_result = _RFRRate._efficient_calculation(
                rate_curve=rate_curve,  # type: ignore[arg-type]  # is pre-checked
                bounds_obs=bounds_obs,
                float_spread=float_spread,
            )
            if isinstance(r_result, Err):
                return r_result
            else:
                return Ok((r_result.unwrap(), bounds_obs, bounds_dcf) + (None,) * 7)

        dates_obs, dates_dcf, dcfs_obs, dcfs_dcf, populated, unpopulated, fixing_rates = (
            _RFRRate._get_dates_and_fixing_rates_from_fixings(
                rate_series=rate_series_,
                bounds_obs=bounds_obs,
                bounds_dcf=bounds_dcf,
                is_matching=is_matching,
                rate_fixings=rate_fixings,
            )
        )

        # >>> short circuit and perform a semi-efficient calculation splicing fixings with DFs
        # SEMI-EFFICIENT CALCULATION:
        if _RFRRate._is_rfr_efficient(
            rate_curve, NoInput(0), float_spread, spread_compound_method, fixing_method
        ):
            r = _RFRRate._semi_efficient_calculation(
                rate_curve=rate_curve,  # type: ignore[arg-type]  # guaranteed by if statement
                populated=populated,
                unpopulated=unpopulated,
                obs_date_boundary=bounds_obs,
                float_spread=float_spread,
                fixing_dcfs=dcfs_dcf,
            )
            return Ok(
                (
                    r,
                    bounds_obs,
                    bounds_dcf,
                    dates_obs,
                    dates_dcf,
                    dcfs_obs,
                    dcfs_dcf,
                    fixing_rates,
                    populated,
                    unpopulated,
                )
            )

        update = _RFRRate._forecast_fixing_rates_from_curve(
            unpopulated=unpopulated,
            populated=populated,
            fixing_rates=fixing_rates,
            rate_curve=rate_curve,
            dates_obs=dates_obs,
            dcfs_obs=dcfs_obs,
        )
        if isinstance(update, Err):
            return update

        # INEFFICIENT CALCULATION having derived all individual fixings.
        r_result = _RFRRate._inefficient_calculation(
            fixing_rates=fixing_rates,
            fixing_dcfs=dcfs_dcf,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            float_spread=float_spread,
        )
        if isinstance(r_result, Err):
            return r_result
        else:
            return Ok(
                (
                    r_result.unwrap(),
                    bounds_obs,
                    bounds_dcf,
                    dates_obs,
                    dates_dcf,
                    dcfs_obs,
                    dcfs_dcf,
                    fixing_rates,
                    populated,
                    unpopulated,
                )
            )

    @staticmethod
    def _efficient_calculation(
        rate_curve: _BaseCurve,  # discount factors only
        bounds_obs: tuple[datetime, datetime],
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Perform an efficient calculation only after the `_is_rfr_efficient` check is performed.

        This calculation uses only discount factors and does not calculate individual fixing rates.
        """
        try:
            r = (
                rate_curve._rate_with_raise(
                    effective=bounds_obs[0],
                    termination=bounds_obs[1],
                    # no other arguments are necessary following _is_efficient check
                )
                + float_spread / 100.0
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(r)

    @staticmethod
    def _semi_efficient_calculation(
        rate_curve: _BaseCurve,
        populated: Series[DualTypes],  # type: ignore[type-var]
        fixing_dcfs: Arr1dF64,
        unpopulated: Series[DualTypes],  # type: ignore[type-var]
        obs_date_boundary: tuple[datetime, datetime],
        float_spread: DualTypes,
    ) -> DualTypes:
        """
        Perform an efficient calculation only after the `_is_rfr_efficient` check is performed.

        This calculation combines some known fixing values with a forecast people calculated
        using discount factors and not by calculating a number of individual fixing rates.
        """
        populated_index = prod(
            [
                1.0 + d * r / 100.0
                for r, d in zip(populated, fixing_dcfs[: len(populated)], strict=False)
            ]
        )
        # TODO this is not date safe, i.e. a date maybe before the curve starts and DF is zero.
        if len(unpopulated) == 0:  # i.e. all fixings are known without needing to forecast
            unpopulated_index: DualTypes = 1.0
        else:
            unpopulated_index = rate_curve[unpopulated.index[0]] / rate_curve[obs_date_boundary[1]]
        rate: DualTypes = ((populated_index * unpopulated_index) - 1.0) * 100.0 / fixing_dcfs.sum()
        return rate + float_spread / 100.0

    @staticmethod
    def _inefficient_calculation(
        fixing_rates: Series,
        fixing_dcfs: Arr1dF64,
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Perform a full calculation forecasting every individual fixing rate and then compounding
        or averaging each of them up in turn, combining a float spread if necessary.
        """
        if fixing_method in [FloatFixingMethod.RFRLockout, FloatFixingMethod.RFRLockoutAverage]:
            # overwrite fixings
            if method_param >= len(fixing_rates):
                return Err(
                    ValueError(err.VE_LOCKOUT_METHOD_PARAM.format(method_param, fixing_rates))
                )
            for i in range(1, method_param + 1):
                fixing_rates.iloc[-i] = fixing_rates.iloc[-(method_param + 1)]

        if fixing_method in [
            FloatFixingMethod.RFRLockoutAverage,
            FloatFixingMethod.RFRLookbackAverage,
            FloatFixingMethod.RFRObservationShiftAverage,
            FloatFixingMethod.RFRPaymentDelayAverage,
        ]:
            return _RFRRate._calculator_rate_rfr_avg_with_spread(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                rates=fixing_rates.to_numpy(),
                dcf_vals=fixing_dcfs,
            )
        else:
            return _RFRRate._calculator_rate_rfr_isda_compounded_with_spread(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                rates=fixing_rates.to_numpy(),
                dcf_vals=fixing_dcfs,
            )

    @staticmethod
    def _get_dates_and_fixing_rates_from_fixings(
        rate_series: FloatRateSeries,
        bounds_obs: tuple[datetime, datetime],
        bounds_dcf: tuple[datetime, datetime],
        is_matching: bool,
        rate_fixings: Series[DualTypes] | str_,  # type: ignore[type-var]
    ) -> tuple[  # type: ignore[type-var]
        Arr1dObj,
        Arr1dObj,
        Arr1dF64,
        Arr1dF64,
        Series[DualTypes],
        Series[DualTypes],
        Series[DualTypes],
    ]:
        """
        For an RFR period, construct the necessary fixing dates and DCF schedule.
        Populate fixings from a Series if any values are available to yield.
        Return Series objects.

        """
        dates_obs, dates_dcf, fixing_rates = _RFRRate._get_obs_and_dcf_dates(
            fixing_calendar=rate_series.calendar,
            fixing_convention=rate_series.convention,
            obs_date_boundary=bounds_obs,
            dcf_date_boundary=bounds_dcf,
            is_matching=is_matching,
        )
        dcfs_dcf = _RFRRate._get_dcf_values(
            dcf_dates=dates_dcf,
            fixing_convention=rate_series.convention,
            fixing_calendar=rate_series.calendar,
        )
        if is_matching:
            dcfs_obs = dcfs_dcf.copy()
        else:
            dcfs_obs = _RFRRate._get_dcf_values(
                dcf_dates=dates_obs,
                fixing_convention=rate_series.convention,
                fixing_calendar=rate_series.calendar,
            )

        # populate Series with values
        if isinstance(rate_fixings, NoInput):
            populated: Series[DualTypes] = Series(index=[], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
            unpopulated: Series[DualTypes] = Series(index=dates_obs[:-1], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
        elif isinstance(rate_fixings, str | Series):
            fixing_rates, populated, unpopulated = (
                _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
                    fixing_rates=fixing_rates,
                    rate_fixings=rate_fixings,
                )
            )
        else:
            raise ValueError(err.VE_FIXINGS_BAD_TYPE)  # unknown fixings type fixings runtime issue

        return dates_obs, dates_dcf, dcfs_obs, dcfs_dcf, populated, unpopulated, fixing_rates

    @staticmethod
    def _forecast_fixing_rates_from_curve(
        unpopulated: Series[DualTypes],  # type: ignore[type-var]
        populated: Series[DualTypes],  # type: ignore[type-var]
        fixing_rates: Series[DualTypes],  # type: ignore[type-var]
        rate_curve: _BaseCurve_,
        dates_obs: Arr1dObj,
        dcfs_obs: Arr1dF64,
    ) -> Result[None]:
        # determine upopulated fixings from the curve
        if len(unpopulated) > 0 and isinstance(rate_curve, NoInput):
            return Err(FixingMissingForecasterError())  # missing data - needs a rate_curve

        unpopulated_obs_dates = dates_obs[len(populated) :]
        if len(unpopulated_obs_dates) > 1:
            if isinstance(rate_curve, NoInput):
                return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_RFR))

            if rate_curve._base_type == _CurveType.values:
                try:
                    r = [
                        rate_curve._rate_with_raise(unpopulated_obs_dates[_], NoInput(0))
                        for _ in range(len(unpopulated))
                    ]
                except Exception as e:
                    return Err(e)
            else:
                v = np.array([rate_curve[_] for _ in unpopulated_obs_dates])
                r = (v[:-1] / v[1:] - 1) * 100 / dcfs_obs[len(populated) :]
            unpopulated = Series(
                index=unpopulated.index,
                data=r,
            )
        fixing_rates.update(unpopulated)
        return Ok(None)

    @staticmethod
    def _push_rate_fixings_as_series_to_fixing_rates(
        fixing_rates: Series[DualTypes],  # type: ignore[type-var]
        rate_fixings: str | Series[DualTypes],  # type: ignore[type-var]
    ) -> tuple[Series[DualTypes], Series[DualTypes], Series[DualTypes]]:  # type: ignore[type-var]
        """
        Populates an empty fixings_rates Series with values from a looked up fixings collection.
        """
        if isinstance(rate_fixings, str):
            fixing_series = defaults.fixings[rate_fixings][1]
        else:
            fixing_series = rate_fixings
        if fixing_rates.index[0] > fixing_series.index[-1]:
            # then no fixings in scope, so no changes
            return fixing_rates, Series(index=[], data=np.nan), fixing_rates.copy()  # type: ignore[return-value]
        else:
            fixing_rates.update(fixing_series)

        # validate for missing and expected fixings in the fixing Series
        nans = isna(fixing_rates)
        populated, unpopulated = fixing_rates[~nans], fixing_rates[nans]
        if (
            len(unpopulated) > 0
            and len(populated) > 0
            and unpopulated.index[0] < populated.index[-1]
        ):
            raise ValueError(
                err.VE02_5.format(  # there is at least one missing fixing data item
                    rate_fixings,
                    fixing_rates[nans].index[0].strftime("%d-%m-%Y"),
                    fixing_rates[~nans].index[-1].strftime("%d-%m-%Y"),
                )
            )

        # validate for unexpected fixings provided in the fixings Series
        if 0 < len(populated) < len(fixing_series[populated.index[0] : populated.index[-1]]):
            # then fixing series contains an unexpected fixing.
            warnings.warn(
                err.W02_0.format(
                    rate_fixings,
                    populated.index[0].strftime("%d-%m-%Y"),
                    populated.index[-1].strftime("%d-%m-%Y"),
                ),
                UserWarning,
            )

        return fixing_rates, populated, unpopulated

    @staticmethod
    def _adjust_dates(
        start: datetime,
        end: datetime,
        fixing_method: FloatFixingMethod,
        method_param: int,
        fixing_calendar: CalTypes,
    ) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime], bool]:
        """
        For each different RFR fixing method adjust the start and end date of the associated
        period to return adjusted start and end dates for the fixing set as well as the
        DCF set.

        For all methods except 'lookback', these dates will align with each other.
        For 'lookback' the observed RFRs are applied over different DCFs that do not naturally
        align.
        """
        # Depending upon method get the observation dates and dcf dates
        if fixing_method in [
            FloatFixingMethod.RFRPaymentDelay,
            FloatFixingMethod.RFRPaymentDelayAverage,
            FloatFixingMethod.RFRLockout,
            FloatFixingMethod.RFRLockoutAverage,
        ]:
            start_obs, end_obs = start, end
            start_dcf, end_dcf = start, end
            is_matching = True
        elif fixing_method in [
            FloatFixingMethod.RFRObservationShift,
            FloatFixingMethod.RFRObservationShiftAverage,
        ]:
            start_obs = fixing_calendar.lag_bus_days(start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(end, -method_param, settlement=False)
            start_dcf, end_dcf = start_obs, end_obs
            is_matching = True
        else:
            # fixing_method in [
            #    FloatFixingMethod.RFRLookback,
            #    FloatFixingMethod.RFRLookbackAverage,
            # ]:
            start_obs = fixing_calendar.lag_bus_days(start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(end, -method_param, settlement=False)
            start_dcf, end_dcf = start, end
            is_matching = False

        return (start_obs, end_obs), (start_dcf, end_dcf), is_matching

    @staticmethod
    def _get_obs_and_dcf_dates(
        fixing_calendar: CalTypes,
        fixing_convention: Convention,
        obs_date_boundary: tuple[datetime, datetime],
        dcf_date_boundary: tuple[datetime, datetime],
        is_matching: bool,
    ) -> tuple[Arr1dObj, Arr1dObj, Series[DualTypes]]:  # type: ignore[type-var]
        # construct empty Series for rates and DCFs
        obs_dates = np.array(fixing_calendar.bus_date_range(*obs_date_boundary))
        fixing_rates: Series[DualTypes] = Series(index=obs_dates[:-1], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
        if is_matching:
            dcf_dates = obs_dates
        else:
            dcf_dates = np.array(fixing_calendar.bus_date_range(*dcf_date_boundary))
        return obs_dates, dcf_dates, fixing_rates

    @staticmethod
    def _get_dcf_values(
        dcf_dates: Arr1dObj,
        fixing_convention: Convention,
        fixing_calendar: CalTypes,
    ) -> Arr1dF64:
        if fixing_convention == Convention.Act365F:
            days = np.fromiter((_.days for _ in dcf_dates[1:] - dcf_dates[:-1]), float)
            return days / 365.0
        elif fixing_convention == Convention.Act360:
            days = np.fromiter((_.days for _ in dcf_dates[1:] - dcf_dates[:-1]), float)
            return days / 360.0
        elif fixing_convention == Convention.Bus252:
            return np.array([1.0 / 252.0] * (len(dcf_dates) - 1))
        else:
            # this is unconventional fixing convention. Should maybe be avoided altogether.
            return np.array(
                [
                    dcf(
                        start=dcf_dates[i],
                        end=dcf_dates[i + 1],
                        convention=fixing_convention,
                        calendar=fixing_calendar,
                    )
                    for i in range(len(dcf_dates) - 1)
                ]
            )

    @staticmethod
    def _is_rfr_efficient(
        rate_curve: _BaseCurve_,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        fixing_method: FloatFixingMethod,
    ) -> bool:
        """
        Check all of the conditions to return an RFR rate directly from discount factors.

        - A rate curve must be available and be based on DFs.
        - There cannot be any known fixings that must be incorporated into the calculation.
        - Only PaymentDelay and ObservationShift fixing methods are suitable for this calculation.
        - Only NoneSimple spread compound method is suitable, or the float spread must be 0.0.

        """
        return (
            isinstance(rate_curve, _BaseCurve)
            and rate_curve._base_type == _CurveType.dfs
            and isinstance(rate_fixings, NoInput)
            and fixing_method
            in [FloatFixingMethod.RFRPaymentDelay, FloatFixingMethod.RFRObservationShift]
            and (float_spread == 0.0 or spread_compound_method == SpreadCompoundMethod.NoneSimple)
        )

    @staticmethod
    def _calculator_rate_rfr_avg_with_spread(
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        rates: Arr1dF64,
        dcf_vals: Arr1dF64,
    ) -> Result[DualTypes]:
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
        if spread_compound_method != SpreadCompoundMethod.NoneSimple:
            return Err(ValueError(err.VE_SPREAD_METHOD_RFR.format(spread_compound_method)))
        else:
            _: DualTypes = (dcf_vals * rates).sum() / dcf_vals.sum() + float_spread / 100
            return Ok(_)

    @staticmethod
    def _calculator_rate_rfr_isda_compounded_with_spread(
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        rates: Arr1dObj,
        dcf_vals: Arr1dF64,
    ) -> Result[DualTypes]:
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
        if float_spread == 0 or spread_compound_method == SpreadCompoundMethod.NoneSimple:
            _: DualTypes = (
                (1 + dcf_vals * rates / 100).prod() - 1
            ) * 100 / dcf_vals.sum() + float_spread / 100
            return Ok(_)
        elif spread_compound_method == SpreadCompoundMethod.ISDACompounding:
            _ = (
                ((1 + dcf_vals * (rates / 100 + float_spread / 10000)).prod() - 1)
                * 100
                / dcf_vals.sum()
            )
            return Ok(_)
        else:  # spread_compound_method == SpreadCompoundMethod.ISDAFlatCompounding:
            sub_cashflows = (rates / 100 + float_spread / 10000) * dcf_vals
            C_i = 0.0
            for i in range(1, len(sub_cashflows)):
                C_i += sub_cashflows[i - 1]
                sub_cashflows[i] += C_i * rates[i] / 100 * dcf_vals[i]
            _ = sub_cashflows.sum() * 100 / dcf_vals.sum()
            return Ok(_)


def _maybe_get_rate_series_from_curve(
    rate_curve: CurveOption_,
    rate_series: FloatRateSeries | NoInput,
    method_param: int,
) -> FloatRateSeries:
    """Get a rate fixing calendar and convention from a Curve or the alternatives if not given."""

    if isinstance(rate_curve, NoInput):
        if isinstance(rate_series, NoInput):
            raise ValueError(err.VE_NEEDS_CURVE_OR_INDEX)
        else:
            # get params from rate_index
            return rate_series
    else:
        if isinstance(rate_curve, dict):
            cal_ = list(rate_curve.values())[0].meta.calendar
            conv_ = list(rate_curve.values())[0].meta.convention
            mod_ = list(rate_curve.values())[0].meta.modifier
        else:
            cal_ = rate_curve.meta.calendar
            conv_ = rate_curve.meta.convention
            mod_ = rate_curve.meta.modifier

        if isinstance(rate_series, NoInput):
            # get params from rate_curve
            return FloatRateSeries(
                lag=method_param,
                calendar=cal_,
                convention=conv_,
                modifier=mod_,
                eom=False,  # TODO: un hard code this
            )
        else:
            if rate_series.convention != conv_:
                raise ValueError(
                    err.MISMATCH_RATE_INDEX_PARAMETERS.format(
                        "convention", conv_, rate_series.convention
                    )
                )
            # dual parameters may be specified
            # get params from rate_index
            return rate_series
