from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat, isna

import rateslib.errors as err
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
    _try_disc_required_maybe_from_curve,
    _validate_obj_not_no_input,
)
from rateslib.curves.utils import average_rate
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.fixing_data import _find_neighbouring_tenors
from rateslib.periods.components.base_period import BasePeriod
from rateslib.periods.components.float_rate import (
    _maybe_get_rate_series_from_curve,
    _RFRRate,
    try_rate_value,
)
from rateslib.periods.components.parameters import IBORFixing, IBORStubFixing, _init_FloatRateParams
from rateslib.periods.utils import _get_rfr_curve_from_dict, _trim_df_by_index
from rateslib.scheduling import dcf
from rateslib.scheduling.float_rate_index import FloatRateSeries
from rateslib.scheduling.frequency import _get_tenor_from_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Arr1dF64,
        Arr1dObj,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Frequency,
        Result,
        _BaseCurve,
        _BaseCurve_,
        _FloatRateParams,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FloatPeriod(BasePeriod):
    rate_params: _FloatRateParams

    def __init__(
        self,
        *,
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rate_params = _init_FloatRateParams(
            _method_param=method_param,
            _float_spread=float_spread,
            _spread_compound_method=spread_compound_method,
            _fixing_method=fixing_method,
            _fixing_series=fixing_series,
            _fixing_frequency=fixing_frequency,
            _rate_fixings=rate_fixings,
            _accrual_start=self.period_params.start,
            _accrual_end=self.period_params.end,
            _period_calendar=self.period_params.calendar,
            _period_convention=self.period_params.convention,
            _period_adjuster=self.period_params.adjuster,
            _period_frequency=self.period_params.frequency,
            _period_stub=self.period_params.stub,
        )

    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> Result[DualTypes]:
        r = self.try_rate(rate_curve)
        if r.is_err:
            return r
        return Ok(-self.settlement_params.notional * r.unwrap() * 0.01 * self.period_params.dcf)

    def try_unindexed_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in ``reference_currency``
        without indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        disc_curve_ = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(disc_curve_, Err):
            return disc_curve_

        if (
            self.rate_params.spread_compound_method == SpreadCompoundMethod.NoneSimple
            or self.rate_params.float_spread == 0
        ):
            # then analytic_delta is not impacted by float_spread compounding
            dr_dz: float = 1.0
        else:
            _ = self.rate_params.float_spread
            self.rate_params.float_spread = Variable(_dual_float(_), ["z_float_spread"])
            rate: Result[DualTypes] = self.try_rate(rate_curve)
            if rate.is_err:
                return rate
            dr_dz = gradient(rate.unwrap(), ["z_float_spread"])[0] * 100
            self.rate_params.float_spread = _

        return Ok(
            self.settlement_params.notional
            * 0.0001
            * dr_dz
            * self.period_params.dcf
            * disc_curve_.unwrap()[self.settlement_params.payment]
        )

    def try_unindexed_reference_fixings_exposure(
        self,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        right: datetime_ = NoInput(0),
    ) -> Result[DataFrame]:
        if self.rate_params.fixing_method == FloatFixingMethod.IBOR:
            return _FixingsExposureCalculator.ibor(
                p=self,
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                right=right,
            )
        else:
            if isinstance(rate_curve, dict):
                rate_curve_: _BaseCurve_ = _get_rfr_curve_from_dict(rate_curve)
            else:
                rate_curve_ = rate_curve
            return _FixingsExposureCalculator.rfr(
                p=self,
                rate_curve=_validate_obj_not_no_input(rate_curve_, "rate_curve"),
                disc_curve=disc_curve,
                right=right,
            )

    def try_rate(self, rate_curve: CurveOption_) -> Result[DualTypes]:
        rate_fixing = self.rate_params.rate_fixing.value
        if isinstance(rate_fixing, NoInput):
            return try_rate_value(
                start=self.rate_params.accrual_start,
                end=self.rate_params.accrual_end,
                rate_curve=NoInput(0) if rate_curve is None else rate_curve,
                rate_fixings=self.rate_params.rate_fixing.identifier,
                fixing_method=self.rate_params.fixing_method,
                method_param=self.rate_params.method_param,
                spread_compound_method=self.rate_params.spread_compound_method,
                float_spread=self.rate_params.float_spread,
                stub=self.period_params.stub,
                frequency=self.rate_params.fixing_frequency,
                rate_series=self.rate_params.fixing_series,
            )
        else:
            # the fixing value is a scalar so a Curve should not be required for this calculation
            return try_rate_value(
                start=self.rate_params.accrual_start,
                end=self.rate_params.accrual_end,
                rate_curve=NoInput(0),
                rate_fixings=rate_fixing,
                fixing_method=self.rate_params.fixing_method,
                method_param=self.rate_params.method_param,
                spread_compound_method=self.rate_params.spread_compound_method,
                float_spread=self.rate_params.float_spread,
                stub=self.period_params.stub,
                frequency=self.rate_params.fixing_frequency,
                rate_series=self.rate_params.fixing_series,
            )

    def rate(self, rate_curve: CurveOption_) -> DualTypes:
        return self.try_rate(rate_curve).unwrap()


class NonDeliverableFloatPeriod(FloatPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
        if self.is_indexed:
            raise ValueError(err.VE_HAS_INDEX_PARAMS.format(type(self).__name__))


class IndexFloatPeriod(FloatPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if self.is_non_deliverable:
            raise ValueError(err.VE_HAS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class NonDeliverableIndexFloatPeriod(FloatPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class _FixingsExposureCalculator:
    @classmethod
    def rfr(
        cls, p: FloatPeriod, rate_curve: _BaseCurve, disc_curve: _BaseCurve_, right: datetime_
    ) -> Result[DataFrame]:
        rate_fixing: DualTypes_ = p.rate_params.rate_fixing.value

        rate_series_ = _maybe_get_rate_series_from_curve(
            rate_curve=rate_curve,
            rate_series=p.rate_params.fixing_series,
            method_param=p.rate_params.method_param,
        )

        data = _RFRRate._rate(
            start=p.period_params.start,
            end=p.period_params.end,
            rate_curve=rate_curve,
            rate_fixings=p.rate_params.rate_fixing.identifier
            if isinstance(rate_fixing, NoInput)
            else rate_fixing,
            fixing_method=p.rate_params.fixing_method,
            method_param=p.rate_params.method_param,
            spread_compound_method=p.rate_params.spread_compound_method,
            float_spread=p.rate_params.float_spread,
            rate_series=rate_series_,
        )
        if isinstance(data, Err):
            return data
        else:
            (
                r_star,
                bounds_obs,
                bounds_dcf,
                dates_obs,
                dates_dcf,
                dcfs_obs,
                dcfs_dcf,
                fixing_rates,
                populated,
                unpopulated,
            ) = data.unwrap()

        if bounds_obs is None:
            bounds_obs, bounds_dcf, is_matching = _RFRRate._adjust_dates(
                start=p.period_params.start,
                end=p.period_params.end,
                fixing_method=p.rate_params.fixing_method,
                method_param=p.rate_params.method_param,
                fixing_calendar=p.rate_params.fixing_series.calendar,
            )

        if not isinstance(right, NoInput) and bounds_obs[0] > right:
            df = cls._make_dataframe(
                obs_dates=[],
                notional=[],
                risk=[],
                dcf=[],
                rates=[],
                name=rate_curve.id,
            )
            return Ok(df)

        if dates_obs is None:
            dates_obs, dates_dcf, dcfs_obs, dcfs_dcf, populated, unpopulated, fixing_rates = (
                _RFRRate._get_dates_and_fixing_rates_from_fixings(
                    rate_series=rate_series_,
                    bounds_obs=bounds_obs,
                    bounds_dcf=bounds_dcf,  # type: ignore[arg-type]  # prechecked
                    is_matching=False,
                    rate_fixings=p.rate_params.rate_fixing.identifier,
                )
            )

        if isna(fixing_rates).any():  # type: ignore[union-attr]
            _RFRRate._forecast_fixing_rates_from_curve(
                unpopulated=unpopulated,  # type: ignore[arg-type]
                populated=populated,  # type: ignore[arg-type]
                fixing_rates=fixing_rates,  # type: ignore[arg-type]
                rate_curve=rate_curve,
                dates_obs=dates_obs,
                dcfs_obs=dcfs_obs,  # type: ignore[arg-type]
            )

        d_star = p.period_params.dcf

        drdr = _FixingsExposureCalculator._drdr_rfr_approximation(
            p=p,
            rate_series=rate_series_,
            r_star=r_star,
            di=dcfs_dcf,  # type: ignore[arg-type]
            z=p.rate_params.float_spread,
            fixing_method=p.rate_params.fixing_method,
            method_param=p.rate_params.method_param,
            spread_compound_method=p.rate_params.spread_compound_method,
            fixing_rates=fixing_rates,  # type: ignore[arg-type]
        )

        dc_result = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_result, Err):
            return dc_result
        else:
            disc_curve_: _BaseCurve = dc_result.unwrap()

        dPdr = (
            -p.settlement_params.notional
            * disc_curve_[p.settlement_params.payment]
            * d_star
            * drdr
            / 10000.0
        )
        vi = np.array([disc_curve_[_] for _ in dates_obs[1:]])  # TODO: can approximate this
        ni = 1 / (dcfs_obs * vi) * dPdr * 10000.0
        fixing_notionals = Series(data=ni, index=fixing_rates.index)  # type: ignore[union-attr]

        df = fixing_notionals.astype(float).to_frame(name="notional")
        df["risk"] = dPdr
        df["dcf"] = dcfs_obs
        df["rates"] = fixing_rates

        if populated is None or len(populated) == 0:
            pass  # nothing to set to zero as being fixed.
        else:
            df.loc[populated.index, ["notional", "risk"]] = 0.0

        df.columns = MultiIndex.from_tuples(
            [
                (rate_curve.id, "notional"),
                (rate_curve.id, "risk"),
                (rate_curve.id, "dcf"),
                (rate_curve.id, "rates"),
            ]
        )
        df.index.name = "obs_dates"
        return Ok(_trim_df_by_index(df.astype(float), NoInput(0), right))

    @staticmethod
    def _drdr_rfr_approximation(
        p: FloatPeriod,
        rate_series: FloatRateSeries,
        r_star: DualTypes,
        di: Arr1dF64,
        z: DualTypes,
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        fixing_rates: Series[DualTypes],  # type: ignore[type-var]
    ) -> Arr1dObj:
        # approximate sensitivity to each fixing
        z = z / 100.0

        d = di.sum()
        if fixing_method in [
            FloatFixingMethod.RFRLockoutAverage,
            FloatFixingMethod.RFRObservationShiftAverage,
            FloatFixingMethod.RFRLookbackAverage,
            FloatFixingMethod.RFRPaymentDelayAverage,
        ]:
            drdri: Arr1dObj = di / d
        elif spread_compound_method == SpreadCompoundMethod.ISDACompounding:
            # this makes a number of approximations including reversing a compounded spread
            # with a simple formula
            r_bar, d_bar, n = average_rate(
                effective=p.period_params.start,
                termination=p.period_params.end,
                convention=rate_series.convention,
                rate=r_star - z,
                dcf=p.period_params.dcf,
            )
            drdri = di / (1 + di * (r_bar + z) / 100.0) * (r_star / 100.0 * d + 1) / d  # type: ignore[operator]
        # elif spread_compound_method == SpreadCompoundMethod.ISDAFlatCompounding:
        #     r_star = ((1 + d_bar * (r_bar + z) / 100.0) ** n - 1) * 100.0 / (n * d_bar)
        #     drdri1 = di / (1 + di * (r_bar + z) / 100.0) * ((r_star / 100.0 * d) + 1) / d
        #     drdri2 = di / (1 + di * r_bar / 100.0) * ((r_star0 / 100.0 * d) + 1) / d
        #     drdri = (drdri1 + drdri2) / 2.0
        else:  # spread_compound_method == SpreadCompoundMethod.NoneSimple:
            ri = fixing_rates.to_numpy()
            drdri = di / (1 + di * ri / 100.0) * ((r_star - z) / 100.0 * d + 1) / d

        if fixing_method in [FloatFixingMethod.RFRLockoutAverage, FloatFixingMethod.RFRLockout]:
            for i in range(method_param):
                drdri[-(method_param + 1)] += drdri[-(i + 1)]
                drdri[-(i + 1)] = 0.0

        return drdri

    ###############################

    @classmethod
    def ibor(
        cls,
        p: FloatPeriod,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve_,
        right: datetime_,
        risk: DualTypes_ = NoInput(0),
    ) -> Result[DataFrame]:
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
        if isinstance(rate_curve, dict):
            if p.period_params.stub:
                # then must perform an interpolated calculation
                return cls._ibor_stub(
                    p=p,
                    rate_fixing=p.rate_params.rate_fixing,  # type: ignore[arg-type]
                    rate_curve=rate_curve,
                    disc_curve=_validate_obj_not_no_input(disc_curve, "disc_curve"),
                    right=right,
                    risk=risk,
                )
            else:  # not self.stub:
                # then extract the one relevant curve from dict
                curve_: _BaseCurve = _get_ibor_curve_from_dict(
                    p.rate_params.fixing_frequency, rate_curve
                )
        elif isinstance(rate_curve, NoInput):
            raise ValueError(
                "`rate_curve` must be supplied as Curve or dict for IBOR fixings exposure."
            )
        else:
            curve_ = rate_curve

        return cls._ibor_single_tenor(
            p=p,
            rate_fixing=p.rate_params.rate_fixing,  # type: ignore[arg-type]  # pre-checked
            rate_curve=curve_,
            disc_curve=_disc_required_maybe_from_curve(curve_, disc_curve),
            right=right,
        )

    @classmethod
    def _ibor_single_tenor(
        cls,
        p: FloatPeriod,
        rate_fixing: IBORFixing,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        right: datetime_,
        risk: DualTypes_ = NoInput(0),
    ) -> Result[DataFrame]:
        reg_dcf = dcf(
            start=rate_fixing.accrual_start,
            end=rate_fixing.accrual_end,
            convention=rate_fixing.index.convention,
            frequency=rate_fixing.index.frequency,
            stub=False,
            calendar=rate_fixing.index.calendar,
        )

        dc_result = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_result, Err):
            return dc_result
        else:
            disc_curve_: _BaseCurve = dc_result.unwrap()

        if not isinstance(rate_fixing.value, NoInput):
            risk, notional = 0.0, 0.0  # then fixing is set so return zero exposure.
            _rate = _dual_float(rate_fixing.value)
        elif rate_fixing.date < disc_curve_.nodes.initial:
            _rate = np.nan
            risk, notional = 0.0, 0.0
            notional = 0.0
        else:
            risk = (
                (
                    -p.settlement_params.notional
                    * p.period_params.dcf
                    * disc_curve_[p.settlement_params.payment]
                )
                if isinstance(risk, NoInput)
                else risk
            )
            notional = _dual_float(risk / (reg_dcf * disc_curve_[rate_fixing.accrual_end]))
            risk = _dual_float(risk) * 0.0001  # scale to bp
            _rate = _dual_float(p.rate(rate_curve=rate_curve))

        df = cls._make_dataframe(
            obs_dates=[rate_fixing.date],
            notional=[notional],
            risk=[risk],
            dcf=[reg_dcf],
            rates=[_rate],
            name=rate_curve.id,
        )
        return Ok(_trim_df_by_index(df, NoInput(0), right))

    @classmethod
    def _ibor_stub(
        cls,
        p: FloatPeriod,
        rate_fixing: IBORStubFixing,
        rate_curve: dict[str, _BaseCurve],
        disc_curve: _BaseCurve,
        right: datetime_,
        risk: DualTypes_ = NoInput(0),
    ) -> Result[DataFrame]:
        dc_result = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(dc_result, Err):
            return dc_result
        else:
            disc_curve_: _BaseCurve = dc_result.unwrap()

        risk = (
            -p.settlement_params.notional
            * p.period_params.dcf
            * disc_curve_[p.settlement_params.payment]
            if isinstance(risk, NoInput)
            else risk
        )
        # get the relevant tenors
        tenors, ends = _find_neighbouring_tenors(
            end=rate_fixing.accrual_end,
            start=rate_fixing.accrual_start,
            tenors=[_ for _ in rate_curve if _.upper() != "RFR"],
            rate_series=rate_fixing.series,
        )
        w1 = (rate_fixing.accrual_end - ends[0]) / (ends[1] - ends[0])
        w0 = 1 - w1
        d0 = dcf(rate_fixing.accrual_start, ends[0], rate_fixing.series.convention)
        d1 = dcf(rate_fixing.accrual_start, ends[1], rate_fixing.series.convention)
        v0, v1 = disc_curve_[ends[0]], disc_curve_[ends[1]]

        df0 = cls._make_dataframe(
            [rate_fixing.date],
            [risk * w0 / (d0 * v0)],  # type: ignore[list-item]
            [0.0001 * risk * w0],  # type: ignore[list-item]
            [d0],
            [np.nan],
            rate_curve[tenors[0]].id,
        )
        df1 = cls._make_dataframe(
            [rate_fixing.date],
            [risk * w1 / (d1 * v1)],  # type: ignore[list-item]
            [0.0001 * risk * w1],  # type: ignore[list-item]
            [d1],
            [np.nan],
            rate_curve[tenors[1]].id,
        )

        df = concat([df0, df1], axis=1)
        return Ok(_trim_df_by_index(df.astype(float), NoInput(0), right))

    @staticmethod
    def _make_dataframe(
        obs_dates: list[datetime],
        notional: list[float],
        risk: list[float],
        dcf: list[float],
        rates: list[float],
        name: str,
    ) -> DataFrame:
        df = DataFrame(
            {
                "obs_dates": obs_dates,
                "notional": notional,
                "risk": risk,
                "dcf": dcf,
                "rates": rates,
            },
        ).set_index("obs_dates")

        df.columns = MultiIndex.from_tuples(
            [
                (name, "notional"),
                (name, "risk"),
                (name, "dcf"),
                (name, "rates"),
            ]
        )
        return df


def _get_ibor_curve_from_dict(fixing_frequency: Frequency, d: dict[str, _BaseCurve]) -> _BaseCurve:
    remapped = {k.upper(): v for k, v in d.items()}
    try:
        freq_str = _get_tenor_from_frequency(fixing_frequency)
        return remapped[freq_str]
    except KeyError:
        raise ValueError(
            "If supplying `curve` as dict must provide a tenor mapping key and curve for"
            f"the frequency of the given Period. The missing mapping is '{freq_str}'."
        )
