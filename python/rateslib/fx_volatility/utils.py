from __future__ import annotations  # type hinting

from datetime import datetime, timedelta
from datetime import datetime as dt
from typing import TYPE_CHECKING, TypeAlias
from dataclasses import dataclass

from pandas import Series
from pytz import UTC

from rateslib.calendars import get_calendar
from rateslib.default import (
    NoInput,
)
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
)
from rateslib.dual.utils import _to_number
from rateslib.rs import _sabr_x0 as _rs_sabr_x0
from rateslib.rs import _sabr_x1 as _rs_sabr_x1
from rateslib.rs import _sabr_x2 as _rs_sabr_x2

if TYPE_CHECKING:
    from rateslib.typing import Number

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure

TERMINAL_DATE = dt(2100, 1, 1)


@dataclass(frozen=True)
class _FXDeltaVolSmileMeta:
    delta_type: str
    eval_date: datetime
    expiry: datetime
    plot_x_axis: str


@dataclass(frozen=True)
class _FXDeltaVolSurfaceMeta:
    delta_type: str
    plot_x_axis: str


@dataclass(frozen=True)
class _FXSabrSmileMeta:
    delta_type: str
    plot_x_axis: str


def _validate_delta_type(delta_type: str) -> str:
    if delta_type.lower() not in ["spot", "spot_pa", "forward", "forward_pa"]:
        raise ValueError("`delta_type` must be in {'spot', 'spot_pa', 'forward', 'forward_pa'}.")
    return delta_type.lower()


def _validate_weights(
    weights: Series[float] | NoInput,
    eval_date: datetime,
    expiries: list[datetime],
) -> Series[float] | NoInput:
    if isinstance(weights, NoInput):
        return weights

    w: Series[float] = Series(
        1.0, index=get_calendar("all").cal_date_range(eval_date, TERMINAL_DATE)
    )
    w.update(weights)
    # restrict to sorted and filtered for outliers
    w = w.sort_index()
    w = w[eval_date:]  # type: ignore[misc]

    node_points: list[datetime] = [eval_date] + expiries + [TERMINAL_DATE]
    for i in range(len(expiries) + 1):
        s, e = node_points[i] + timedelta(days=1), node_points[i + 1]
        days = (e - s).days + 1
        w[s:e] = (  # type: ignore[misc]
            w[s:e] * days / w[s:e].sum()  # type: ignore[misc]
        )  # scale the weights to allocate the correct time between nodes.
    w[eval_date] = 0.0
    return w


def _t_var_interp(
    expiries: list[datetime],
    expiries_posix: list[float],
    expiry: datetime,
    expiry_posix: float,
    expiry_index: int,
    eval_posix: float,
    weights_cum: Series[float] | NoInput,
    vol1: DualTypes,
    vol2: DualTypes,
    bounds_flag: int,
) -> DualTypes:
    """
    Return the volatility of an intermediate timestamp via total linear variance interpolation.
    Possibly scaled by time weights if weights is available.

    Parameters
    ----------
    expiry_index: int
        The index defining the interval within which expiry falls.
    expiries_posix: list[datetime]
        The list of datetimes associated with the expiries of the *Surface*.
    expiries_posix: list[float]
        The list of posix timestamps associated with the expiries of the *Surface*.
    expiry: datetime
        The target expiry to be interpolated.
    expiry_posix: float
        The pre-calculated posix timestamp for expiry.
    eval_posix: float
         The pre-calculated posix timestamp for eval date of the *Surface*
    weights_cum: Series[float] or NoInput
         The cumulative sum of the weights indexes by date
    vol1: float, Dual, DUal2
        The volatility of the left side
    vol2: float, Dual, Dual2
        The volatility on the right side
    bounds_flag: int
        -1: left side extrapolation, 0: normal interpolation, 1: right side extrapolation

    Notes
    -----
    This function performs different interpolation if weights are given or not. ``bounds_flag``
    is used to parse the inputs when *Smiles* to the left and/or right are not available.
    """
    return _t_var_interp_d_sabr_d_k_or_f(
        expiries,
        expiries_posix,
        expiry,
        expiry_posix,
        expiry_index,
        eval_posix,
        weights_cum,
        vol1,
        dvol1_dk=0.0,
        vol2=vol2,
        dvol2_dk=0.0,
        bounds_flag=bounds_flag,
        derivative=False,
    )[0]


def _t_var_interp_d_sabr_d_k_or_f(
    expiries: list[datetime],
    expiries_posix: list[float],
    expiry: datetime,
    expiry_posix: float,
    expiry_index: int,
    eval_posix: float,
    weights_cum: Series[float] | NoInput,
    vol1: DualTypes,
    dvol1_dk: DualTypes,
    vol2: DualTypes,
    dvol2_dk: DualTypes,
    bounds_flag: int,
    derivative: bool,
) -> tuple[DualTypes, DualTypes | None]:
    if isinstance(weights_cum, NoInput):  # weights must also be NoInput
        if bounds_flag == 0:
            t1 = expiries_posix[expiry_index] - eval_posix
            t2 = expiries_posix[expiry_index + 1] - eval_posix
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = expiries_posix[expiry_index] - eval_posix
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = expiries_posix[expiry_index + 1] - eval_posix
            t2 = TERMINAL_DATE.replace(tzinfo=UTC).timestamp() - eval_posix

        t_hat = expiry_posix - eval_posix
        t = expiry_posix - eval_posix
    else:
        if bounds_flag == 0:
            t1 = weights_cum[expiries[expiry_index]]
            t2 = weights_cum[expiries[expiry_index + 1]]
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = weights_cum[expiries[expiry_index]]
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = weights_cum[expiries[expiry_index + 1]]
            t2 = weights_cum[TERMINAL_DATE]

        t_hat = weights_cum[expiry]  # number of vol weighted calendar days
        t = (expiry_posix - eval_posix) / 86400.0  # number of calendar days

    t_quotient = (t_hat - t1) / (t2 - t1)
    vol = ((t1 * vol1**2 + t_quotient * (t2 * vol2**2 - t1 * vol1**2)) / t) ** 0.5
    if derivative:
        dvol_dk = (
            (t2 / t) * t_quotient * vol2 * dvol2_dk + (t1 / t) * (1 - t_quotient) * vol1 * dvol1_dk
        ) / vol
    else:
        dvol_dk = None
    return vol, dvol_dk


def _black76(
    F: DualTypes,
    K: DualTypes,
    t_e: float,
    v1: NoInput,
    v2: DualTypes,
    vol: DualTypes,
    phi: float,
) -> DualTypes:
    """
    Option price in points terms for immediate premium settlement.

    Parameters
    -----------
    F: float, Dual, Dual2
        The forward price for settlement at the delivery date.
    K: float, Dual, Dual2
        The strike price of the option.
    t_e: float
        The annualised time to expiry.
    v1: float
        Not used. The discounting rate on ccy1 side.
    v2: float, Dual, Dual2
        The discounting rate to delivery on ccy2, at the appropriate collateral rate.
    vol: float, Dual, Dual2
        The volatility measured over the period until expiry.
    phi: float
        Whether to calculate for call (1.0) or put (-1.0).

    Returns
    --------
    float, Dual, Dual2
    """
    vs = vol * t_e**0.5
    d1 = _d_plus(K, F, vs)
    d2 = d1 - vs
    Nd1, Nd2 = dual_norm_cdf(phi * d1), dual_norm_cdf(phi * d2)
    _: DualTypes = phi * (F * Nd1 - K * Nd2)
    # Spot formulation instead of F (Garman Kohlhagen formulation)
    # https://quant.stackexchange.com/a/63661/29443
    # r1, r2 = dual_log(df1) / -t, dual_log(df2) / -t
    # S_imm = F * df2 / df1
    # d1 = (dual_log(S_imm / K) + (r2 - r1 + 0.5 * vol ** 2) * t) / vs
    # d2 = d1 - vs
    # Nd1, Nd2 = dual_norm_cdf(d1), dual_norm_cdf(d2)
    # _ = df1 * S_imm * Nd1 - K * df2 * Nd2
    return _ * v2


def _d_plus_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
    # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
    return dual_log(f / K) / vol_sqrt_t + eta * vol_sqrt_t


def _d_plus_min_u(u: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
    # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
    return -dual_log(u) / vol_sqrt_t + eta * vol_sqrt_t


def _d_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
    return _d_plus_min(K, f, vol_sqrt_t, -0.5)


def _d_plus(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
    return _d_plus_min(K, f, vol_sqrt_t, +0.5)


def _delta_type_constants(
    delta_type: str, w: DualTypes | NoInput, u: DualTypes | NoInput
) -> tuple[float, DualTypes, DualTypes]:
    """
    Get the values: (eta, z_w, z_u) for the type of expressed delta

    w: should be input as w_deli / w_spot
    u: should be input as K / f_d
    """
    if delta_type == "forward":
        return 0.5, 1.0, 1.0
    elif delta_type == "spot":
        return 0.5, w, 1.0  # type: ignore[return-value]
    elif delta_type == "forward_pa":
        return -0.5, 1.0, u  # type: ignore[return-value]
    else:  # "spot_pa"
        return -0.5, w, u  # type: ignore[return-value]


def _moneyness_from_atm_delta_closed_form(vol: DualTypes, t_e: DualTypes) -> DualTypes:
    """
    Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

    This function preserves AD.

    Book2: section "Strike and Volatility implied from ATM delta" (FXDeltaVolSMile)

    Parameters
    -----------
    vol: float, Dual, Dual2
        The volatility (in %, e.g. 10.0) to use in calculations.
    t_e: float,
        The time to expiry.

    Returns
    -------
    float, Dual or Dual2
    """
    return dual_exp((vol / 100.0) ** 2 * t_e / 2.0)


def _moneyness_from_delta_closed_form(
    delta: DualTypes,
    vol: DualTypes,
    t_e: DualTypes,
    z_w_0: DualTypes,
    phi: float,
) -> DualTypes:
    """
    Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

    This function preserves AD.

    Book2: section "Strike and Volatility implied from a given option's delta" (FXDeltaVolSmile)

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
    phi: float
        1.0 if is call, -1.0 if is put.

    Returns
    -------
    float, Dual or Dual2
    """
    vol_sqrt_t = vol * t_e**0.5 / 100.0
    _: DualTypes = dual_inv_norm_cdf(phi * delta / z_w_0)
    _ = dual_exp(vol_sqrt_t * (0.5 * vol_sqrt_t - phi * _))
    return _


def _d_sabr_d_k_or_f(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: float | Variable,
    p: Number,
    v: Number,
    derivative: int,
) -> tuple[Number, Number | None]:
    """
    Calculate the SABR function and its derivative with respect to k or f.

    For formula see for example I. Clark "Foreign Exchange Option
    Pricing" section 3.10.

    Rateslib uses the representation sigma(k) = X0 * X1 * X2, with these variables as defined in
    "Coding Interest Rates" chapter 13 to handle AD using dual numbers effectively.

    For no derivative and just the SABR function value use 0.
    For derivatives with respect to `k` use 1.
    For derivatives with respect to `f` use 2.

    See "Coding Interest Rates: FX Swaps and Bonds edition 2"
    """
    b_: Number = _to_number(b)
    X0, dX0 = _sabr_X0(k, f, t, a, b_, p, v, derivative)
    X1, dX1 = _sabr_X1(k, f, t, a, b_, p, v, derivative)
    X2, dX2 = _sabr_X2(k, f, t, a, b_, p, v, derivative)

    if derivative == 0:
        return X0 * X1 * X2, None
    else:
        return X0 * X1 * X2, dX0 * X1 * X2 + X0 * dX1 * X2 + X0 * X1 * dX2  # type: ignore[operator]


def _sabr_X0(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X0 = a / ((fk)^((1-b)/2) * (1 + (1-b)^2/24 ln^2(f/k) + (1-b)^4/1920 ln^4(f/k) )

    If ``derivative`` is 1 also returns dX0/dk, derived using sympy auto code generator.
    If ``derivative`` is 2 also returns dX0/df, derived using sympy auto code generator.
    """
    return _rs_sabr_x0(k, f, t, a, b, p, v, derivative)


def _sabr_X1(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X1 = 1 + t ( (1-b)^2 / 24 * a^2 / (fk)^(1-b) + 1/4 p b v a / (fk)^((1-b)/2) + (2-3p^2)/24 v^2 )

    If ``derivative`` also returns dX0/dk, calculated using sympy.
    """
    return _rs_sabr_x1(k, f, t, a, b, p, v, derivative)


def _sabr_X2(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: int = 0,
) -> tuple[Number, Number | None]:
    """
    X2 = z / chi(z)

    z = v / a * (fk) ^((1-b)/2) * ln(f/k)
    chi(z) = ln( (sqrt(1-2pz+z^2) + z -p) / (1-p) )

    If ``derivative`` = 1 also returns dX2/dk, calculated using sympy.
    If ``derivative`` = 2 also returns dX2/df, calculated using sympy.
    """
    return _rs_sabr_x2(k, f, t, a, b, p, v, derivative)
