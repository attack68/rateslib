# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################


from __future__ import annotations  # type hinting

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypeAlias

from pandas import Series

from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_log,
    dual_norm_cdf,
    dual_norm_pdf,
)
from rateslib.dual.utils import _to_number
from rateslib.enums.generics import (
    NoInput,
)
from rateslib.rs import _sabr_x0 as _rs_sabr_x0
from rateslib.rs import _sabr_x1 as _rs_sabr_x1
from rateslib.rs import _sabr_x2 as _rs_sabr_x2
from rateslib.rs import index_left_f64
from rateslib.scheduling import get_calendar

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Number,
    )

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure

TERMINAL_DATE = datetime(2100, 1, 1)
UTC = timezone.utc


@dataclass(frozen=True)
class _SabrSmileNodes:
    """
    A container for data relating to the SABR parameters of a
    :class:`~rateslib.volatility.FXSabrSmile` and :class:`~rateslib.volatility.IRSabrSmile`.
    """

    _alpha: Number
    _beta: float | Variable
    _rho: Number
    _nu: Number

    @property
    def alpha(self) -> Number:
        """The :math:`\\alpha` parameter of the SABR function."""
        return self._alpha

    @property
    def beta(self) -> float | Variable:
        """The :math:`\\beta` parameter of the SABR function."""
        return self._beta

    @property
    def rho(self) -> Number:
        """The :math:`\\rho` parameter of the SABR function."""
        return self._rho

    @property
    def nu(self) -> Number:
        """The :math:`\\nu` parameter of the SABR function."""
        return self._nu

    @property
    def n(self) -> int:
        return 4


def _validate_weights(
    weights: Series[float] | NoInput,
    eval_date: datetime,
    expiries: list[datetime],
) -> Series[float] | None:
    if isinstance(weights, NoInput):
        return None

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
    w[eval_date] = 0.0  # type: ignore[call-overload]
    return w


def _t_var_interp(
    expiries: list[datetime],
    expiries_posix: list[float],
    expiry: datetime,
    expiry_posix: float,
    expiry_index: int,
    expiry_next_index: int,
    eval_posix: float,
    weights_cum: Series[float] | None,
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
    expiry_index: int
        The integer index of the expiries period in which the expiry falls.
    expiry_next_index: int
        Will be expiry_index + 1, unless the surface only has one expiry, in which case it will
        equal the expiry_index.
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
        expiry_next_index,
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
    expiry_next_index: int,
    eval_posix: float,
    weights_cum: Series[float] | None,
    vol1: DualTypes,
    dvol1_dk: DualTypes,
    vol2: DualTypes,
    dvol2_dk: DualTypes,
    bounds_flag: int,
    derivative: bool,
) -> tuple[DualTypes, DualTypes | None]:
    if weights_cum is None:  # weights must also be NoInput
        if bounds_flag == 0:
            t1 = expiries_posix[expiry_index] - eval_posix
            t2 = expiries_posix[expiry_next_index] - eval_posix
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = expiries_posix[expiry_index] - eval_posix
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = expiries_posix[expiry_next_index] - eval_posix
            t2 = TERMINAL_DATE.replace(tzinfo=UTC).timestamp() - eval_posix

        t_hat = expiry_posix - eval_posix
        t = expiry_posix - eval_posix
    else:
        if bounds_flag == 0:
            t1 = weights_cum[expiries[expiry_index]]
            t2 = weights_cum[expiries[expiry_next_index]]
        elif bounds_flag == -1:
            # left side extrapolation
            t1 = 0.0
            t2 = weights_cum[expiries[expiry_index]]
        else:  # bounds_flag == 1:
            # right side extrapolation
            t1 = weights_cum[expiries[expiry_next_index]]
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


class _OptionModelBlack76:
    """Container for option pricing formulae relating to the lognormal Black-76 model."""

    @staticmethod
    def _d_plus_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
        # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
        return dual_log(f / K) / vol_sqrt_t + eta * vol_sqrt_t

    @staticmethod
    def _d_plus_min_u(u: DualTypes, vol_sqrt_t: DualTypes, eta: float) -> DualTypes:
        # AD preserving calculation of d_plus in Black-76 formula  (eta should +/- 0.5)
        return -dual_log(u) / vol_sqrt_t + eta * vol_sqrt_t

    @staticmethod
    def _d_min(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
        return _OptionModelBlack76._d_plus_min(K, f, vol_sqrt_t, -0.5)

    @staticmethod
    def _d_plus(K: DualTypes, f: DualTypes, vol_sqrt_t: DualTypes) -> DualTypes:
        return _OptionModelBlack76._d_plus_min(K, f, vol_sqrt_t, +0.5)

    @staticmethod
    def _value(
        F: DualTypes,
        K: DualTypes,
        t_e: float,
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
        v2: float, Dual, Dual2
            The discounting rate to delivery (ccy2 on FX options), at the appropriate collateral
            rate.
        vol: float, Dual, Dual2
            The volatility measured over the period until expiry.
        phi: float
            Whether to calculate for call (1.0) or put (-1.0).

        Returns
        --------
        float, Dual, Dual2
        """
        vs = vol * t_e**0.5
        d1 = _OptionModelBlack76._d_plus(K, F, vs)
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


class _OptionModelBachelier:
    """Container for option pricing formulae relating to the lognormal Black-76 model."""

    @staticmethod
    def _value(
        F: DualTypes,
        K: DualTypes,
        t_e: float,
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
        v2: float, Dual, Dual2
            The discounting rate to delivery (ccy2 on FX options), at the appropriate collateral
            rate.
        vol: float, Dual, Dual2
            The volatility measured over the period until expiry.
        phi: float
            Whether to calculate for call (1.0) or put (-1.0).

        Returns
        --------
        float, Dual, Dual2
        """
        vs = vol * t_e**0.5
        d = (F - K) / vs

        P = dual_norm_cdf(phi * d)
        p = dual_norm_pdf(d)

        _: DualTypes = phi * (F - K) * P + vs * p
        return _ * v2


class _SabrModel:
    """Container for formulae relating to the SABR volatility model."""

    @staticmethod
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
        X0, dX0 = _SabrModel._sabr_X0(k, f, t, a, b_, p, v, derivative)
        X1, dX1 = _SabrModel._sabr_X1(k, f, t, a, b_, p, v, derivative)
        X2, dX2 = _SabrModel._sabr_X2(k, f, t, a, b_, p, v, derivative)

        if derivative == 0:
            return X0 * X1 * X2, None
        else:
            return X0 * X1 * X2, dX0 * X1 * X2 + X0 * dX1 * X2 + X0 * X1 * dX2  # type: ignore[operator]

    @staticmethod
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

    @staticmethod
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
        """  # noqa: E501
        return _rs_sabr_x1(k, f, t, a, b, p, v, derivative)

    @staticmethod
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


def _surface_index_left(expiries_posix: list[float], expiry_posix: float) -> tuple[int, int]:
    """use `index_left_f64` to derive left and right index,
    but exclude surfaces with only one expiry."""
    if len(expiries_posix) == 1:
        return 0, 0
    else:
        e_idx = index_left_f64(expiries_posix, expiry_posix)
        e_next_idx = e_idx + 1
        return e_idx, e_next_idx
