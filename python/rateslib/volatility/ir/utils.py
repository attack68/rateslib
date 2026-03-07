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
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from pandas import Series

from rateslib.data.fixings import IRSFixing, _get_irs_series
from rateslib.dual import set_order_convert
from rateslib.enums.generics import NoInput
from rateslib.scheduling import Adjuster, add_tenor
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64
from rateslib.splines.evaluate import evaluate

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        Arr2dObj,
        DualTypes,
        IRSSeries,
        Number,
        OptionPricingModel,
        datetime_,
    )

UTC = timezone.utc

SPLINE_LOWER = -5000.0
SPLINE_UPPER = 10000.0


class _IRVolPricingParams(NamedTuple):
    """Container for parameters for pricing IR options."""

    vol: DualTypes  # vol appropriate for `pricing_model`
    k: DualTypes  # strike
    f: DualTypes  # forward
    shift: DualTypes  # shift to apply to `k` and `f` to use with `vol`
    t_e: DualTypes  # time to expiry
    pricing_model: OptionPricingModel


class _IRSmileMeta:
    """
    A container of meta data associated with a :class:`~rateslib.volatility._BaseIRSmile`
    used to make calculations.
    """

    def __init__(
        self,
        _eval_date: datetime,
        _expiry_input: datetime | str,
        _tenor_input: datetime | str,
        _irs_series: IRSSeries,
        _shift: DualTypes,
        _plot_x_axis: str,
        _plot_y_axis: str,
        _pricing_model: OptionPricingModel,
    ):
        self._eval_date = _eval_date
        self._expiry_input = _expiry_input
        self._tenor_input = _tenor_input
        self._irs_series = _irs_series
        self._plot_x_axis = _plot_x_axis
        self._plot_y_axis = _plot_y_axis
        self._irs_fixing = IRSFixing(
            irs_series=self.irs_series,
            publication=self.expiry,
            tenor=self.tenor_input,
            value=NoInput(0),
            identifier=NoInput(0),
        )
        self._shift = _shift
        self._pricing_model = _pricing_model

    @property
    def pricing_model(self) -> OptionPricingModel:
        """The option pricing model associated with this *Smile* volatility output."""
        return self._pricing_model

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Smile*."""
        return self._eval_date

    @property
    def shift(self) -> DualTypes:
        """
        The number of basis points used by this *Smile* when using 'Black Shifted Volatility'.
        """
        return self._shift

    @cached_property
    def rate_shift(self) -> DualTypes:
        """
        The ``shift`` amount expressed in rate percentage terms.
        """
        return self.shift / 100.0

    @property
    def plot_x_axis(self) -> str:
        """The default ``x_axis`` parameter passed to
        :meth:`~rateslib.volatility._BaseIRSmile.plot`"""
        return self._plot_x_axis

    @property
    def plot_y_axis(self) -> str:
        """The default ``y_axis`` parameter passed to
        :meth:`~rateslib.volatility._BaseIRSmile.plot`"""
        return self._plot_y_axis

    @property
    def irs_series(self) -> IRSSeries:
        """The :class:`~rateslib.data.fixings.IRSSeries` of for the conventions of the *Smile*."""
        return self._irs_series

    @property
    def expiry_input(self) -> datetime | str:
        """Expiry input of the options priced by this *Smile*."""
        return self._expiry_input

    @cached_property
    def expiry(self) -> datetime:
        """Derived expiry date of the options priced by this *Smile*."""
        if isinstance(self.expiry_input, str):
            return add_tenor(
                start=self.eval_date,
                tenor=self.expiry_input,
                modifier=self.irs_series.modifier,
                calendar=self.irs_series.calendar,
            )
        else:
            return self.expiry_input

    @property
    def tenor_input(self) -> datetime | str:
        """Tenor input of the underlying IRS priced by this *Smile*."""
        return self._tenor_input

    @property
    def irs_fixing(self) -> IRSFixing:
        """The :class:`~rateslib.data.fixings.IRSFixing` underlying for the swaptions priced
        by this *Smile*."""
        return self._irs_fixing

    @cached_property
    def t_expiry(self) -> float:
        """Calendar days from eval to expiry divided by 365."""
        return (self.expiry - self.eval_date).days / 365.0

    def _t_expiry(self, expiry: datetime) -> float:
        """Calendar days from eval to specified expiry divided by 365."""
        return (expiry - self.eval_date).days / 365.0

    @cached_property
    def t_expiry_sqrt(self) -> float:
        """Square root of ``t_expiry``."""
        ret: float = self.t_expiry**0.5
        return ret


class _IRSplineSmileNodes:
    """
    A container for data relating to interpolating the `nodes` of a
    :class:`~rateslib.volatility.IRSplineSmile`.
    """

    _nodes: dict[float, DualTypes]
    _spline: _IRVolSpline

    def __init__(self, nodes: dict[float, DualTypes], k: int) -> None:
        self._nodes = dict(sorted(nodes.items()))

        match (self.n, k):
            case (1, _) | (2, _):
                # 1 DoF yields a flat smile, but treat it as a line of zero gradient
                # 2 DoF yields a straight line, usually with some non-zero gradient
                k = 2
                t = [SPLINE_LOWER, SPLINE_LOWER, SPLINE_UPPER, SPLINE_UPPER]
            case (_, 2):
                # more DoF but piecewise linear so can be extended
                t = [SPLINE_LOWER, SPLINE_LOWER] + self.keys + [SPLINE_UPPER, SPLINE_UPPER]
            case (_, 4):
                # more DoF but piecewise cubic so cannot
                t = [SPLINE_LOWER] * 4 + self.keys + [SPLINE_UPPER] * 4

        self._spline = _IRVolSpline(t=t, k=k)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _IRSplineSmileNodes):
            return False
        return self._nodes == other._nodes and self.k == other.k

    @property
    def nodes(self) -> dict[float, DualTypes]:
        """The initial nodes dict passed for construction of this class."""
        return self._nodes

    @cached_property
    def keys(self) -> list[float]:
        """A list of the relative strike keys in ``nodes``."""
        return list(self.nodes.keys())

    @cached_property
    def values(self) -> list[DualTypes]:
        """A list of the delta index values in ``nodes``."""
        return list(self.nodes.values())

    @property
    def n(self) -> int:
        """The number of pricing parameters in ``nodes``."""
        return len(self.keys)

    @property
    def k(self) -> int:
        """The order of the interpolating polynomial spline."""
        return self.spline.k

    @property
    def spline(self) -> _IRVolSpline:
        """An instance of :class:`~rateslib.volatility.ir._IRVolSpline`."""
        return self._spline


class _IRVolSpline:
    """
    A container for data relating to interpolating the `nodes` of
    a :class:`~rateslib.volatility.IRSplineSmile` using a PPSpline.
    """

    _k: int
    _t: list[float]
    _spline: PPSplineF64 | PPSplineDual | PPSplineDual2

    def __init__(self, t: list[float], k: int) -> None:
        self._t = t
        self._k = k
        self._spline = PPSplineF64(k, [0.0] * 5, None)  # placeholder: csolve will reengineer

    @property
    def t(self) -> list[float]:
        """The knot sequence of the PPSpline."""
        return self._t

    @property
    def k(self) -> int:
        """The order of the spline."""
        return self._k

    @property
    def spline(self) -> PPSplineF64 | PPSplineDual | PPSplineDual2:
        """An instance of :class:`~rateslib.splines.PPSplineF64`,
        :class:`~rateslib.splines.PPSplineDual` or :class:`~rateslib.splines.PPSplineDual2`"""
        return self._spline

    def evaluate(self, x: DualTypes, m: int = 0) -> Number:
        """Perform the :meth:`~rateslib.splines.evaluate` method on the object's ``spline``."""
        return evaluate(spline=self.spline, x=x, m=m)

    def _csolve_n_other(
        self, nodes: _IRSplineSmileNodes, ad: int
    ) -> tuple[list[float], list[DualTypes], int, int]:
        """
        Solve a spline with more than one node value.
        Premium adjusted delta types have an unbounded right side delta index so a derivative of
        0 is applied to the spline as a boundary condition.
        Premium unadjusted delta types have a right side delta index approximately equal to 1.0.
        Use a natural spline boundary condition here.
        """
        tau = nodes.keys.copy()
        y = nodes.values.copy()

        # left side constraint
        gradient = (y[1] - y[0]) / (tau[1] - tau[0])
        # project the gradient backwards to SPLINE_LOWER: this simulates an inner left side
        # 1st order gradient constraint whilst ensuring a wider domain.
        y.insert(0, (SPLINE_LOWER - tau[0]) * gradient + y[0])
        tau.insert(0, SPLINE_LOWER)
        if self.k == 4:
            # now insert the natural spline 2nd derivative constraint
            y.insert(0, set_order_convert(0.0, ad, None))
            tau.insert(0, SPLINE_LOWER)
            left_n = 2  # natural spline
        else:  # == 2
            left_n = 0

        # right side constraint
        gradient = (y[-1] - y[-2]) / (tau[-1] - tau[-2])
        # project the gradient forwards to SPLINE_UPPER: this simulates an inner left side
        # 1st order gradient constraint whilst ensuring a wider domain.
        y.append((SPLINE_UPPER - tau[-1]) * gradient + y[-1])
        tau.append(SPLINE_UPPER)
        if self.k == 4:
            tau.append(self.t[-1])
            y.append(set_order_convert(0.0, ad, None))
            right_n = 2  # natural spline
        else:  # == 2
            right_n = 0

        return tau, y, left_n, right_n

    def csolve(self, nodes: _IRSplineSmileNodes, ad: int) -> None:
        """
        Construct a spline of appropriate AD order and solve the spline coefficients for the
        given ``nodes``.

        Parameters
        ----------
        nodes: _IRSplineSmileNodes
            Required information for constructing a PPSpline.
        ad: int
            The AD order of the constructed PPSPline.

        Returns
        -------
        None
        """
        if ad == 0:
            Spline: type[PPSplineF64] | type[PPSplineDual] | type[PPSplineDual2] = PPSplineF64
        elif ad == 1:
            Spline = PPSplineDual
        else:
            Spline = PPSplineDual2

        if nodes.n == 1:
            # one node defines a flat line, all spline coefficients are the equivalent value.
            self._spline = Spline(self.k, self.t, nodes.values * self.k)  # type: ignore[arg-type]
        else:
            tau, y, left_n, right_n = self._csolve_n_other(nodes, ad)
            self._spline = Spline(self.k, self.t, None)
            self._spline.csolve(tau, y, left_n, right_n, False)  # type: ignore[arg-type]

    # def to_json(self) -> str:
    #     """
    #     Serialize this object to JSON format.
    #
    #     The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.
    #
    #     Returns
    #     -------
    #     str
    #     """
    #     obj = dict(
    #         PyNative=dict(
    #             _FXDeltaVolSpline=dict(
    #                 t=self.t,
    #             )
    #         )
    #     )
    #     return json.dumps(obj)
    #
    # @classmethod
    # def _from_json(cls, loaded_json: dict[str, Any]) -> _FXDeltaVolSpline:
    #     return _FXDeltaVolSpline(
    #         t=loaded_json["t"],
    #     )

    def __eq__(self, other: Any) -> bool:
        """CurveSplines are considered equal if their knot sequence and endpoints are equivalent.
        For the same nodes this will resolve to give the same spline coefficients.
        """
        if not isinstance(other, _IRVolSpline):
            return False
        else:
            return self.t == other.t and self.k == other.k


@dataclass(frozen=True)
class _IRSabrCubeMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.volatility.FXSabrSurface` used to make calculations.
    """

    _eval_date: datetime
    _weights: Series[float] | NoInput
    _expiries: list[str | datetime]
    _tenors: list[str]
    _irs_series: IRSSeries

    def __post_init__(self) -> None:
        for idx in range(1, len(self.expiries)):
            if self.expiry_dates[idx - 1] >= self.expiry_dates[idx]:
                raise ValueError("Cube `expiries` are not sorted or contain duplicates.\n")

    @property
    def _n_expiries(self) -> int:
        """The number of expiries."""
        return len(self._expiries)

    @property
    def _n_tenors(self) -> int:
        """The number of tenors."""
        return len(self._tenors)

    @property
    def irs_series(self) -> IRSSeries:
        """
        The :class:`~rateslib.data.fixings.IRSSeries` of the underlying
        :class:`~rateslib.instruments.IRS`
        """
        return self._irs_series

    @property
    def weights(self) -> Series[float] | NoInput:
        """Weights used for temporal volatility interpolation."""
        return self._weights

    @cached_property
    def weights_cum(self) -> Series[float] | NoInput:
        """Weight adjusted time to expiry (in calendar days) per date for temporal volatility
        interpolation."""
        if isinstance(self.weights, NoInput):
            return self.weights
        else:
            return self.weights.cumsum()

    @property
    def tenors(self) -> list[str]:
        """A list of the tenors as measured according the underlying from each expiry."""
        return self._tenors

    @cached_property
    def tenor_dates(self) -> Arr2dObj:
        """An array of *IRS* termination dates measured from each expiry's effective date."""
        arr = np.empty(shape=(self._n_expiries, self._n_tenors), dtype=object)
        for i, expiry in enumerate(self.expiry_dates):
            effective = self.irs_series.calendar.adjust(expiry, self.irs_series.settle)
            for j, tenor in enumerate(self.tenors):
                arr[i, j] = add_tenor(
                    start=effective,
                    tenor=tenor,
                    modifier=self.irs_series.modifier,
                    calendar=self.irs_series.calendar,
                )
        return arr

    @cached_property
    def tenor_dates_posix(self) -> Arr2dObj:
        """An array of *IRS* termination dates as unix timestamp."""
        return np.reshape(
            [_.replace(tzinfo=UTC).timestamp() for _ in self.tenor_dates.ravel()],
            (self._n_expiries, self._n_tenors),
        )

    def _t_expiry(self, expiry: datetime) -> float:
        """Calendar days from eval to specified expiry divided by 365."""
        return (expiry - self.eval_date).days / 365.0

    # @cached_property
    # def tenor_posix(self) -> list[float]:
    #     """A list of the tenors as posix timestamp."""
    #     return [_.replace(tzinfo=UTC).timestamp() for _ in self.tenor_dates]

    @property
    def expiries(self) -> list[datetime | str]:
        """A list of the expiries."""
        return self._expiries

    @cached_property
    def expiry_dates(self) -> list[datetime]:
        """A list of the expiries as datetime."""
        _: list[datetime] = []
        for date in self.expiries:
            if isinstance(date, str):
                _.append(
                    add_tenor(
                        start=self._eval_date,
                        tenor=date,
                        modifier=self.irs_series.modifier,
                        calendar=self.irs_series.calendar,
                    )
                )
            else:
                _.append(date)
        return _

    @cached_property
    def expiries_posix(self) -> list[float]:
        """A list of the unix timestamps of each date in ``expiries``."""
        return [_.replace(tzinfo=UTC).timestamp() for _ in self.expiry_dates]

    @cached_property
    def eval_posix(self) -> float:
        """The unix timestamp of the ``eval_date``."""
        return self.eval_date.replace(tzinfo=UTC).timestamp()

    @property
    def eval_date(self) -> datetime:
        """Evaluation date of the *Surface*."""
        return self._eval_date


def _get_ir_expiry_and_payment(
    eval_date: datetime_,
    expiry: str | datetime,
    irs_series: str | IRSSeries,
    payment_lag: int | datetime_,
) -> tuple[datetime, datetime]:
    """
    Determines the expiry and payment date of an IR option using the following rules.

    Parameters
    ----------
    eval_date: datetime
        The evaluation date, which is today (if required)
    expiry: str, datetime
        The expiry date
    irs_series: IRSSeries, str
        The :class:`~rateslib.enums.parameters.IRSSeries` of the underlying IRS.
    payment_lag: Adjuster, int, datetime
        Number of business days to lag payment by after expiry.

    Returns
    -------
    tuple of datetime
    """
    irs_series_ = _get_irs_series(irs_series)
    del irs_series

    if isinstance(payment_lag, int):
        payment_lag_: datetime | Adjuster = Adjuster.BusDaysLagSettle(payment_lag)
    elif isinstance(payment_lag, NoInput):
        payment_lag_ = irs_series_.settle
    else:
        payment_lag_ = payment_lag
    del payment_lag

    if isinstance(expiry, str):
        # then use the objects to derive the expiry

        if isinstance(eval_date, NoInput):
            raise ValueError("`expiry` as string tenor requires `eval_date`.")
        # then the expiry will be implied
        expiry_ = add_tenor(
            start=eval_date,
            tenor=expiry,
            modifier=irs_series_.modifier,
            calendar=irs_series_.calendar,
            roll=eval_date.day,
            settlement=False,
            mod_days=False,
        )
    else:
        expiry_ = expiry

    if isinstance(payment_lag_, datetime):
        payment_ = payment_lag_
    else:
        payment_ = payment_lag_.adjust(expiry_, irs_series_.calendar)

    return expiry_, payment_


def _bilinear_interp(
    tl: DualTypes,
    tr: DualTypes,
    bl: DualTypes,
    br: DualTypes,
    h: tuple[float, float],
    v: tuple[float, float],
) -> DualTypes:
    """
    tl, tr, bl, br: the values on the vertices of a unit square.
    h: the progression along the horizontal top edge and the horizontal bottom edge in [0,1].
    v: the progression along the vertical left edge and the vertical right edge in [0,1].
    p: the interior point as the intersection when lines are drawn between the progression on edges.
    """
    return (
        tl * (1 - h[0]) * (1 - v[0])
        + tr * (h[0]) * (1 - v[1])
        + bl * (1 - h[1]) * v[0]
        + br * h[1] * v[1]
    )
