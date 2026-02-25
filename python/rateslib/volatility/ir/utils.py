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
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from pandas import Series
from pytz import UTC

from rateslib.data.fixings import IRSFixing, _get_irs_series
from rateslib.dual import (
    Variable,
)
from rateslib.enums.generics import NoInput
from rateslib.scheduling import Adjuster, add_tenor

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Arr2dObj,
        DualTypes,
        IRSSeries,
        Number,
        datetime_,
    )


class _IRVolPricingParams(NamedTuple):
    vol: DualTypes  # Black Shifted Vol
    k: DualTypes  # Strike
    f: DualTypes  # Forward
    shift: DualTypes  # Shift to apply to `k` and `f` to use with `vol`


class _IRSmileMeta:
    def __init__(
        self,
        _eval_date: datetime,
        _expiry_input: datetime | str,
        _tenor_input: datetime | str,
        _irs_series: IRSSeries,
        _shift: DualTypes,
        _plot_x_axis: str,
    ):
        self._eval_date = _eval_date
        self._expiry_input = _expiry_input
        self._tenor_input = _tenor_input
        self._irs_series = _irs_series
        self._plot_x_axis = _plot_x_axis
        self._irs_fixing = IRSFixing(
            irs_series=self.irs_series,
            publication=self.expiry,
            tenor=self.tenor_input,
            value=NoInput(0),
            identifier=NoInput(0),
        )
        self._shift = _shift

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

    @cached_property
    def t_expiry_sqrt(self) -> float:
        """Square root of ``t_expiry``."""
        ret: float = self.t_expiry**0.5
        return ret


@dataclass(frozen=True)
class _IRSabrSmileNodes:
    """
    A container for data relating to the SABR parameters of a
    :class:`~rateslib.ir_volatility.IRSabrSmile`.
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


@dataclass(frozen=True)
class _IRSabrCubeMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.fx_volatility.FXSabrSurface` used to make calculations.
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
            shape=(self._n_expiries, self._n_tenors),
        )

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
        effective = self.irs_series.calendar.adjust(self._eval_date, self.irs_series.settle)
        for date in self.expiries:
            if isinstance(date, str):
                _.append(
                    add_tenor(
                        start=effective,
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
