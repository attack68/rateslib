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
from rateslib.enums.generics import NoInput
from rateslib.scheduling import Adjuster, add_tenor

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        Arr2dObj,
        DualTypes,
        IRSSeries,
        OptionPricingModel,
        datetime_,
    )

UTC = timezone.utc


class _IRVolPricingParams(NamedTuple):
    """Container for parameters for pricing IR options."""

    vol: DualTypes  # vol appropriate for `pricing_model`
    k: DualTypes  # strike
    f: DualTypes  # forward
    shift: DualTypes  # shift to apply to `k` and `f` to use with `vol` in bps
    t_e: DualTypes  # time to expiry
    pricing_model: OptionPricingModel

    @property
    def rate_shift(self) -> DualTypes:
        return self.shift / 100.0


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


@dataclass(frozen=True)
class _IRCubeMeta:
    """
    An immutable container of meta data associated with a
    :class:`~rateslib.volatility._BaseIRCube` used to make calculations.
    """

    _eval_date: datetime
    _weights: Series[float] | NoInput
    _expiries: list[str | datetime]
    _tenors: list[str]
    _irs_series: IRSSeries
    _shift: DualTypes
    _indexes: list[Any]
    _smile_params: dict[str, Any]

    def __post_init__(self) -> None:
        for idx in range(1, len(self.expiries)):
            if self.expiry_dates[idx - 1] >= self.expiry_dates[idx]:
                raise ValueError("Cube `expiries` are not sorted or contain duplicates.\n")

    @property
    def shift(self) -> DualTypes:
        """
        The number of basis points used by any *Smile* when using 'Black Shifted Volatility'.
        """
        return self._shift

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
    def smile_params(self) -> dict[str, Any]:
        """
        A list of additional parameters used only by the specific *Cube* in constructing its
        individual *Smile* types.
        """
        return self._smile_params

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

    @property
    def indexes(self) -> list[Any]:
        """A list of the indexes used as strikes for the third dimension of the *Cube*."""
        return self._indexes

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

    if isinstance(payment_lag, int):
        payment_lag_: datetime | Adjuster = Adjuster.BusDaysLagSettle(payment_lag)
    elif isinstance(payment_lag, NoInput):
        payment_lag_ = irs_series_.settle
    else:
        payment_lag_ = payment_lag
    del payment_lag

    if isinstance(payment_lag_, datetime):
        payment_ = payment_lag_
    else:
        payment_ = payment_lag_.adjust(expiry_, irs_series_.calendar)

    return expiry_, payment_


def _get_ir_expiry(
    eval_date: datetime,
    irs_series: str | IRSSeries,
    expiry: datetime | str,
) -> datetime:
    """
    Determines the expiry of a Swaption possibly from string tenor.

    Parameters
    ----------
    eval_date: datetime
        The horizon or evaluation date, i.e. today.
    irs_series: IRSSeries, str
        The :class:`~rateslib.enums.parameters.IRSSeries` of the underlying IRS.
    expiry: str, datetime
        The expiry for the swaption.

    Returns
    -------
    datetime
    """
    if isinstance(expiry, datetime):
        return expiry

    irs_series_ = _get_irs_series(irs_series)
    del irs_series

    expiry_ = add_tenor(  # TODO: maybe adopt a Schedule here instead of add tenor
        start=eval_date,
        tenor=expiry,
        modifier=irs_series_.modifier,
        calendar=irs_series_.calendar,
        roll=eval_date.day,
        settlement=False,
        mod_days=False,
    )
    return expiry_


def _get_ir_tenor(
    expiry: datetime,
    irs_series: str | IRSSeries,
    tenor: str | datetime,
) -> datetime:
    """
    Determines the termination of an IRS associated with a Swaption expiry.

    Parameters
    ----------
    expiry: datetime
        The expiry date
    irs_series: IRSSeries, str
        The :class:`~rateslib.enums.parameters.IRSSeries` of the underlying IRS.
    tenor: str, datetime
        The tenor for the IRS

    Returns
    -------
    tuple of datetime
    """
    if isinstance(tenor, datetime):
        return tenor

    irs_series_ = _get_irs_series(irs_series)
    del irs_series

    effective = irs_series_.settle.adjust(expiry, irs_series_.calendar)
    tenor_ = add_tenor(  # TODO: maybe adopt a Schedule here instead of add tenor
        start=effective,
        tenor=tenor,
        modifier=irs_series_.modifier,
        calendar=irs_series_.calendar,
        roll=effective.day,
        settlement=False,
        mod_days=False,
    )
    return tenor_


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
