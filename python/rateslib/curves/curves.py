from __future__ import annotations

import json
from abc import ABC
from calendar import monthrange
from dataclasses import replace
from datetime import datetime, timedelta
from math import prod
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import uuid4

from pandas import Series

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.calendars.rs import get_calendar
from rateslib.curves.base import _BaseCurve, _CurveMutation
from rateslib.curves.utils import (
    _CurveInterpolator,
    _CurveMeta,
    _CurveNodes,
    _CurveType,
    _ProxyCurveInterpolator,
    average_rate,
)
from rateslib.default import (
    NoInput,
    _drb,
)
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
)
from rateslib.dual.utils import _dual_float, _get_order_of
from rateslib.mutability import (
    _clear_cache_post,
    _no_interior_validation,
    _validate_states,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Arr1dF64,
        Arr1dObj,
        CalInput,
        CurveOption_,
        FXForwards,
        Number,
        datetime_,
        str_,
    )
DualTypes: TypeAlias = (
    "Dual | Dual2 | Variable | float"  # required for non-cyclic import on _WithCache
)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class _WithOperations(ABC):
    _base_type: _CurveType
    _nodes: _CurveNodes
    _meta: _CurveMeta
    _interpolator: _CurveInterpolator
    _ad: int

    # Operations

    @_validate_states
    def shift(
        self,
        spread: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
    ) -> _ShiftedCurve:
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        This curve adjustment preserves the shape of the curve but moves it up or
        down as a translation.
        This method is suitable as a way to assess value changes of instruments when
        a parallel move higher or lower in yields is predicted.

        Parameters
        ----------
        spread : float, Dual, Dual2, Variable
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.

        Returns
        -------
        _ShiftedCurve

        Notes
        -----
        The output :class:`~rateslib.curves.CompositeCurve` will have an AD order of the maximum
        of the AD order of the input ``spread`` and of `Self`.
        That is, if the input ``spread`` is
        a *float* (with AD order 0) and the input *Curve* is parametrised with *Dual* and has an AD
        order of 1, then the result will have an AD order of 1.

        .. warning::

           If ``spread`` is given as :class:`~rateslib.dual.Dual2` but the AD order of `Self`
           is only 1, then `Self` will be upcast to use :class:`~rateslib.dual.Dual2` types.

        Examples
        --------
        .. ipython:: python

           from rateslib.curves import Curve

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           shifted_curve = curve.shift(25)
           curve.plot("1d", comparators=[shifted_curve], labels=["orig", "shift"])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           spread_curve = curve.shift(25)
           fig, ax, line = curve.plot("1d", comparators=[spread_curve], labels=["orig", "shift"])
           plt.show()

        """
        _: _BaseCurve = self  # type: ignore[assignment]
        return _ShiftedCurve(curve=_, shift=spread, id=id)

    @_validate_states
    def translate(self, start: datetime, id: str_ = NoInput(0)) -> _TranslatedCurve:  # noqa: A002
        """
        Create a new curve with an initial node date moved forward keeping all else
        constant.

        This curve adjustment preserves forward curve expectations as time evolves.
        This method is suitable as a way to create a subsequent *opening* curve from a
        previous day's *closing* curve.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve, must be in the domain:
            (node_date[0], node_date[1]]
        t : bool
            Set to *True* if the initial knots of the knot sequence should be
            translated forward.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = curve.translate(dt(2022, 12, 1))
           curve.plot(
               "1d",
               comparators=[translated_curve],
               labels=["orig", "translated"],
               left=dt(2022, 12, 1),
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
               interpolation="log_linear",
           )
           translated_curve = curve.translate(dt(2022, 12, 1))
           fig, ax, line = curve.plot("1d", comparators=[translated_curve], labels=["orig", "translated"], left=dt(2022, 12, 1))
           plt.show()
           plt.close()

        .. ipython:: python

           curve.nodes
           translated_curve.nodes

        When a curve has a log-cubic spline the knot dates can be preserved or
        translated with the ``t`` argument. Preserving the knot dates preserves the
        interpolation of the curve. A knot sequence for a mixed curve which begins
        after ``start`` will not be affected in either case.

        .. ipython:: python

           curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2022, 2, 1): 0.999,
                   dt(2022, 3, 1): 0.9978,
                   dt(2022, 4, 1): 0.9963,
                   dt(2022, 5, 1): 0.9940
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated_curve = curve.translate(dt(2022, 1, 15))
           translated_curve2 = curve.translate(dt(2022, 1, 15), t=True)
           curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated_curve, translated_curve2], labels=["orig", "translated", "translated2"])

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2022, 2, 1): 0.999,
                   dt(2022, 3, 1): 0.9978,
                   dt(2022, 4, 1): 0.9963,
                   dt(2022, 5, 1): 0.9940
               },
               t = [dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
                    dt(2022, 2, 1), dt(2022, 3, 1), dt(2022, 4, 1),
                    dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1)]
           )
           translated = curve.translate(dt(2022, 1, 15))
           translated2 = curve.translate(dt(2022, 1, 15), t=True)
           fig, ax, line = curve.plot("1d", left=dt(2022, 1, 15), comparators=[translated, translated2], labels=["orig", "translated", "translated2"])
           plt.show()

        """  # noqa: E501
        _: _BaseCurve = self  # type: ignore[assignment]
        return _TranslatedCurve(curve=_, start=start, id=id)

    @_validate_states
    def roll(self, tenor: datetime | str) -> _RolledCurve:
        """
        Create a new curve with its shape translated in time but an identical initial
        node date.

        This curve adjustment is a simulation of a future state of the market where
        forward rates are assumed to have moved so that the present day's curve shape
        is reflected in the future (or the past). This is often used in trade
        strategy analysis.

        Parameters
        ----------
        tenor : datetime or str
            The date or tenor by which to roll the curve. If a tenor, as str, will
            derive the datetime as measured from the initial node date. If supplying a
            negative tenor, or a past datetime, there is a limit to how far back the
            curve can be rolled - it will first roll backwards and then attempt to
            :meth:`translate` forward to maintain the initial node date.

        Returns
        -------
        Curve

        Examples
        ---------
        The basic use of this function translates a curve forward in time and the
        plot demonstrates its rates are exactly the same as initially forecast.

        .. ipython:: python

           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = curve.roll("6m")
           rolled_curve2 = curve.roll("-6m")
           curve.plot(
               "1d",
               comparators=[rolled_curve, rolled_curve2],
               labels=["orig", "rolled", "rolled2"],
               right=dt(2026, 6, 30),
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           curve = Curve(
               nodes = {
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.988,
                   dt(2024, 1, 1): 0.975,
                   dt(2025, 1, 1): 0.965,
                   dt(2026, 1, 1): 0.955,
                   dt(2027, 1, 1): 0.9475
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = curve.roll("6m")
           rolled_curve2 = curve.roll("-6m")
           fig, ax, line = curve.plot("1d", comparators=[rolled_curve, rolled_curve2], labels=["orig", "rolled", "rolled2"], right=dt(2026,6,30))
           plt.show()

        """  # noqa: E501
        if isinstance(tenor, str):
            tenor_ = add_tenor(self._nodes.initial, tenor, "NONE", NoInput(0))
        else:
            tenor_ = tenor
        roll_days = (tenor_ - self._nodes.initial).days
        _: _BaseCurve = self  # type: ignore[assignment]
        return _RolledCurve(curve=_, roll_days=roll_days)


class _ShiftedCurve(_WithOperations, _BaseCurve):
    """A class which wraps a :class:`~rateslib.curves.CompositeCurve` designed to produce the
    required vertical basis points shift of the underlying ``curve``, according to *rateslib's*
    vector space metric.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    shift: float | Variable
        The amount by which to shift the curve.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.
    """

    _obj: _BaseCurve

    def __init__(
        self,
        curve: _BaseCurve,
        shift: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
    ) -> None:
        start, end = curve._nodes.initial, curve._nodes.final

        if curve._base_type == _CurveType.dfs:
            dcf_ = dcf(start, end, curve._meta.convention, calendar=curve._meta.calendar)
            _, d, n = average_rate(start, end, curve._meta.convention, 0.0, dcf_)
            shifted: _BaseCurve = Curve(
                nodes={start: 1.0, end: 1.0 / (1 + d * shift / 10000) ** n},
                convention=curve._meta.convention,
                calendar=curve._meta.calendar,
                modifier=curve._meta.modifier,
                interpolation="log_linear",
                index_base=curve._meta.index_base,
                index_lag=curve._meta.index_lag,
                ad=_get_order_of(shift),
            )
        else:  # base type is values: LineCurve
            shifted = LineCurve(
                nodes={start: shift / 100.0, end: shift / 100.0},
                convention=curve._meta.convention,
                calendar=curve._meta.calendar,
                modifier=curve._meta.modifier,
                interpolation="flat_backward",
                ad=_get_order_of(shift),
            )

        id_ = _drb(curve.id + "_shift_" + f"{_dual_float(shift):.1f}", id)

        if shifted._ad + curve._ad == 3:
            raise TypeError(
                "Cannot create a _ShiftedCurve with mixed AD orders.\n"
                f"`curve` has AD order: {curve.ad}\n"
                f"`shift` has AD order: {shifted.ad}"
            )
        self._obj = CompositeCurve(curves=[curve, shifted], id=id_, _no_validation=True)

    def __getitem__(self, date: datetime) -> DualTypes:
        return self._obj.__getitem__(date)

    def _set_ad_order(self, ad: int) -> None:
        return self._obj._set_ad_order(ad)

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self._obj._ad

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        return self._obj._meta

    @property
    def _id(self) -> str:  # type: ignore[override]
        return self._obj._id

    @property
    def _nodes(self) -> _CurveNodes:  # type: ignore[override]
        return self._obj._nodes

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self._obj._interpolator

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self._obj._base_type


class _TranslatedCurve(_WithOperations, _BaseCurve):
    """A class which wraps the underlying curve and returns identical rates but which acts as if the
    initial node date is moved forward in time.

    This is mostly used by discount factor (DF) based curves whose DFs are adjusted to have a
    value of 1.0 on the requested start date.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    start: datetime
        The date which acts as the new initial node date. Must be later than the initial node
        date of ``curve``.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.
    """

    _obj: _BaseCurve

    def __init__(
        self,
        curve: _BaseCurve,
        start: datetime,
        id: str_ = NoInput(0),  # noqa: A002
    ) -> None:
        if start < curve.nodes.initial:
            raise ValueError("Cannot translate into the past.")
        self._id = _drb(curve.id + "_translated_" + f"{start.strftime('yy_mm_dd')}", id)
        self._nodes = _CurveNodes(_nodes={start: 0.0, curve.nodes.final: 0.0})
        self._obj = curve

    def __getitem__(self, date: datetime) -> DualTypes:
        if date < self.nodes.initial:
            return 0.0
        elif self._base_type == _CurveType.dfs:
            return self._obj.__getitem__(date) / self._obj.__getitem__(self.nodes.initial)
        else:  # _CurveType.values
            return self._obj.__getitem__(date)

    def _set_ad_order(self, ad: int) -> None:
        return self._obj._set_ad_order(ad)

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self._obj._ad

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self._obj._interpolator

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        if self._base_type == _CurveType.dfs and not isinstance(self._obj.meta.index_base, NoInput):
            return replace(
                self._obj.meta,
                _index_base=self._obj.index_value(self.nodes.initial, self._obj.meta.index_lag),  # type: ignore[arg-type]
            )
        else:
            return self._obj._meta

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self._obj._base_type


class _RolledCurve(_WithOperations, _BaseCurve):
    """A class which wraps the underlying curve and returns rates which are rolled in time,
    measured by a set number of calendar days.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    start: datetime
        The date which acts as the new initial node date. Must be later than the initial node
        date of ``curve``.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.
    """

    _obj: _BaseCurve
    _roll_days: int

    def __init__(
        self,
        curve: _BaseCurve,
        roll_days: int,
        id: str_ = NoInput(0),  # noqa: A002
    ) -> None:
        self._roll_days = roll_days
        self._id = _drb(curve.id + "_rolled_" + f"{roll_days}", id)
        self._obj = curve

    def __getitem__(self, date: datetime) -> DualTypes:
        if date < self.nodes.initial:
            return 0.0

        boundary = self.nodes.initial + timedelta(days=self._roll_days)
        if self._base_type == _CurveType.dfs:
            if self._roll_days <= 0:
                # boundary is irrelevant
                scalar_date = self._obj.nodes.initial + timedelta(days=-self._roll_days)
                return self._obj.__getitem__(
                    date - timedelta(days=self._roll_days)
                ) / self._obj.__getitem__(scalar_date)
            else:
                next_day = add_tenor(self.nodes.initial, "1b", "F", self._obj.meta.calendar)
                on_rate = self._obj._rate_with_raise(self.nodes.initial, next_day)
                dcf_ = dcf(
                    self.nodes.initial,
                    next_day,
                    self._obj.meta.convention,
                    calendar=self._obj.meta.calendar,
                )
                r_, d_, n_ = average_rate(
                    self.nodes.initial, next_day, self._obj.meta.convention, on_rate, dcf_
                )
                if self.nodes.initial <= date < boundary:
                    # must project forward
                    return 1.0 / (1 + r_ * d_ / 100.0) ** (date - self.nodes.initial).days
                else:  # boundary <= date:
                    scalar = (1.0 + d_ * r_ / 100) ** self._roll_days
                    return self._obj.__getitem__(date - timedelta(days=self._roll_days)) / scalar
        else:  # _CurveType.values
            if self.nodes.initial <= date < boundary:
                return self._obj.__getitem__(self.nodes.initial)
            else:  # boundary <= date:
                return self._obj.__getitem__(date - timedelta(days=self._roll_days))

    def _set_ad_order(self, ad: int) -> None:
        return self._obj._set_ad_order(ad)

    @property
    def roll_days(self) -> int:
        """The number of calendar days by which rates are rolled on the underlying curve."""
        return self._roll_days

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self._obj._ad

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self._obj._interpolator

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        return self._obj._meta

    @property
    def _nodes(self) -> _CurveNodes:  # type: ignore[override]
        return self._obj._nodes

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self._obj._base_type


class Curve(_WithOperations, _CurveMutation):
    """
    Curve based on DF parametrisation at given node dates with interpolation.

    Parameters
    ----------
    nodes : dict[datetime: float]
        Parameters of the curve denoted by a node date and a corresponding
        DF at that point.
    interpolation : str or callable
        The interpolation used in the non-spline section of the curve. That is the part
        of the curve between the first node in ``nodes`` and the first knot in ``t``.
        If a callable, this allows a user-defined interpolation scheme, and this must
        have the signature ``method(date, curve)``, where ``date`` is the datetime
        whose DF will be returned and ``curve`` is passed as ``self``.
    t : list[datetime], optional
        The knot locations for the B-spline log-cubic interpolation section of the
        curve. If *None* all interpolation will be done by the local method specified in
        ``interpolation``.
    endpoints : 2-tuple of str, optional
        The left and then right endpoint constraint for the spline solution. Valid values are
        in {"natural", "not_a_knot"}.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multicurve framework.
    convention : str, optional, set by Default
        The convention of the curve for determining rates. Please see
        :meth:`dcf()<rateslib.calendars.dcf>` for all available options.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates when input as
        a tenor, e.g. "3M".
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data. Used for determining rates.
    ad : int in {0, 1, 2}, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.
    index_base: float, optional
        The initial index value at the initial node date of the curve. Used for
        forecasting future index values.
    index_lag : int, optional
        Number of months of by which the index lags the date. For example if the initial
        curve node date is 1st Sep 2021 based on the inflation index published
        17th June 2023 then the lag is 3 months. Best practice is to use 0 months.
    collateral : str
        A currency identifier to denote the collateral currency against which the discount factors
        for this *Curve* are measured.
    credit_discretization : int
        A parameter for numerically solving the integral for credit protection legs and default
        events. Expressed in calendar days. Only used by *Curves* functioning as *hazard Curves*.
    credit_recovery_rate : Variable | float
        A parameter used in pricing credit protection legs and default events.

    Notes
    -----
    This curve type is **discount factor (DF)** based and is parametrised by a set of
    (date, DF) pairs set as ``nodes``. The initial node date of the curve is defined
    to be today and should **always** have a DF of precisely 1.0. The initial DF
    will **not** be affected by a :class:`~rateslib.solver.Solver`.

    Intermediate DFs are determined through ``interpolation``. If local interpolation
    is adopted a DF for an arbitrary date is dependent only on its immediately
    neighbouring nodes via the interpolation routine. Available options are:

    - *"log_linear"* (default for this curve type)
    - *"linear_index"*

    And also the following which are not recommended for this curve type:

    - *"linear"*,
    - *"linear_zero_rate"*,
    - *"flat_forward"*,
    - *"flat_backward"*,

    **Spline Interpolation**

    Global interpolation in the form of a **log-cubic** spline is also configurable
    with the parameters ``t``, and ``endpoints``. Setting an ``interpolation`` of *"spline"*
    is syntactic sugar for automatically determining the most obvious
    knot sequence ``t`` to use all specified *node dates*. See
    :ref:`splines<splines-doc>` for instruction of knot sequence calibration.

    If the knot sequence is provided directly then any dates prior to the first knot date in ``t``
    will be determined through the local interpolation method. This allows for
    **mixed interpolation**, permitting the most common form of a stepped curve followed by a
    smooth curve at some boundary.

    For defining rates by a given tenor, the ``modifier`` and ``calendar`` arguments
    will be used. For correct scaling of the rate a ``convention`` is attached to the
    curve, which is usually one of "Act360" or "Act365F".

    Examples
    --------

    .. ipython:: python

       nodes={
           dt(2022,1,1): 1.0,  # <- initial DF should always be 1.0
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       }
       curve1 = Curve(nodes=nodes, interpolation="log_linear")
       curve2 = Curve(nodes=nodes, interpolation="spline")
       curve1.plot("1d", comparators=[curve2], labels=["log_linear", "log_cubic_spline"])

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       import numpy as np

       nodes={
           dt(2022,1,1): 1.0,  # <- initial DF should always be 1.0
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       }
       curve1 = Curve(nodes=nodes, interpolation="log_linear")
       curve2 = Curve(nodes=nodes, interpolation="spline")
       fig, ax, line = curve1.plot("1d", comparators=[curve2], labels=["log_linear", "log_cubic_spline"])
       plt.show()
       plt.close()
    """  # noqa: E501

    _ini_solve: int = 1  # Curve is assumed to have initial DF node at 1.0 as constraint
    _base_type = _CurveType.dfs

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, date: datetime) -> DualTypes:
        return super().__getitem__(date)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    # Serialization

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> Curve:
        """
        Reconstitute a curve from JSON.

        Parameters
        ----------
        curve : str
            The JSON string representation of the curve.

        Returns
        -------
        Curve or LineCurve
        """
        from rateslib.serialization import from_json

        meta = from_json(loaded_json["meta"])
        interpolator = from_json(loaded_json["interpolator"])
        nodes = from_json(loaded_json["nodes"])
        spl = interpolator.spline

        if interpolator.local_name == "spline":
            t = NoInput(0)
        else:
            t = NoInput(0) if spl is None else spl.t

        return cls(
            nodes=nodes.nodes,
            interpolation=interpolator.local_name,
            t=t,
            endpoints=spl.endpoints if spl is not None else NoInput(0),
            id=loaded_json["id"],
            convention=meta.convention,
            modifier=meta.modifier,
            calendar=meta.calendar,
            ad=loaded_json["ad"],
            index_base=meta.index_base,
            index_lag=meta.index_lag,
            collateral=meta.collateral,
            credit_discretization=meta.credit_discretization,
            credit_recovery_rate=meta.credit_recovery_rate,
        )

    def to_json(self) -> str:
        """
        Serialize this object to JSON format.

        The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.

        Returns
        -------
        str

        Notes
        -----
        Some *Curves* will **not** be serializable, for example those that possess user defined
        interpolation functions.
        """
        obj = dict(
            PyNative={
                f"{type(self).__name__}": dict(
                    meta=self._meta.to_json(),
                    interpolator=self._interpolator.to_json(),
                    id=self._id,
                    ad=self._ad,
                    nodes=self._nodes.to_json(),
                )
            }
        )
        return json.dumps(obj)


class LineCurve(_WithOperations, _CurveMutation):
    """
    Curve based on value parametrisation at given node dates with interpolation.

    Parameters
    ----------
    nodes : dict[datetime: float]
        Parameters of the curve denoted by a node date and a corresponding
        value at that point.
    interpolation : str in {"log_linear", "linear"} or callable
        The interpolation used in the non-spline section of the curve. That is the part
        of the curve between the first node in ``nodes`` and the first knot in ``t``.
        If a callable, this allows a user-defined interpolation scheme, and this must
        have the signature ``method(date, nodes)``, where ``date`` is the datetime
        whose DF will be returned and ``nodes`` is as above and is passed to the
        callable.
    t : list[datetime], optional
        The knot locations for the B-spline cubic interpolation section of the
        curve. If *None* all interpolation will be done by the method specified in
        ``interpolation``.
    endpoints : str or list, optional
        The left and right endpoint constraint for the spline solution. Valid values are
        in {"natural", "not_a_knot"}. If a list, supply the left endpoint then the
        right endpoint.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.
        convention : str, optional, set by Default
        The convention of the curve for determining rates. Please see
        :meth:`dcf()<rateslib.calendars.dcf>` for all available options.
    convention : str, optional, set by Default
        The convention of the curve for determining rates. Please see
        :meth:`dcf()<rateslib.calendars.dcf>` for all available options.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates when input as
        a tenor, e.g. "3M".
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data. Used for determining rates.
    ad : int in {0, 1, 2}, optional
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`Dual` or :class:`Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    The arguments ``index_base``, ``index_lag``, and ``collateral`` available on
    :class:`~rateslib.curves.Curve` are not used by, or relevant for, a :class:`LineCurve`.

    This curve type is **value** based and it is parametrised by a set of
    (date, value) pairs set as ``nodes``. The initial node date of the curve is defined
    to be today, and can take a general value. The initial value
    will be affected by a :class:`~rateslib.solver.Solver`.

    .. note::

       This curve type can only ever be used for **forecasting** rates and projecting
       cashflow calculations. It cannot be used to discount cashflows becuase it is
       not DF based and there is no mathematical one-to-one conversion available to
       imply DFs.

    Intermediate values are determined through ``interpolation``. If local interpolation
    is adopted a value for an arbitrary date is dependent only on its immediately
    neighbouring nodes via the interpolation routine. Available options are:

    - *"linear"* (default for this curve type)
    - *"log_linear"* (useful for values that exponential, e.g. stock indexes or GDP)
    - *"spline"*
    - *"flat_forward"*, (useful for replicating a DF based log-linear type curve)
    - *"flat_backward"*,

    And also the following which are not recommended for this curve type:

    - *"linear_index"*
    - *"linear_zero_rate"*,

    **Spline Interpolation**

    Global interpolation in the form of a **cubic** spline is also configurable
    with the parameters ``t``, and ``endpoints``. Setting an ``interpolation`` of *"spline"*
    is syntactic sugar for automatically determining the most obvious
    knot sequence ``t`` to use all specified *node dates*. See
    :ref:`splines<splines-doc>` for instruction of knot sequence calibration.

    If the knot sequence is provided directly then any dates prior to the first knot date in ``t``
    will be determined through the local interpolation method. This allows for
    **mixed interpolation**.

    This curve type cannot return arbitrary tenor rates. It will only return a single
    value which is applicable to that date. It is recommended to review
    :ref:`RFR and IBOR Indexing<c-curves-ibor-rfr>` to ensure indexing is done in a
    way that is consistent with internal instrument configuration.

    Examples
    --------

    .. ipython:: python

       nodes = {
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       }
       line_curve1 = LineCurve(nodes=nodes, interpolation="linear")
       line_curve2 = LineCurve(nodes=nodes, interpolation="spline")
       line_curve1.plot("1d", comparators=[line_curve2], labels=["linear", "cubic spline"])

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       import numpy as np
       nodes = {
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       }
       line_curve1 = LineCurve(nodes=nodes, interpolation="linear")
       line_curve2 = LineCurve(nodes=nodes, interpolation="spline")
       fig, ax, line = line_curve1.plot("1d", comparators=[line_curve2], labels=["linear", "cubic spline"])
       plt.show()
       plt.close()

    """  # noqa: E501

    _ini_solve = 0  # No constraint placed on initial node in Solver
    _base_type = _CurveType.values

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, date: datetime) -> DualTypes:
        return super().__getitem__(date)


class CompositeCurve(_WithOperations, _BaseCurve):
    """
    A dynamic composition of a sequence of other curves.

    .. note::
       Can only composite curves of the same type: :class:`Curve`
       or :class:`LineCurve`. Other curve parameters such as ``modifier``, ``calendar``
       and ``convention`` must also match.

    Parameters
    ----------
    curves : sequence of :class:`Curve` or sequence of :class:`LineCurve`
        The curves to be composited.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.

    Examples
    --------
    Composite two :class:`LineCurve` s. Here, simulating the effect of adding
    quarter-end turns to a cubic spline interpolator, which is otherwise difficult to
    mathematically derive.

    .. ipython:: python
       :suppress:

       from datetime import datetime as dt

    .. ipython:: python

       from rateslib.curves import LineCurve, CompositeCurve
       line_curve1 = LineCurve(
           nodes={
               dt(2022, 1, 1): 2.5,
               dt(2023, 1, 1): 3.5,
               dt(2024, 1, 1): 3.0,
           },
           t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
              dt(2023, 1, 1),
              dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1)],
       )
       line_curve2 = LineCurve(
           nodes={
               dt(2022, 1, 1): 0,
               dt(2022, 3, 31): -0.2,
               dt(2022, 4, 1): 0,
               dt(2022, 6, 30): -0.2,
               dt(2022, 7, 1): 0,
               dt(2022, 9, 30): -0.2,
               dt(2022, 10, 1): 0,
               dt(2022, 12, 31): -0.2,
               dt(2023, 1, 1): 0,
               dt(2023, 3, 31): -0.2,
               dt(2023, 4, 1): 0,
               dt(2023, 6, 30): -0.2,
               dt(2023, 7, 1): 0,
               dt(2023, 9, 30): -0.2,
           },
           interpolation="flat_forward",
       )
       curve = CompositeCurve([line_curve1, line_curve2])
       curve.plot("1d")

    .. plot::

       from rateslib.curves import LineCurve, CompositeCurve
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       line_curve1 = LineCurve(
           nodes={
               dt(2022, 1, 1): 2.5,
               dt(2023, 1, 1): 3.5,
               dt(2024, 1, 1): 3.0,
           },
           t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
              dt(2023, 1, 1),
              dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1)],
       )
       line_curve2 = LineCurve(
           nodes={
               dt(2022, 1, 1): 0,
               dt(2022, 3, 31): -0.2,
               dt(2022, 4, 1): 0,
               dt(2022, 6, 30): -0.2,
               dt(2022, 7, 1): 0,
               dt(2022, 9, 30): -0.2,
               dt(2022, 10, 1): 0,
               dt(2022, 12, 31): -0.2,
               dt(2023, 1, 1): 0,
               dt(2023, 3, 31): -0.2,
               dt(2023, 4, 1): 0,
               dt(2023, 6, 30): -0.2,
               dt(2023, 7, 1): 0,
               dt(2023, 9, 30): -0.2,
           },
           interpolation="flat_forward",
       )
       curve = CompositeCurve([line_curve1, line_curve2])
       fig, ax, line = curve.plot("1D")
       plt.show()

    We can also composite DF based curves by using a fast approximation or an
    exact match.

    .. ipython:: python

       from rateslib.curves import Curve, CompositeCurve
       curve1 = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.98,
               dt(2024, 1, 1): 0.965,
               dt(2025, 1, 1): 0.955
           },
           t=[dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1),
              dt(2024, 1, 1),
              dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1)],
       )
       curve2 =Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2022, 6, 30): 1.0,
               dt(2022, 7, 1): 0.999992,
               dt(2022, 12, 31): 0.999992,
               dt(2023, 1, 1): 0.999984,
               dt(2023, 6, 30): 0.999984,
               dt(2023, 7, 1): 0.999976,
               dt(2023, 12, 31): 0.999976,
               dt(2024, 1, 1): 0.999968,
               dt(2024, 6, 30): 0.999968,
               dt(2024, 7, 1): 0.999960,
               dt(2025, 1, 1): 0.999960,
           },
       )
       curve = CompositeCurve([curve1, curve2])
       curve.plot("1D", comparators=[curve1, curve2], labels=["Composite", "C1", "C2"])

    .. plot::

       from rateslib.curves import Curve, CompositeCurve
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       curve1 = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.98,
               dt(2024, 1, 1): 0.965,
               dt(2025, 1, 1): 0.955
           },
           t=[dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1), dt(2023, 1, 1),
              dt(2024, 1, 1),
              dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1), dt(2025, 1, 1)],
       )
       curve2 =Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2022, 6, 30): 1.0,
               dt(2022, 7, 1): 0.999992,
               dt(2022, 12, 31): 0.999992,
               dt(2023, 1, 1): 0.999984,
               dt(2023, 6, 30): 0.999984,
               dt(2023, 7, 1): 0.999976,
               dt(2023, 12, 31): 0.999976,
               dt(2024, 1, 1): 0.999968,
               dt(2024, 6, 30): 0.999968,
               dt(2024, 7, 1): 0.999960,
               dt(2025, 1, 1): 0.999960,
           },
       )
       curve = CompositeCurve([curve1, curve2])
       fig, ax, line = curve.plot("1D", comparators=[curve1, curve2], labels=["Composite", "C1", "C2"])
       plt.show()

    """  # noqa: E501

    _mutable_by_association = True
    _do_not_validate = False

    def __init__(
        self,
        curves: list[_BaseCurve] | tuple[_BaseCurve, ...],
        id: str_ = NoInput(0),  # noqa: A002
        _no_validation: bool = False,
    ) -> None:
        self._id = _drb(uuid4().hex[:5], id)  # 1 in a million clash
        self.curves = tuple(curves)

        # TODO why does a CompositeCurve require node_dates?
        nodes_proxy: dict[datetime, DualTypes] = dict.fromkeys(self.curves[0].nodes.keys, 0.0)
        self._nodes = _CurveNodes(nodes_proxy)
        self._meta = _CurveMeta(
            curves[0].meta.calendar,
            curves[0].meta.convention,
            curves[0].meta.modifier,
            curves[0].meta.index_base,
            curves[0].meta.index_lag,
            curves[0].meta.collateral,
            curves[0].meta.credit_discretization,
            curves[0].meta.credit_recovery_rate,
        )
        self._base_type = curves[0]._base_type

        # validate
        if not _no_validation:
            self._validate_curve_collection()
        self._ad = max(_._ad for _ in self.curves)
        self._clear_cache()
        self._set_new_state()

    def _validate_curve_collection(self) -> None:
        """Perform checks to ensure CompositeCurve can exist"""
        if type(self) is MultiCsaCurve and isinstance(self.curves[0], LineCurve):
            raise TypeError("Multi-CSA curves must be of type `Curve`.")

        if type(self) is MultiCsaCurve and self.multi_csa_min_step > self.multi_csa_max_step:
            raise ValueError("`multi_csa_max_step` cannot be less than `min_step`.")

        types = [_._base_type for _ in self.curves]
        if not all(_ == types[0] for _ in types):
            # then at least one curve is value based and one is DF based
            raise TypeError("CompositeCurve can only contain curves of the same type.")

        ini_dates = [_.nodes.initial for _ in self.curves]
        if not all(_ == ini_dates[0] for _ in ini_dates[1:]):
            raise ValueError(f"`curves` must share the same initial node date, got {ini_dates}")

        if type(self) is not MultiCsaCurve:  # for multi_csa DF curve do not check calendars
            self._check_meta_attribute("calendar")

        if self._base_type == _CurveType.dfs:
            self._check_meta_attribute("modifier")
            self._check_meta_attribute("convention")
            # self._check_meta_attribute("collateral")  # not used due to inconsistent labelling

        _ad = [_._ad for _ in self.curves]
        if 1 in _ad and 2 in _ad:
            raise TypeError(
                "CompositeCurve cannot composite curves of AD order 1 and 2.\n"
                "Either downcast curves using `curve._set_ad_order(1)`.\n"
                "Or upcast curves using `curve._set_ad_order(2)`.\n"
            )

    def _check_meta_attribute(self, attr: str) -> None:
        """Ensure attributes are the same across curve collection"""
        attrs = [getattr(_.meta, attr, None) for _ in self.curves]
        if not all(_ == attrs[0] for _ in attrs[1:]):
            raise ValueError(
                f"Cannot composite curves with different attributes, got for "
                f"'{attr}': {[getattr(_.meta, attr, None) for _ in self.curves]},",
            )

    @_validate_states
    @_no_interior_validation
    def rate(  # type: ignore[override]
        self,
        effective: datetime,
        termination: datetime | str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(1),
    ) -> DualTypes | None:
        """
        Calculate the composited rate on the curve.

        If rates are sought for dates prior to the initial node of the curve `None`
        will be returned.

        Parameters
        ----------
        effective : datetime
            The start date of the period for which to calculate the rate.
        termination : datetime or str
            The end date of the period for which to calculate the rate.
        modifier : str, optional
            The day rule if determining the termination from tenor. If `False` is
            determined from the `Curve` modifier.

        Returns
        -------
        Dual, Dual2 or float
        """
        if effective < self.nodes.initial:  # Alternative solution to PR 172.
            return None

        if self._base_type == _CurveType.values:
            _: DualTypes = 0.0
            for i in range(len(self.curves)):
                # let regular TypeErrors be raised if curve.rate returns None
                _ += self.curves[i].rate(effective, termination, modifier)  # type: ignore[operator]
            return _
        else:  #  self._base_type == "dfs":
            modifier_ = _drb(self.meta.modifier, modifier)

            if isinstance(termination, NoInput):
                raise ValueError("`termination` must be give for rate of DF based Curve.")
            elif isinstance(termination, str):
                termination = add_tenor(effective, termination, modifier_, self.meta.calendar)

            # using determined and cached discount factors
            df_start = self.__getitem__(effective)
            df_end = self.__getitem__(termination)
            d = dcf(
                effective,
                termination,
                self.meta.convention,
                NoInput(0),
                NoInput(0),
                NoInput(0),
                NoInput(0),
                self.meta.calendar,
            )
            _ = (df_start / df_end - 1) * 100 / d

        return _

    @_validate_states
    @_no_interior_validation
    def __getitem__(self, date: datetime) -> DualTypes:
        if defaults.curve_caching and date in self._cache:
            return self._cache[date]
        if self._base_type == _CurveType.dfs:
            # will return a composited discount factor
            if date == self.nodes.initial:
                # this value is 1.0, but by multiplying capture AD versus initial nodes.
                ret: DualTypes = prod(crv[date] for crv in self.curves)
                return ret
            elif date < self.nodes.initial:
                return 0.0  # Any DF in the past is set to zero consistent with behaviour on `Curve`

            dcf_ = dcf(
                start=self.nodes.initial,
                end=date,
                convention=self.meta.convention,
                calendar=self.meta.calendar,
            )
            _, d, n = average_rate(self.nodes.initial, date, self.meta.convention, 0.0, dcf_)
            total_rate: Number = 0.0
            for curve in self.curves:
                avg_rate = ((1.0 / curve[date]) ** (1.0 / n) - 1) / d
                total_rate += avg_rate
            ret = 1.0 / (1 + total_rate * d) ** n
            return self._cached_value(date, ret)

        else:  # self._base_type == _CurveType.values:
            # will return a composited rate
            _ = 0.0
            for curve in self.curves:
                _ += curve[date]
            return self._cached_value(date, _)

    # Solver interaction

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        """
        Change the node values on each curve to float, Dual or Dual2 based on input parameter.
        """
        if order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
        for curve in self.curves:
            if type(curve) is ProxyCurve:
                continue
                # raise TypeError("Cannot directly set the ad of ProxyCurve. Set the FXForwards.")
                # TODO: decide if setting the AD of the associated FXForwards is viable
            curve._set_ad_order(order)

    # Mutation

    def _validate_state(self) -> None:
        if self._do_not_validate:
            return None
        if self._state != self._get_composited_state():
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    def _get_composited_state(self) -> int:
        return hash(sum(curve._state for curve in self.curves))


class MultiCsaCurve(CompositeCurve):
    """
    A dynamic composition of a sequence of other curves.

    .. note::
       Can only combine curves of the type: :class:`Curve`. Other curve parameters such as
       ``modifier``, and ``convention`` must also match.

    .. warning::
       Intrinsic *MultiCsaCurves*, by definition, are not natively AD safe, due to having
       discontinuities and no available derivatives in certain cases. See
       :ref:`discontinuous MultiCsaCurves <cook-multicsadisc-doc>`.

    Parameters
    ----------
    curves : sequence of :class:`Curve`
        The curves to be composited.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.
    multi_csa_min_step: int, optional
        The minimum calculation step between subsequent DF evaluations to determine a multi-CSA
        curve term DF. Higher numbers make faster calculations but are less accurate. Should be
        in [1, max_step].
    multi_csa_max_step: int, optional
        The minimum calculation step between subsequent DF evaluations to determine a multi-CSA
        curve term DF. Higher numbers make faster calculations but are less accurate. Should be
        in [min_step, 1825].

    Notes
    -----
    A *MultiCsaCurve* uses a different calculation methodology than a *CompositeCurve* for
    determining the *rate* by selecting the curve within the collection with the highest rate.
    """

    def __init__(
        self,
        curves: list[_BaseCurve] | tuple[_BaseCurve, ...],
        id: str | NoInput = NoInput(0),  # noqa: A002
        multi_csa_min_step: int = 1,
        multi_csa_max_step: int = 1825,
    ) -> None:
        self.multi_csa_min_step = max(1, multi_csa_min_step)
        self.multi_csa_max_step = min(1825, multi_csa_max_step)
        super().__init__(curves, id)

    @_validate_states
    @_no_interior_validation
    def rate(  # type: ignore[override]
        self,
        effective: datetime,
        termination: datetime | str,
        modifier: str | NoInput = NoInput(1),
    ) -> DualTypes | None:
        """
        Calculate the cheapest-to-deliver (CTD) rate on the curve.

        If rates are sought for dates prior to the initial node of the curve `None`
        will be returned.

        Parameters
        ----------
        effective : datetime
            The start date of the period for which to calculate the rate.
        termination : datetime or str
            The end date of the period for which to calculate the rate.
        modifier : str, optional
            The day rule if determining the termination from tenor. If `False` is
            determined from the `Curve` modifier.

        Returns
        -------
        Dual, Dual2 or float
        """
        if effective < self.curves[0].nodes.initial:  # Alternative solution to PR 172.
            return None

        modifier_ = self.meta.modifier if isinstance(modifier, NoInput) else modifier
        if isinstance(termination, str):
            termination = add_tenor(effective, termination, modifier_, self.meta.calendar)

        dcf_ = dcf(effective, termination, self.meta.convention, calendar=self.meta.calendar)
        _, d, n = average_rate(effective, termination, self.meta.convention, 0.0, dcf_)
        # TODO (low:perf) when these discount factors are looked up the curve repeats
        # the lookup could be vectorised to return two values at once.
        df_num = self[effective]
        df_den = self[termination]
        ret: DualTypes = (df_num / df_den - 1) * 100 / (d * n)
        return ret

    @_validate_states
    @_no_interior_validation
    def __getitem__(self, date: datetime) -> DualTypes:
        # TODO: changing the multi_csa_step size should force a cache clear. This is a mutation.

        # will return a composited discount factor
        if defaults.curve_caching and date in self._cache:
            return self._cache[date]

        if date == self.nodes.initial:
            # this value is 1.0, but by multiplying capture AD versus initial nodes.
            ret: DualTypes = prod(crv[date] for crv in self.curves)
            return ret
        elif date < self.nodes.initial:
            return 0.0  # Any DF in the past is set to zero consistent with behaviour on `Curve`

        def _get_step(step: int) -> int:
            return min(max(step, self.multi_csa_min_step), self.multi_csa_max_step)

        # method uses the step and picks the highest (cheapest rate) in each step
        d1 = self.nodes.initial
        d2 = d1 + timedelta(days=_get_step(defaults.multi_csa_steps[0]))

        v: DualTypes = self.__getitem__(d1)
        v_i_1_j: list[DualTypes] = [curve[d1] for curve in self.curves]
        v_i_j: list[DualTypes] = [0.0 for curve in self.curves]

        k: int = 1
        while d2 < date:
            if defaults.curve_caching and d2 in self._cache:
                v = self._cache[d2]
                v_i_1_j = [curve[d2] for curve in self.curves]
            else:
                min_ratio: DualTypes = 1e5
                for j, curve in enumerate(self.curves):
                    v_i_j[j] = curve[d2]
                    ratio_ = v_i_j[j] / v_i_1_j[j]
                    min_ratio = ratio_ if ratio_ < min_ratio else min_ratio
                    v_i_1_j[j] = v_i_j[j]
                v *= min_ratio
                self._cached_value(d2, v)

            try:
                step = _get_step(defaults.multi_csa_steps[k])
            except IndexError:
                step = self.multi_csa_max_step
            d1, d2, k = d2, d2 + timedelta(days=step), k + 1

        # finish the loop on the correct date
        if date == d1:
            return self._cached_value(date, v)
        else:
            min_ratio = 1e5
            for j, curve in enumerate(self.curves):
                ratio_ = curve[date] / v_i_1_j[j]
                min_ratio = ratio_ if ratio_ < min_ratio else min_ratio
            v *= min_ratio
            return self._cached_value(date, v)


# class HazardCurve(Curve):
#     """
#     A subclass of :class:`~rateslib.curves.Curve` with additional methods for
#     credit default calculations.
#
#     Parameters
#     ----------
#     args : tuple
#         Position arguments required by :class:`Curve`.
#     kwargs : dict
#         Keyword arguments required by :class:`Curve`.
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def survival_rate(self, date: datetime):
#         return self[date]


class ProxyCurve(Curve):
    """
    A subclass of :class:`~rateslib.curves.Curve` which returns dynamic DFs based on
    other curves related via :class:`~rateslib.fx.FXForwards` parity.

    Parameters
    ----------
    cashflow : str
        The currency in which cashflows are represented (3-digit code).
    collateral : str
        The currency of the CSA against which cashflows are collateralised (3-digit
        code).
    fx_forwards : FXForwards
        The :class:`~rateslib.fx.FXForwards` object which contains the relating
        FX information and the available :class:`~rateslib.curves.Curve` s.
    convention : str
        The day count convention used for calculating rates. If `None` defaults
        to the convention in the local cashflow currency.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates.
        If `False` will default to the modifier in the local cashflow currency.
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, lookups named calendar
        from static data. Used for determining rates. If `False` will
        default to the calendar in the local cashflow currency.
    id : str, optional, set by Default
        The unique identifier to distinguish between curves in a multi-curve framework.

    Notes
    -----
    The DFs returned are calculated via the chaining method and the below formula,
    relating the DF curve in the local collateral currency and FX forward rates.

    .. math::

       w_{dom:for,i} = \\frac{f_{DOMFOR,i}}{F_{DOMFOR,0}} v_{for:for,i}

    The returned curve contains contrived methods to calculate this dynamically and
    efficiently from the combination of curves and FX rates that are available within
    the given :class:`FXForwards` instance.
    """

    _base_type = _CurveType.dfs
    _interpolator: _ProxyCurveInterpolator  # type: ignore[assignment]
    _nodes: _CurveNodes
    _meta: _CurveMeta

    def __init__(
        self,
        cashflow: str,
        collateral: str,
        fx_forwards: FXForwards,
        convention: str_ = NoInput(1),  # inherits from existing curve objects
        modifier: str_ = NoInput(1),  # inherits from existing curve objects
        calendar: CalInput = NoInput(1),  # inherits from existing curve objects
        id: str_ = NoInput(0),  # noqa: A002
    ):
        self._id = _drb(uuid4().hex[:5], id)  # 1 in a million clash

        self._interpolator = _ProxyCurveInterpolator(
            _fx_forwards=fx_forwards, _cash=cashflow.lower(), _collateral=collateral.lower()
        )

        self._meta = _CurveMeta(
            get_calendar(
                _drb(fx_forwards.fx_curves[self.interpolator.cash_pair].meta.calendar, calendar)
            ),
            _drb(
                fx_forwards.fx_curves[self.interpolator.cash_pair].meta.convention, convention
            ).lower(),
            _drb(
                fx_forwards.fx_curves[self.interpolator.cash_pair].meta.modifier, modifier
            ).upper(),
            NoInput(0),  # index meta not relevant for ProxyCurve
            0,
            self.interpolator.collateral,
            100,  # credit elements irrelevant for a PxyCv
            1.0,  # credit elements irrelevant for a PxyCv
        )
        # CurveNodes attached for date attribution
        self._nodes = _CurveNodes(
            {
                fx_forwards.immediate: 0.0,
                fx_forwards.fx_curves[self.interpolator.cash_pair].nodes.final: 0.0,
            }
        )

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.interpolator.fx_forwards._ad

    @property
    def interpolator(self) -> _ProxyCurveInterpolator:  # type: ignore[override]
        """An instance of :class:`~rateslib.curves.utils._ProxyCurveInterpolator`."""
        return self._interpolator

    @property
    def _state(self) -> int:  # type: ignore[override]
        # ProxyCurve is directly associated with its FXForwards object
        self.interpolator.fx_forwards._validate_state()
        return self.interpolator.fx_forwards._state

    def __getitem__(self, date: datetime) -> DualTypes:
        _1: DualTypes = self.interpolator.fx_forwards.rate(self.interpolator.pair, date)
        _2: DualTypes = self.interpolator.fx_forwards.fx_rates_immediate._fx_array_el(
            self.interpolator.cash_index, self.interpolator.collateral_index
        )
        _3: DualTypes = self.interpolator.fx_forwards.fx_curves[self.interpolator.collateral_pair][
            date
        ]
        return _1 / _2 * _3

    # Not Implemented

    def plot_index(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override] # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types are not index curves.")

    def csolve(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types are associations without parameters.")

    def index_value(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override] # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError(
            "ProxyCurve types are not index curves with an `index base` attribute."
        )

    def to_json(self) -> str:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types are associations that cannot be serialized.")

    @classmethod
    def from_json(cls, curve: str, **kwargs: Any) -> Curve:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types are associations that cannot be serialized.")

    def _set_ad_order(self, order: int) -> None:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError(
            "ProxyCurve types derive their AD order from their parent FXForwards."
        )

    def _get_node_vector(self) -> Arr1dF64 | Arr1dObj:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("Instances of ProxyCurve do not have solvable variables.")

    def copy(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override] # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types are associations that cannot be copied.")

    def update(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types do not provide update methods.")

    def update_node(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types do not provide update methods.")

    def update_meta(self, key: datetime, value: DualTypes) -> None:  # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types do not provide update methods.")


def index_value(
    index_lag: int,
    index_method: str,
    index_fixings: DualTypes | Series[DualTypes] | NoInput = NoInput(0),  # type: ignore[type-var]
    index_date: datetime_ = NoInput(0),
    index_curve: CurveOption_ = NoInput(0),
) -> DualTypes | NoInput:
    """
    Determine an index value from a reference date using combinations of known fixings and
    forecast from a *Curve*.

    Parameters
    ----------
    index_lag: int
        The number of months by which the reference ``index_date`` should be lagged to derive a
        value.
    index_method: str in {"curve", "daily", "monthly"}
        The method used to derive and interpolate index values.
    index_fixings: float, Dual, Dual2, Variable, Series[DualTypes], optional
        A specific index value which is returned directly, or if given as a Series applies the
        appropriate ``index_method`` to determine a value. May also forecast from *Curve* if
        necessary. See notes.
    index_date: datetime, optional
        The reference index date for which the index value is sought. Not required if
        ``index_fixings`` is returned directly.
    index_curve: Curve, optional
        The forecast curve from which to derive index values under the appropriate ``index_method``.
        If using *'curve'*, then curve calculations are used directly.

    Returns
    -------
    DualTypes or NoInput

    Notes
    -----
    A *Series* **must** be given with a unique, monotonic increasing index. This will **not** be
    validated.

    When using the *'daily'* or *'monthly'* type ``index_methods`` index values **must** be
    assigned to **the first of the month** to which the publication is relevant.

    The below image is a snippet taken from the UK DMO *'Formulae for Calculating Gilt Prices
    and Yield'*. It outlines the calculation of an *index value* for a reference date using their
    3 month lag and *'daily'* indexing method.

    .. image:: _static/ukdmo_rpi_ex.png
       :alt: Index value calculations
       :align: center
       :width: 291

    This calculation is replicated in *rateslib* in the following way:

    .. ipython:: python

       from rateslib import index_value
       from pandas import Series

       rpi_series = Series(
           [172.2, 173.1, 174.2, 174.4],
           index=[dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 5, 1), dt(2001, 6, 1)]
       )

       index_value(
           index_lag=3,
           index_method="daily",
           index_fixings=rpi_series,
           index_date=dt(2001, 7, 20)
       )

    """
    if isinstance(index_fixings, int | float | Dual | Dual2 | Variable):
        # i_fixings is a given value, probably aligned with an ``index_base``: return directly
        return index_fixings

    if isinstance(index_curve, dict):
        raise NotImplementedError(
            "`index_curve` cannot currently be supplied as dict. Use a Curve type or NoInput(0)."
        )

    if isinstance(index_date, NoInput):
        raise ValueError(
            "Must supply an `index_date` from which to forecast if `index_fixings` is not a value."
        )

    if isinstance(index_fixings, NoInput):
        # forecast from curve if available
        if isinstance(index_curve, NoInput):
            return NoInput(0)
        return index_curve.index_value(index_date, index_lag, index_method)
    elif isinstance(index_fixings, Series):
        if isinstance(index_curve, NoInput):
            return _index_value_from_series_no_curve(
                index_lag,
                index_method,
                index_fixings,  # type: ignore[arg-type]
                index_date,
            )
        else:
            return _index_value_from_mixed_series_and_curve(
                index_lag,
                index_method,
                index_fixings,  # type: ignore[arg-type]
                index_date,
                index_curve,
            )
    else:
        raise TypeError(
            "`index_fixings` must be of type: Series, DualTypes or NoInput.\n"
            f"{type(index_fixings)} was given."
        )


def _index_value_from_mixed_series_and_curve(
    index_lag: int,
    index_method: str,
    index_fixings: Series[DualTypes],  # type: ignore[type-var]
    index_date: datetime,
    index_curve: _BaseCurve,
) -> DualTypes | NoInput:
    """
    Iterate through possibilities assuming a Curve and fixings as series exists.

    For returning a value from the Series the ``index_lag`` must be zero.
    If the lag is not zero then a Curve method will be used instead which will omit the Series.
    """
    if index_method == "curve":
        if index_date in index_fixings.index:
            # simplest case returns Series value if all checks pass.
            if index_lag == 0:
                return index_fixings.loc[index_date]
            else:
                raise ValueError(
                    "`index_lag` must be zero when using a 'curve' `index_method`.\n"
                    f"`index_date`: {index_date}, is in Series but got `index_lag`: {index_lag}."
                )
        elif len(index_fixings.index) == 0:
            # recall with the curve
            return index_curve.index_value(index_date, index_lag, index_method)
        elif index_lag == 0 and (index_fixings.index[0] < index_date < index_fixings.index[-1]):
            # index date is within the Series index range but not found and the index lag is
            # zero so this should be available
            raise ValueError(
                f"The Series given for `index_fixings` requires, but does not contain, "
                f"the value for date: {index_date}.\n"
                "For inflation indexes using 'monthly' or 'daily' `index_method` the "
                "values associated for a month should be assigned "
                "to the first day of that month."
            )
        else:
            return index_curve.index_value(index_date, index_lag, index_method)
    elif index_method == "monthly":
        date_ = add_tenor(index_date, f"-{index_lag}M", "none", NoInput(0), 1)
        # a monthly value can only be derived from one source.
        # make separate determinations to avoid the issue of mis-matching index lags
        value_from_fixings = index_value(0, "curve", index_fixings, date_, NoInput(0))
        if not isinstance(value_from_fixings, NoInput):
            return value_from_fixings
        else:
            value_from_curve = index_value(
                index_lag, "monthly", NoInput(0), index_date, index_curve
            )
            return value_from_curve
    else:  # i_method == "daily":
        n = monthrange(index_date.year, index_date.month)[1]
        date_som = datetime(index_date.year, index_date.month, 1)
        date_sonm = add_tenor(index_date, "1M", "none", NoInput(0), 1)
        m1 = index_value(index_lag, "monthly", index_fixings, date_som, index_curve)
        if index_date == date_som:
            return m1
        m2 = index_value(index_lag, "monthly", index_fixings, date_sonm, index_curve)
        if isinstance(m2, NoInput) or isinstance(m1, NoInput):
            # then the period is 'future' based, and the fixing is not yet available, or a
            # curve has not been provided to forecast it
            # this line cannot be hit when a curve returns DualTypes and not a NoInput
            # will raise a warning when the curve returns 0.0
            return NoInput(0)
        return m1 + (index_date.day - 1) / n * (m2 - m1)


def _index_value_from_series_no_curve(
    index_lag: int,
    index_method: str,
    index_fixings: Series[DualTypes],  # type: ignore[type-var]
    index_date: datetime,
) -> DualTypes | NoInput:
    """
    Derive a value from a Series only, detecting cases where the errors might be raised.
    """
    if index_method == "curve":
        if index_lag != 0:
            raise ValueError(
                "`index_lag` must be zero when using a 'curve' `index_method`.\n"
                f"`index_date`: {index_date}, is in Series but got `index_lag`: {index_lag}."
            )
        if len(index_fixings.index) == 0:
            return NoInput(0)
        if index_date in index_fixings.index:
            # simplest case returns Series value if all checks pass.
            return index_fixings.loc[index_date]
        if index_date < index_fixings.index[0] or index_date > index_fixings.index[-1]:
            # if requested index date is outside of the scope of the Series return NoInput
            # this handles historic and future cases
            return NoInput(0)
        else:
            # date falls inside the dates of the Series but does not exist.
            raise ValueError(
                f"The Series given for `index_fixings` requires, but does not contain, "
                f"the value for date: {index_date}.\n"
                "For inflation indexes using 'monthly' or 'daily' `index_method` the "
                "values associated for a month should be assigned "
                "to the first day of that month."
            )
    elif index_method == "monthly":
        date_ = add_tenor(index_date, f"-{index_lag}M", "none", NoInput(0), 1)
        return index_value(0, "curve", index_fixings, date_, NoInput(0))
    else:  # i_method == "daily":
        n = monthrange(index_date.year, index_date.month)[1]
        date_som = datetime(index_date.year, index_date.month, 1)
        date_sonm = add_tenor(index_date, "1M", "none", NoInput(0), 1)
        m1 = index_value(index_lag, "monthly", index_fixings, date_som, NoInput(0))
        if index_date == date_som:
            return m1
        m2 = index_value(index_lag, "monthly", index_fixings, date_sonm, NoInput(0))
        if isinstance(m2, NoInput) or isinstance(m1, NoInput):
            # then the period is 'future' based, and the fixing is not yet available, or a
            # curve has not been provided to forecast it
            return NoInput(0)
        return m1 + (index_date.day - 1) / n * (m2 - m1)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
