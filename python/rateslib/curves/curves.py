from __future__ import annotations

from abc import ABC
from calendar import monthrange
from dataclasses import replace
from datetime import datetime, timedelta
from math import prod
from typing import TYPE_CHECKING, Any, TypeAlias

from pandas import Series

from rateslib import defaults
from rateslib.curves.base import _BaseCurve, _WithMutability
from rateslib.curves.utils import (
    _CreditImpliedType,
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
    _new_state_post,
    _no_interior_validation,
    _validate_states,
)
from rateslib.scheduling import add_tenor, dcf

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
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
    """Provides automatic implementation of the curve operations required on a
    :class:`~rateslib.curves._BaseCurve`."""

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
    ) -> ShiftedCurve:
        """
        Create a :class:`~rateslib.curves.ShiftedCurve`: moving *Self* vertically in rate space.

        For examples see the documentation for :class:`~rateslib.curves.ShiftedCurve`.

        Parameters
        ----------
        spread : float, Dual, Dual2, Variable
            The number of basis points added to the existing curve.
        id : str, optional
            Set the id of the returned curve.

        Returns
        -------
        ShiftedCurve

        """
        _: _BaseCurve = self  # type: ignore[assignment]
        return ShiftedCurve(curve=_, shift=spread, id=id)

    @_validate_states
    def translate(self, start: datetime, id: str_ = NoInput(0)) -> TranslatedCurve:  # noqa: A002
        """
        Create a :class:`~rateslib.curves.TranslatedCurve`: maintaining an identical rate space,
        but moving the initial node date forwards in time.

        For examples see the documentation for :class:`~rateslib.curves.TranslatedCurve`.

        Parameters
        ----------
        start : datetime
            The new initial node date for the curve. Must be after the original initial node date.
        id : str, optional
            Set the id of the returned curve.

        Returns
        -------
        TranslatedCurve
        """  # noqa: E501
        _: _BaseCurve = self  # type: ignore[assignment]
        return TranslatedCurve(curve=_, start=start, id=id)

    @_validate_states
    def roll(self, tenor: datetime | str, id: str_ = NoInput(0)) -> RolledCurve:  # noqa: A002
        """
        Create a :class:`~rateslib.curves.RolledCurve`: translating the rate space of *Self* in
        time.

        For examples see the documentation for :class:`~rateslib.curves.RolledCurve`.

        Parameters
        ----------
        tenor : datetime, str or int
            The measure of time by which to translate the curve through time.
        id : str, optional
            Set the id of the returned curve.

        Returns
        -------
        RolledCurve

        """  # noqa: E501
        if isinstance(tenor, str):
            tenor_ = add_tenor(self._nodes.initial, tenor, "NONE", NoInput(0))
        else:
            tenor_ = tenor

        if isinstance(tenor_, int):
            roll_days: int = tenor_
        else:
            roll_days = (tenor_ - self._nodes.initial).days
        _: _BaseCurve = self  # type: ignore[assignment]
        return RolledCurve(curve=_, roll_days=roll_days, id=id)


class ShiftedCurve(_WithOperations, _BaseCurve):
    """
    Create a new :class:`~rateslib.curves._BaseCurve` type by compositing an input with
    another flat curve of a set number of basis points.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    shift: float | Variable
        The amount by which to shift the curve.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.

    Notes
    -----
    For **values** based curves this will add the ``shift`` to every output *rate* generated
    by ``curve``.

    For **discount factor** based curves this will add the ``shift`` as a geometric 1-day average
    rate to the input ``curve``, in accordance with *rateslib*'s definition of curve metric spaces.

    This implies that the *shape* of the ``curve`` is preserved but it undergoes a vertical
    translation in rate space. This class works by wrapping a
    :class:`~rateslib.curves.CompositeCurve` and designing the spread curve according to these
    definitions.

    The **ad order** will be the maximum order of ``curve`` and ``spread``. The usual `TypeError`
    will be raised if mixing of :class:`~rateslib.dual.Dual` and :class:`~rateslib.dual.Dual2`
    is attempted.

    Examples
    --------

    .. ipython:: python
       :suppress:

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
       plt.close()
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
            dcf_ = dcf(start, end, curve.meta.convention, calendar=curve.meta.calendar)
            _, d, n = average_rate(start, end, curve.meta.convention, 0.0, dcf_)
            shifted: _BaseCurve = Curve(
                nodes={start: 1.0, end: 1.0 / (1 + d * shift / 10000) ** n},
                convention=curve.meta.convention,
                calendar=curve.meta.calendar,
                modifier=curve.meta.modifier,
                interpolation="log_linear",
                index_base=curve.meta.index_base,
                index_lag=curve.meta.index_lag,
                ad=_get_order_of(shift),
            )
        else:  # base type is values: LineCurve
            shifted = LineCurve(
                nodes={start: shift / 100.0, end: shift / 100.0},
                convention=curve.meta.convention,
                calendar=curve.meta.calendar,
                modifier=curve.meta.modifier,
                interpolation="flat_backward",
                ad=_get_order_of(shift),
            )

        id_ = _drb(curve.id + "_shift_" + f"{_dual_float(shift):.1f}", id)

        if shifted._ad + curve._ad == 3:
            raise TypeError(
                "Cannot create a ShiftedCurve with mixed AD orders.\n"
                f"`curve` has AD order: {curve.ad}\n"
                f"`shift` has AD order: {shifted.ad}"
            )
        self._obj = CompositeCurve(curves=[curve, shifted], id=id_, _no_validation=True)

    def __getitem__(self, date: datetime) -> DualTypes:
        return self.obj.__getitem__(date)

    def _set_ad_order(self, ad: int) -> None:
        return self.obj._set_ad_order(ad)

    @property
    def obj(self) -> _BaseCurve:
        """The wrapped :class:`~rateslib.curves.CompositeCurve` that performs calculations."""
        return self._obj

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.obj.ad

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        return self.obj.meta

    @property
    def _id(self) -> str:
        return self.obj.id

    @property
    def _nodes(self) -> _CurveNodes:  # type: ignore[override]
        return self.obj.nodes

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self.obj.interpolator

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self.obj._base_type


class TranslatedCurve(_WithOperations, _BaseCurve):
    """
    Create a new :class:`~rateslib.curves._BaseCurve` type by maintaining the rate space of an
    input curve but shifting the initial node date forwards in time.

    A class which wraps the underlying curve and returns rates and/or discount factors which are
    impacted by a change to initial node date. This is mostly used by discount factor (DF) based
    curves whose DFs are adjusted to have a value of 1.0 on the requested start date.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    start: datetime
        The new initial node date for the curve. Must be after the initial node date of the input
        ``curve``.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.

    Examples
    ---------
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

       # Discount factors
       curve[dt(2022, 12, 1)]
       translated_curve[dt(2022, 12, 1)]

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
    """  # noqa: E501

    _obj: _BaseCurve

    # abcs

    _id: str = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]

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
            return self.obj.__getitem__(date) / self.obj.__getitem__(self.nodes.initial)
        else:  # _CurveType.values
            return self.obj.__getitem__(date)

    def _set_ad_order(self, ad: int) -> None:
        return self.obj._set_ad_order(ad)

    @property
    def obj(self) -> _BaseCurve:
        """The wrapped :class:`~rateslib.curves._BaseCurve` object that performs calculations."""
        return self._obj

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.obj.ad

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self.obj.interpolator

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        if self._base_type == _CurveType.dfs and not isinstance(self.obj.meta.index_base, NoInput):
            return replace(
                self.obj.meta,
                _index_base=self.obj.index_value(self.nodes.initial, self.obj.meta.index_lag),  # type: ignore[arg-type]
            )
        else:
            return self.obj.meta

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self.obj._base_type


class RolledCurve(_WithOperations, _BaseCurve):
    """
    Create a new :class:`~rateslib.curves._BaseCurve` type by translating the rate space of an
    input curve horizontally in time.

    A class which wraps the underlying curve and returns rates which are rolled in time,
    measured by a set number of calendar days.

    Parameters
    ----------
    curve: _BaseCurve
        Any *BaseCurve* type.
    roll_days: int
        The number of calendar days by which to translate the curve's rate space.
    id: str, optional
        Identifier used for :class:`~rateslib.solver.Solver` mappings.

    Notes
    -----
    A positive number of ``roll_days`` will shift the ``curve`` rate space to the right.
    This is the traditional direction for measuring *roll down* on a trade strategy.

    The gap between the initial node date and the roll date (if ``roll_days`` is positive) is
    determined by forward filling the first rate on a **values** based curve, or forward filling
    the first overnight rate on a **discount factor** based curve.

    Examples
    ---------
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
           labels=["orig", "6m roll", "-6m roll"],
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
       fig, ax, line = curve.plot("1d", comparators=[rolled_curve, rolled_curve2], labels=["orig", "6m roll", "-6m roll"], right=dt(2026,6,30))
       plt.show()
       plt.close()
    """  # noqa: E501

    _obj: _BaseCurve
    _roll_days: int

    # abcs

    _id: str = None  # type: ignore[assignment]

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
                scalar_date = self.obj.nodes.initial + timedelta(days=-self._roll_days)
                return self.obj.__getitem__(
                    date - timedelta(days=self._roll_days)
                ) / self.obj.__getitem__(scalar_date)
            else:
                next_day = add_tenor(self.nodes.initial, "1b", "F", self.obj.meta.calendar)
                on_rate = self.obj._rate_with_raise(self.nodes.initial, next_day)
                dcf_ = dcf(
                    self.nodes.initial,
                    next_day,
                    self.obj.meta.convention,
                    calendar=self.obj.meta.calendar,
                )
                r_, d_, n_ = average_rate(
                    self.nodes.initial, next_day, self.obj.meta.convention, on_rate, dcf_
                )
                if self.nodes.initial <= date < boundary:
                    # must project forward
                    return 1.0 / (1 + r_ * d_ / 100.0) ** (date - self.nodes.initial).days
                else:  # boundary <= date:
                    scalar = (1.0 + d_ * r_ / 100) ** self._roll_days
                    return self.obj.__getitem__(date - timedelta(days=self._roll_days)) / scalar
        else:  # _CurveType.values
            if self.nodes.initial <= date < boundary:
                return self.obj.__getitem__(self.nodes.initial)
            else:  # boundary <= date:
                return self.obj.__getitem__(date - timedelta(days=self._roll_days))

    def _set_ad_order(self, order: int) -> None:
        return self.obj._set_ad_order(order)

    @property
    def obj(self) -> _BaseCurve:
        """The wrapped :class:`~rateslib.curves._BaseCurve` object that performs calculations."""
        return self._obj

    @property
    def roll_days(self) -> int:
        """The number of calendar days by which rates are rolled on the underlying curve."""
        return self._roll_days

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.obj.ad

    @property
    def _interpolator(self) -> _CurveInterpolator:  # type: ignore[override]
        return self.obj.interpolator

    @property
    def _meta(self) -> _CurveMeta:  # type: ignore[override]
        return self.obj.meta

    @property
    def _nodes(self) -> _CurveNodes:  # type: ignore[override]
        return self.obj.nodes

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self.obj._base_type


class Curve(_WithMutability, _WithOperations, _BaseCurve):  # type: ignore[misc]
    """
    A :class:`~rateslib.curves._BaseCurve` with DF parametrisation at given node dates with
    interpolation.

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
        :meth:`dcf()<rateslib.scheduling.dcf>` for all available options.
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

    # abcs - set by init

    _base_type: _CurveType = _CurveType.dfs
    _id: str = None  # type: ignore[assignment]
    _ad: int = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _interpolator: _CurveInterpolator = None  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, date: datetime) -> DualTypes:
        return super().__getitem__(date)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class LineCurve(_WithMutability, _WithOperations, _BaseCurve):  # type: ignore[misc]
    """
    A :class:`~rateslib.curves._BaseCurve` with value parametrisation at given node dates with
    interpolation.

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
        :meth:`dcf()<rateslib.scheduling.dcf>` for all available options.
    convention : str, optional, set by Default
        The convention of the curve for determining rates. Please see
        :meth:`dcf()<rateslib.scheduling.dcf>` for all available options.
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

    # abcs - set by init

    _base_type: _CurveType = _CurveType.values
    _id: str = None  # type: ignore[assignment]
    _ad: int = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _interpolator: _CurveInterpolator = None  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, date: datetime) -> DualTypes:
        return super().__getitem__(date)


class CompositeCurve(_WithOperations, _BaseCurve):
    """
    A dynamic composition of a sequence of other :class:`~rateslib.curves._BaseCurve`.

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
    _composite_scalars: list[float | Dual | Dual2 | Variable]

    # abcs - set by init

    _base_type: _CurveType = None  # type: ignore[assignment]
    _id: str = None  # type: ignore[assignment]
    _ad: int = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _interpolator: _CurveInterpolator = None  # type: ignore[assignment]

    @_new_state_post
    @_clear_cache_post
    def __init__(
        self,
        curves: list[_BaseCurve] | tuple[_BaseCurve, ...],
        id: str_ = NoInput(0),  # noqa: A002
        _no_validation: bool = False,
    ) -> None:
        self._id = _drb(super()._id, id)
        self.curves = tuple(curves)

        nodes_proxy: dict[datetime, DualTypes] = dict.fromkeys(self.curves[0].nodes.keys, 0.0)
        self._nodes = _CurveNodes(nodes_proxy)
        self._base_type = curves[0]._base_type
        self._meta = replace(self.curves[0].meta)

        if _no_validation:
            pass
        else:
            _validate_composited_curve_collection(self, self.curves, False)
        self._composite_scalars = [1.0] * len(self.curves)
        self._ad = max(_._ad for _ in self.curves)

    @property
    @_validate_states  # this ensures that the _meta attribute is updated if the curve state changes
    def meta(self) -> _CurveMeta:
        return self._meta

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
            for scalar, curve in zip(self._composite_scalars, self.curves, strict=False):
                avg_rate = ((1.0 / curve[date]) ** (1.0 / n) - 1) / d
                total_rate += avg_rate * scalar  # type: ignore[assignment]
            ret = 1.0 / (1 + total_rate * d) ** n
            return self._cached_value(date, ret)

        else:  # self._base_type == _CurveType.values:
            # will return a composited rate
            _ = 0.0
            for scalar, curve in zip(self._composite_scalars, self.curves, strict=False):
                _ += curve[date] * scalar
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
            curve._set_ad_order(order)

    # Mutation

    def _validate_state(self) -> None:
        if self._do_not_validate:
            return None
        if self._state != self._get_composited_state():
            # re-reference meta preserving own collateral status
            self._meta = replace(self.curves[0].meta, _collateral=self._meta.collateral)
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    def _get_composited_state(self) -> int:
        _: int = hash(sum(curve._state for curve in self.curves))
        return _


class MultiCsaCurve(_WithOperations, _BaseCurve):
    """
    A dynamic composition of a sequence of other :class:`~rateslib.curves._BaseCurve`.

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

    _mutable_by_association = True
    _do_not_validate = False

    # abcs - set by init

    _base_type: _CurveType = None  # type: ignore[assignment]
    _id: str = None  # type: ignore[assignment]
    _ad: int = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _interpolator: _CurveInterpolator = None  # type: ignore[assignment]

    @property
    @_validate_states  # this ensures that the _meta attribute is updated if the curve state changes
    def meta(self) -> _CurveMeta:
        return self._meta

    @_new_state_post
    @_clear_cache_post
    def __init__(
        self,
        curves: list[_BaseCurve] | tuple[_BaseCurve, ...],
        id: str | NoInput = NoInput(0),  # noqa: A002
    ) -> None:
        self._id = _drb(super()._id, id)
        self.curves = tuple(curves)
        nodes_proxy: dict[datetime, DualTypes] = dict.fromkeys(self.curves[0].nodes.keys, 0.0)
        self._nodes = _CurveNodes(nodes_proxy)
        self._base_type = curves[0]._base_type
        self._meta = replace(self.curves[0].meta)
        _validate_composited_curve_collection(self, self.curves, True)
        self._ad = max(_._ad for _ in self.curves)

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
            mins = defaults.multi_csa_min_step
            maxs = defaults.multi_csa_max_step
            return min(max(step, mins), maxs)

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
                step = defaults.multi_csa_max_step
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
            curve._set_ad_order(order)

    # Mutation

    def _validate_state(self) -> None:
        if self._do_not_validate:
            return None
        if self._state != self._get_composited_state():
            # re-reference meta preserving own collateral status
            self._meta = replace(self.curves[0].meta, _collateral=self._meta.collateral)
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    def _get_composited_state(self) -> int:
        _: int = hash(sum(curve._state for curve in self.curves))
        return _


def _validate_composited_curve_collection(
    obj: _BaseCurve, curves: tuple[_BaseCurve, ...], force_dfs: bool
) -> None:
    """Perform checks to ensure CompositeCurve can exist"""
    _base_type = curves[0]._base_type

    if force_dfs and _base_type != _CurveType.dfs:
        raise TypeError(f"{type(obj).__name__} must use discount factors, i.e have _CurveType.dfs.")

    if not all(_._base_type == _base_type for _ in curves):
        # then at least one curve is value based and one is DF based
        raise TypeError(f"{type(obj).__name__} can only contain curves of the same type.")

    ini_dates = [_.nodes.initial for _ in curves]
    if not all(_ == ini_dates[0] for _ in ini_dates[1:]):
        raise ValueError(f"`curves` must share the same initial node date, got {ini_dates}")

    # if type(self) is not MultiCsaCurve:  # for multi_csa DF curve do not check calendars
    #     self._check_meta_attribute("calendar")

    if _base_type == _CurveType.dfs:
        _check_meta_attribute(curves, "modifier")
        _check_meta_attribute(curves, "convention")
        _check_meta_attribute(curves, "calendar")
        # self._check_meta_attribute("collateral")  # not used due to inconsistent labelling

    _ad = [_._ad for _ in curves]
    if 1 in _ad and 2 in _ad:
        raise TypeError(
            f"{type(obj).__name__} cannot composite curves of AD order 1 and 2.\n"
            "Either downcast curves using `curve._set_ad_order(1)`.\n"
            "Or upcast curves using `curve._set_ad_order(2)`.\n"
        )


def _check_meta_attribute(curves: tuple[_BaseCurve, ...], attr: str) -> None:
    """Ensure attributes are the same across curve collection"""
    attrs = [getattr(_.meta, attr, None) for _ in curves]
    if not all(_ == attrs[0] for _ in attrs[1:]):
        raise ValueError(
            f"Cannot composite curves with different attributes, got for "
            f"'{attr}': {[getattr(_.meta, attr, None) for _ in curves]},",
        )


class ProxyCurve(_WithOperations, _BaseCurve):
    """
    A :class:`~rateslib.curves._BaseCurve` which returns dynamic DFs from an
    :class:`~rateslib.fx.FXForwards` object and FX parity.

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

    _mutable_by_association = True
    _do_not_validate = False

    # abcs

    _base_type: _CurveType = None  # type: ignore[assignment]
    _interpolator: _ProxyCurveInterpolator = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _id: str = None  # type: ignore[assignment]

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.interpolator.fx_forwards._ad

    @property
    def interpolator(self) -> _ProxyCurveInterpolator:  # type: ignore[override]
        """An instance of :class:`~rateslib.curves.utils._ProxyCurveInterpolator`."""
        return self._interpolator

    @property
    @_validate_states  # this ensures that the _meta attribute is updated if the curve state changes
    def meta(self) -> _CurveMeta:
        return self._meta

    @_new_state_post
    @_clear_cache_post
    def __init__(
        self,
        cashflow: str,
        collateral: str,
        fx_forwards: FXForwards,
        id: str_ = NoInput(0),  # noqa: A002
    ):
        self._interpolator = _ProxyCurveInterpolator(
            _fx_forwards=fx_forwards, _cash=cashflow.lower(), _collateral=collateral.lower()
        )
        self._id = _drb(super()._id, id)
        self._base_type = fx_forwards.fx_curves[self.interpolator.cash_pair]._base_type
        self._meta = replace(
            self.interpolator.fx_forwards.fx_curves[self.interpolator.cash_pair].meta,
            _collateral=collateral.lower(),
        )
        # CurveNodes attached for date attribution
        self._nodes = _CurveNodes(
            {
                fx_forwards.immediate: 0.0,
                fx_forwards.fx_curves[self.interpolator.cash_pair].nodes.final: 0.0,
            }
        )

    @_validate_states
    @_no_interior_validation
    def __getitem__(self, date: datetime) -> DualTypes:
        _1: DualTypes = self.interpolator.fx_forwards.rate(self.interpolator.pair, date)
        _2: DualTypes = self.interpolator.fx_forwards.fx_rates_immediate._fx_array_el(
            self.interpolator.cash_index, self.interpolator.collateral_index
        )
        _3: DualTypes = self.interpolator.fx_forwards.fx_curves[self.interpolator.collateral_pair][
            date
        ]
        return _1 / _2 * _3

    def _set_ad_order(self, order: int) -> None:
        return self.interpolator.fx_forwards._set_ad_order(order)

    def _validate_state(self) -> None:
        """Used by 'mutable by association' objects to evaluate if their own record of
        associated objects states matches the current state of those objects.

        Mutable by update objects have no concept of state validation, they simply maintain
        a *state* id.
        """
        self.interpolator.fx_forwards._validate_state()  # validate the state of sub-object
        if self._state != self._get_composited_state():
            # re-reference meta preserving own collateral status
            self._meta = replace(
                self.interpolator.fx_forwards.fx_curves[self.interpolator.cash_pair].meta,
                _collateral=self._meta.collateral,
            )
            # If any of the associated curves have been mutated then the cache is invalidated
            self._clear_cache()
            self._set_new_state()

    def _get_composited_state(self) -> int:
        return self.interpolator.fx_forwards._state


class CreditImpliedCurve(_WithOperations, _BaseCurve):
    """
    Imply a :class:`~rateslib.curves._BaseCurve` from credit components.

    .. warning::

       This class is in **beta** status as of v2.1.0

    Parameters
    ----------
    risk_free: _BaseCurve, optional
        The known risk free curve. If not given will be the implied curve.
    credit: _BaseCurve, optional
        The known credit curve.  If not given will be the implied curve.
    hazard: _BaseCurve, optional
        The known hazard curve. If not given will be the implied curve.

    Notes
    -----
    A *risk free*, *credit* or *hazard* curve will be implied from the other known, provided
    curves.

    This class is a wrapper for a :class:`~rateslib.curves.CompositeCurve` where the two known
    curves are added and multiplied by the appropriate recovery rate, obtained from the
    :class:`~rateslib.curves._CurveMeta` (either from the
    ``hazard`` curve or the ``credit`` curve in that order of precedence) to derive the third.

    In traditional papers, such as *Duffie and Singleton (1999)*, the *credit* DF is expressed
    relative to a *risk free* and *hazard* process. I.e.

    .. math::

       exp \\left ( \\int_0^T -r_f(t) - (1-R)\\lambda(t) .dt \\right ) = exp \\left ( \\int_0^T -r_c(t) .dt \\right )

    where :math:`r_f` is the instantaneous risk free rate, :math:`r_c` the instantaneous credit rate
    and :math:`\\lambda` the hazard intensity process.

    In an approximation *rateslib* converts these to discrete overnight rate equivalents and implies
    the curves as follows under rate vector addition:

    - **Credit curve rates**: :math:`r_f(t) + (1-R)\\lambda(t)`
    - **Hazard curve rates**: :math:`\\frac{r_c(t) - r_f(t)}{1-R}`
    - **Risk free rates**: :math:`r_c(t) - (1-R)\\lambda(t)`

    Example
    -------
    Given the following **risk free** curve and **hazard** curve, a **credit** curve is implied.

    .. ipython:: python

       from rateslib.curves import CreditImpliedCurve

       risk_free = Curve(
           nodes={dt(2000, 1, 1): 1.0, dt(2000, 9, 1): 0.98, dt(2001, 4, 1): 0.95, dt(2002, 1, 1): 0.92},
           interpolation="spline",
       )
       hazard = Curve(
           nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.95},
           credit_recovery_rate=0.25,
       )
       credit = CreditImpliedCurve(risk_free=risk_free, hazard=hazard)
       risk_free.plot("1b", comparators=[hazard, credit], labels=["risk free", "hazard", "credit"])

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2000, 9, 1): 0.98, dt(2001, 4, 1): 0.95, dt(2002, 1, 1): 0.92}, interpolation="spline")
       hazard = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.95}, credit_recovery_rate=0.25)
       credit = CreditImpliedCurve(risk_free=risk_free, hazard=hazard)
       fig, ax, line = risk_free.plot("1b", comparators=[hazard, credit], labels=["risk free", "hazard", "credit"])
       plt.show()
       plt.close()

    These associations are dynamic so changes to any of the curves will naturally update the
    :class:`~rateslib.curves.CreditImpliedCurve`.

    .. ipython:: python

       hazard.update_meta("credit_recovery_rate", 0.90)
       risk_free.plot("1b", comparators=[hazard, credit], labels=["risk free", "hazard", "credit"])

    .. plot::

       from rateslib.curves import *
       import matplotlib.pyplot as plt
       from datetime import datetime as dt
       risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2000, 9, 1): 0.98, dt(2001, 4, 1): 0.95, dt(2002, 1, 1): 0.92}, interpolation="spline")
       hazard = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.95}, credit_recovery_rate=0.25)
       credit = CreditImpliedCurve(risk_free=risk_free, hazard=hazard)
       hazard.update_meta("credit_recovery_rate", 0.90)
       fig, ax, line = risk_free.plot("1b", comparators=[hazard, credit], labels=["risk free", "hazard", "credit"])
       plt.show()
       plt.close()

    """  # noqa: E501

    _mutable_by_association = True
    _do_not_validate = False
    _obj: CompositeCurve

    # abcs

    _meta: _CurveMeta = None  # type: ignore[assignment]
    _interpolator: _CurveInterpolator = None  # type: ignore[assignment]

    @property
    def _base_type(self) -> _CurveType:  # type: ignore[override]
        return self.obj._base_type

    @property
    def _id(self) -> str:
        return self.obj.id

    @property
    def _ad(self) -> int:  # type: ignore[override]
        return self.obj.ad

    @_new_state_post
    @_clear_cache_post
    def __init__(
        self,
        risk_free: Curve | NoInput = NoInput(0),
        credit: Curve | NoInput = NoInput(0),
        hazard: Curve | NoInput = NoInput(0),
        id: str_ = NoInput(0),  # noqa: A002
    ) -> None:
        if sum([isinstance(_, NoInput) for _ in [risk_free, credit, hazard]]) != 1:
            raise ValueError(
                "One, and only one, curve must be NoInput in order to be a CreditImpliedCurve."
            )
        elif not isinstance(hazard, NoInput) and not isinstance(credit, NoInput):
            self._implied = _CreditImpliedType.risk_free
            self._obj = CompositeCurve(curves=[hazard, credit], id=id)
        elif not isinstance(hazard, NoInput) and not isinstance(risk_free, NoInput):
            self._implied = _CreditImpliedType.credit
            self._obj = CompositeCurve(curves=[hazard, risk_free], id=id)
        else:  # not isinstance(credit, NoInput) and not isinstance(risk_free, NoInput):
            self._implied = _CreditImpliedType.hazard
            self._obj = CompositeCurve(curves=[credit, risk_free], id=id)  # type: ignore[list-item]
        self._meta = replace(self._obj.meta)

    @_validate_states
    @_no_interior_validation
    def __getitem__(self, date: datetime) -> DualTypes:
        self.obj._composite_scalars = self._composite_scalars()
        return self.obj.__getitem__(date)

    def _set_ad_order(self, order: int) -> None:
        return self.obj._set_ad_order(order)

    @property
    def obj(self) -> CompositeCurve:
        """The wrapped :class:`~rateslib.curves.CompositeCurve` for making calculations."""
        return self._obj

    @property
    @_validate_states  # this ensures that the _meta attribute is updated if the curve state changes
    def meta(self) -> _CurveMeta:
        """An instance of :class:`~rateslib.curves._CurveMeta`."""
        return self._meta

    @property
    def _nodes(self) -> _CurveNodes:  # type: ignore[override]
        return self.obj.nodes

    def _composite_scalars(self) -> list[float | Dual | Dual2 | Variable]:
        lr = 1.0 - self.meta.credit_recovery_rate
        if self._implied == _CreditImpliedType.credit:
            return [lr, 1.0]
        elif self._implied == _CreditImpliedType.hazard:
            return [1.0 / lr, -1.0 / lr]
        else:
            return [-lr, 1.0]

    def _get_composited_state(self) -> int:
        # return the state of the CompositeCurve
        return self._obj._state

    def _validate_state(self) -> None:
        """Used by 'mutable by association' objects to evaluate if their own record of
        associated objects states matches the current state of those objects.

        Mutable by update objects have no concept of state validation, they simply maintain
        a *state* id.
        """
        if self._do_not_validate:
            return None

        self.obj._validate_state()  # validate the obj state in case one its sub components changed
        if self._state != self._get_composited_state():
            self._clear_cache()  # CreditImpliedCurve has no cache but future proofing here
            self._set_new_state()
            self._meta = replace(
                self._obj.meta,
                _collateral=self._meta.collateral,
                _credit_recovery_rate=self._obj._meta.credit_recovery_rate,
                _credit_discretization=self._obj._meta.credit_discretization,
            )
            self._obj._composite_scalars = self._composite_scalars()


def index_value(
    index_lag: int,
    index_method: str,
    index_fixings: Series[DualTypes] | DualTypes | NoInput = NoInput(0),  # type: ignore[type-var]
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
                index_fixings,
                index_date,
            )
        else:
            return _index_value_from_mixed_series_and_curve(
                index_lag,
                index_method,
                index_fixings,
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
