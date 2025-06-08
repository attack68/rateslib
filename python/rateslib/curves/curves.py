from __future__ import annotations

import json
import pickle
import warnings
from calendar import monthrange
from dataclasses import replace
from datetime import datetime, timedelta
from math import comb, prod
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import uuid4

import numpy as np
from pandas import Series
from pytz import UTC

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.calendars.rs import get_calendar
from rateslib.curves.interpolation import InterpolationFunction
from rateslib.curves.utils import (
    _CurveInterpolator,
    _CurveMeta,
    _CurveNodes,
    _CurveType,
    _ProxyCurveInterpolator,
)
from rateslib.default import (
    NoInput,
    PlotOutput,
    _drb,
    plot,
)
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    set_order_convert,
)
from rateslib.dual.utils import _dual_float, _get_order_of
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _no_interior_validation,
    _validate_states,
    _WithCache,
    _WithState,
)
from rateslib.rs import Modifier

if TYPE_CHECKING:
    from rateslib.typing import (
        Arr1dF64,
        Arr1dObj,
        CalInput,
        CurveOption_,
        FXForwards,
        Number,
        datetime_,
        float_,
        int_,
        str_,
    )
DualTypes: TypeAlias = (
    "Dual | Dual2 | Variable | float"  # required for non-cyclic import on _WithCache
)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Curve(_WithState, _WithCache[datetime, DualTypes]):
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

    _ad: int
    _id: str
    _meta: _CurveMeta
    _interpolator: _CurveInterpolator
    _nodes: _CurveNodes

    @_new_state_post
    def __init__(  # type: ignore[no-untyped-def]
        self,
        nodes: dict[datetime, DualTypes],
        *,
        interpolation: str | InterpolationFunction | NoInput = NoInput(0),
        t: list[datetime] | NoInput = NoInput(0),
        endpoints: str | tuple[str, str] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        convention: str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        ad: int = 0,
        index_base: Variable | float_ = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        collateral: str_ = NoInput(0),
        credit_discretization: int_ = NoInput(0),
        credit_recovery_rate: Variable | float_ = NoInput(0),
        **kwargs,
    ) -> None:
        self._id = _drb(uuid4().hex[:5], id)  # 1 in a million clash

        # Parameters for the rate/values derivation
        self._meta = _CurveMeta(
            _calendar=get_calendar(calendar),
            _convention=_drb(defaults.convention, convention).lower(),
            _modifier=_drb(defaults.modifier, modifier).upper(),
            _index_base=index_base,
            _index_lag=_drb(defaults.index_lag_curve, index_lag),
            _collateral=_drb(None, collateral),
            _credit_discretization=_drb(
                defaults.cds_protection_discretization, credit_discretization
            ),
            _credit_recovery_rate=_drb(defaults.cds_recovery_rate, credit_recovery_rate),
        )
        self._nodes = _CurveNodes(nodes)
        self._interpolator = _CurveInterpolator(
            local=interpolation,
            t=t,
            endpoints=self.__set_endpoints__(_drb(defaults.endpoints, endpoints)),
            node_dates=self.nodes.keys,
            convention=self.meta.convention,
            curve_type=self._base_type,
        )
        self._set_ad_order(order=ad)  # will also clear and initialise the cache

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Curve*."""
        return self._ad

    @property
    def meta(self) -> _CurveMeta:
        """An instance of :class:`~rateslib.curves._CurveMeta`."""
        return self._meta

    @property
    def id(self) -> str:
        """A str identifier to name the *Curve* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def nodes(self) -> _CurveNodes:
        """An instance of :class:`~rateslib.curves._CurveNodes`."""
        return self._nodes

    @property
    def _n(self) -> int:
        """The number of pricing parameters of the *Curve*."""
        return self.nodes.n

    @property
    def interpolator(self) -> _CurveInterpolator:
        """An instance of :class:`~rateslib.curves._CurveInterpolator`."""
        return self._interpolator

    def __set_endpoints__(self, endpoints: str | tuple[str, str]) -> tuple[str, str]:
        if isinstance(endpoints, str):
            return (endpoints.lower(), endpoints.lower())
        else:
            return (endpoints[0].lower(), endpoints[1].lower())

    def __eq__(self, other: Any) -> bool:
        """Test two curves are identical"""
        if type(self) is not type(other):
            return False
        attrs = [attr for attr in dir(self) if attr[:1] != "_"]
        for attr in attrs:
            if callable(getattr(self, attr, None)):
                continue
            elif getattr(self, attr, None) != getattr(other, attr, None):
                return False
        return True

    def __getitem__(self, date: datetime) -> DualTypes:
        if defaults.curve_caching and date in self._cache:
            return self._cache[date]

        if date < self.nodes.initial:
            return 0.0

        if self.interpolator.spline is None or date < self.interpolator.spline.t[0]:
            val = self.interpolator.local_func(date, self)
        else:
            date_posix = date.replace(tzinfo=UTC).timestamp()
            if date > self.interpolator.spline.t[-1]:
                warnings.warn(
                    "Evaluating points on a curve beyond the endpoint of the basic "
                    "spline interval is undefined.\n"
                    f"date: {date.strftime('%Y-%m-%d')}, spline end: "
                    f"{self.interpolator.spline.t[-1].strftime('%Y-%m-%d')}",
                    UserWarning,
                )
            if self._base_type == _CurveType.dfs:
                val = dual_exp(self.interpolator.spline.spline.ppev_single(date_posix))  # type: ignore[union-attr]
            else:  #  self._base_type == _CurveType.values:
                val = self.interpolator.spline.spline.ppev_single(date_posix)  # type: ignore[union-attr]

        return self._cached_value(date, val)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def rate(
        self,
        effective: datetime,
        termination: datetime | str | NoInput,
        modifier: str | NoInput = NoInput(1),
        # calendar: CalInput = NoInput(0),
        # convention: Optional[str] = None,
        float_spread: float | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
    ) -> DualTypes | None:
        """
        Calculate the rate on the `Curve` using DFs.

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
        float_spread : float, optional
            A float spread can be added to the rate in certain cases.
        spread_compound_method : str in {"none_simple", "isda_compounding"}
            The method if adding a float spread.
            If *"none_simple"* is used this results in an exact calculation.
            If *"isda_compounding"* or *"isda_flat_compounding"* is used this results
            in an approximation.

        Returns
        -------
        Dual, Dual2 or float

        Notes
        -----
        Calculating rates from a curve implies that the conventions attached to the
        specific index, e.g. USD SOFR, or GBP SONIA, are applicable and these should
        be set at initialisation of the ``Curve``. Thus, the convention used to
        calculate the ``rate`` is taken from the ``Curve`` from which ``rate``
        is called.

        ``modifier`` is only used if a tenor is given as the termination.

        Major indexes, such as legacy IBORs, and modern RFRs typically use a
        ``convention`` which is either `"Act365F"` or `"Act360"`. These conventions
        do not need additional parameters, such as the `termination` of a leg,
        the `frequency` or a leg or whether it is a `stub` to calculate a DCF.

        **Adding Floating Spreads**

        An optimised method for adding floating spreads to a curve rate is provided.
        This is quite restrictive and mainly used internally to facilitate other parts
        of the library.

        - When ``spread_compound_method`` is *"none_simple"* the spread is a simple
          linear addition.
        - When using *"isda_compounding"* or *"isda_flat_compounding"* the curve is
          assumed to be comprised of RFR
          rates and an approximation is used to derive to total rate.

        Examples
        --------

        .. ipython:: python

            curve_act365f = Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2022, 2, 1): 0.98,
                    dt(2022, 3, 1): 0.978,
                },
                convention='Act365F'
            )
            curve_act365f.rate(dt(2022, 2, 1), dt(2022, 3, 1))

        Using a different convention will result in a different rate:

        .. ipython:: python

            curve_act360 = Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2022, 2, 1): 0.98,
                    dt(2022, 3, 1): 0.978,
                },
                convention='Act360'
            )
            curve_act360.rate(dt(2022, 2, 1), dt(2022, 3, 1))
        """
        try:
            _: DualTypes = self._rate_with_raise(
                effective, termination, modifier, float_spread, spread_compound_method
            )
        except ZeroDivisionError as e:
            if "effective:" not in str(e):
                return None
            raise e
        except ValueError as e:
            if "`effective` date for rate period is before" in str(e):
                return None
            raise e
        return _

    def _rate_with_raise(
        self,
        effective: datetime,
        termination: datetime | str | NoInput,
        modifier: str | NoInput = NoInput(1),
        float_spread: float | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
    ) -> DualTypes:
        modifier_ = _drb(self.meta.modifier, modifier)

        if effective < self.nodes.initial:  # Alternative solution to PR 172.
            raise ValueError(
                "`effective` date for rate period is before the initial node date of the Curve.\n"
                "If you are trying to calculate a rate for an historical FloatPeriod have you "
                "neglected to supply appropriate `fixings`?\n"
                "See Documentation > Cookbook > Working with Fixings."
            )
        if isinstance(termination, str):
            termination = add_tenor(effective, termination, modifier_, self.meta.calendar)
        elif isinstance(termination, NoInput):
            raise ValueError("`termination` must be supplied for rate of DF based Curve.")

        if termination == effective:
            raise ZeroDivisionError(f"effective: {effective}, termination: {termination}")

        df_ratio = self[effective] / self[termination]
        n_ = df_ratio - 1.0
        d_ = dcf(effective, termination, self.meta.convention, calendar=self.meta.calendar)
        _: DualTypes = n_ / d_ * 100

        if not isinstance(float_spread, NoInput) and abs(float_spread) > 1e-9:
            if spread_compound_method == "none_simple":
                return _ + float_spread / 100
            elif spread_compound_method == "isda_compounding":
                # this provides an approximated rate
                r_bar, d, n = average_rate(effective, termination, self.meta.convention, _, d_)
                _ = ((1 + (r_bar + float_spread / 100) / 100 * d) ** n - 1) / (n * d)
                return 100 * _
            elif spread_compound_method == "isda_flat_compounding":
                # this provides an approximated rate
                r_bar, d, n = average_rate(effective, termination, self.meta.convention, _, d_)
                rd = r_bar / 100 * d
                _ = (
                    (r_bar + float_spread / 100)
                    / n
                    * (comb(int(n), 1) + comb(int(n), 2) * rd + comb(int(n), 3) * rd**2)
                )
                return _
            else:
                raise ValueError(
                    "Must supply a valid `spread_compound_method`, when `float_spread` "
                    " is not `None`.",
                )

        return _

    def shift(
        self,
        spread: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
        collateral: str_ = NoInput(0),
        composite: bool = True,
    ) -> Curve:
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
        collateral: str, optional
            Designate a collateral tag for the curve which is used by other methods.
        composite: bool, optional
            If True will return a CompositeCurve that adds a flat curve to the existing curve.
            This results in slower calculations but the curve will maintain a dynamic
            association with the underlying curve and will change if the underlying curve changes.

        Returns
        -------
        CompositeCurve or Self

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
        return self._shift(spread, id, collateral, composite, False)

    def _shift(
        self,
        spread: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
        collateral: str_ = NoInput(0),
        composite: bool = True,
        _no_validation: bool = False,
    ) -> Curve:
        """
        Executes the `Curve.shift` method.

        ``_no_validation`` is a performance enhancement to speed up a CompositeCurve init.
        """
        start, end = self.nodes.initial, self.nodes.final

        dcf_ = dcf(start, end, self.meta.convention, calendar=self.meta.calendar)
        _, d, n = average_rate(start, end, self.meta.convention, 0.0, dcf_)
        if self._base_type == _CurveType.dfs:
            shifted = Curve(
                nodes={start: 1.0, end: 1.0 / (1 + d * spread / 10000) ** n},
                convention=self.meta.convention,
                calendar=self.meta.calendar,
                modifier=self.meta.modifier,
                interpolation="log_linear",
                index_base=self.meta.index_base,
                index_lag=self.meta.index_lag,
                ad=_get_order_of(spread),
            )
        else:  # base type is values: LineCurve
            shifted = LineCurve(
                nodes={start: spread / 100.0, end: spread / 100.0},
                convention=self.meta.convention,
                calendar=self.meta.calendar,
                modifier=self.meta.modifier,
                interpolation="flat_backward",
                ad=_get_order_of(spread),
            )

        crv: CompositeCurve = CompositeCurve(
            curves=[self, shifted], id=id, _no_validation=_no_validation
        )
        crv._meta = replace(crv._meta, _collateral=_drb(None, collateral))

        if not composite:
            if self._base_type == _CurveType.dfs:
                CurveClass = Curve
                kwargs = dict(index_base=self.meta.index_base, index_lag=self.meta.index_lag)
            else:
                CurveClass = LineCurve
                kwargs = {}

            spl = self.interpolator.spline
            return CurveClass(
                nodes={k: crv[k] for k in self.nodes.nodes},
                convention=self.meta.convention,
                calendar=self.meta.calendar,
                interpolation=self.interpolator.local,
                t=NoInput(0) if spl is None else spl.t,
                c=NoInput(0),  # call csolve on init
                endpoints=NoInput(0) if spl is None else spl.endpoints,
                modifier=self.meta.modifier,
                ad=crv.ad,
                **kwargs,  # type: ignore[arg-type]
            )
        else:
            return crv

    def _translate_nodes(self, start: datetime) -> dict[datetime, DualTypes]:
        scalar = 1 / self[start]
        new_nodes = {k: scalar * v for k, v in self.nodes.nodes.items()}

        # re-organise the nodes on the new curve
        del new_nodes[self.nodes.initial]
        flag, i = (start >= self.nodes.keys[1]), 1
        while flag:
            del new_nodes[self.nodes.keys[i]]
            flag, i = (start >= self.nodes.keys[i + 1]), i + 1

        new_nodes = {start: 1.0, **new_nodes}
        return new_nodes

    def translate(self, start: datetime, t: bool = False) -> Curve:
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
        if start <= self.nodes.initial:
            raise ValueError("Cannot translate into the past. Review the docs.")

        new_nodes: dict[datetime, DualTypes] = self._translate_nodes(start)

        # re-organise the t-knot sequence
        if self.interpolator.spline is None:
            new_t: list[datetime] | NoInput = NoInput(0)
            new_endpoints: tuple[str, str] | NoInput = NoInput(0)
        else:
            new_t = self.interpolator.spline.t.copy()
            new_endpoints = self.interpolator.spline.endpoints

            if start <= new_t[0]:
                pass  # do nothing to t
            elif new_t[0] < start < new_t[4]:
                if t:
                    for i in range(4):
                        new_t[i] = start  # adjust left side of t to start
            elif new_t[4] <= start:
                raise ValueError(
                    "Cannot translate spline knots for given `start`, review the docs.",
                )

        new_curve = type(self)(
            nodes=new_nodes,
            interpolation=self.interpolator.local,
            t=new_t,
            endpoints=new_endpoints,
            modifier=self.meta.modifier,
            calendar=self.meta.calendar,
            convention=self.meta.convention,
            id=NoInput(0),
            ad=self.ad,
            index_base=NoInput(0)
            if isinstance(self.meta.index_base, NoInput)
            else _dual_float(self.index_value(start, self.meta.index_lag, "curve")),
            index_lag=self.meta.index_lag,
        )
        return new_curve

    def _roll_nodes(self, tenor: datetime, days: int) -> dict[datetime, DualTypes]:
        """
        Roll nodes by adding days to each one and scaling DF values.

        Parameters
        ----------
        tenor : datetime
            The intended roll datetime
        days : int
            The number of days between ``tenor`` and initial curve node date.

        Returns
        -------
        dict
        """
        # let regular TypeErrors raise if curve.rate is None
        on_rate = self.rate(self.nodes.initial, "1d", "NONE")
        d = 1 / 365 if self.meta.convention.upper() != "ACT360" else 1 / 360
        scalar = 1 / ((1 + on_rate * d / 100) ** days)  # type: ignore[operator]
        new_nodes = {k + timedelta(days=days): v * scalar for k, v in self.nodes.nodes.items()}
        if tenor > self.nodes.initial:
            new_nodes = {self.nodes.initial: 1.0, **new_nodes}
        return new_nodes

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def roll(self, tenor: datetime | str) -> Curve:
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
            tenor = add_tenor(self.nodes.initial, tenor, "NONE", NoInput(0))

        if tenor == self.nodes.initial:
            return self.copy()

        days = (tenor - self.nodes.initial).days
        new_nodes = self._roll_nodes(tenor, days)
        if self.interpolator.spline is not None:
            new_t: list[datetime] | NoInput = [
                _ + timedelta(days=days) for _ in self.interpolator.spline.t
            ]
            new_endpoints: tuple[str, str] | NoInput = self.interpolator.spline.endpoints
        else:
            new_t = NoInput(0)
            new_endpoints = NoInput(0)

        new_curve = type(self)(
            nodes=new_nodes,
            interpolation=self.interpolator.local,
            t=new_t,
            endpoints=new_endpoints,
            modifier=self.meta.modifier,
            calendar=self.meta.calendar,
            convention=self.meta.convention,
            id=NoInput(0),
            ad=self.ad,
            index_base=self.meta.index_base,
            index_lag=self.meta.index_lag,
        )
        if tenor > self.nodes.initial:
            return new_curve
        else:  # tenor < self.nodes.initial
            return new_curve.translate(self.nodes.initial)

    def index_value(
        self, date: datetime, index_lag: int, interpolation: str = "curve"
    ) -> DualTypes:
        """
        Calculate the accrued value of the index from the ``index_base``.

        Parameters
        ----------
        date : datetime
            The date for which the index value will be returned.
        index_lag : int
            The number of months by which to lag the index when determining the value.
        interpolation : str in {"curve", "monthly", "daily"}
            The method for returning the index value. Monthly returns the index value
            for the start of the month and daily returns a value based on the
            interpolation between nodes (which is recommended *"linear_index*) for
            :class:`InflationCurve`.

        Returns
        -------
        None, float, Dual, Dual2

        Notes
        ------
        The interpolation methods function as follows:

        - **"curve"**: will raise if the requested ``index_lag`` does not match the lag attributed
          to the *Curve*. In the case the ``index_lag`` matches, then the *index value* for any
          date is derived via the implied interpolation for the discount factors of the *Curve*.

          .. math::

             I_v(m) = \\frac{I_b}{v(m)}

        - **"monthly"**: For any date, *m*, uses the *"curve"* method having adjusted *m* in two
          ways. Firstly it deducts a number of months equal to :math:`L - L_c`, where *L* is
          the given ``index_lag`` and :math:`L_c` is the *index lag* of the *Curve*. And the day
          of the month is set to 1.

          .. math::

             &I^{monthly}_v(m) = I_v(m_adj) \\\\
             &\\text{where,} \\\\
             &m_adj = Date(Year(m), Month(m) - L + L_c, 1) \\\\

        - **"daily"**: For any date, *m*, with a given ``index_lag`` performs calendar day
          interpolation on surrounding *"monthly"* values.

          .. math::

             &I^{daily}_v(m) = I^{monthly}_v(m) + \\frac{Day(m) - 1}{n} \\left ( I^{monthly}_v(m_+) - I^{monthly}_v(m) \\right ) \\\\
             &\\text{where,} \\\\
             &m_+ = \\text{Any date in the month following, }m
             &n = \\text{Calendar days in, } Month(m)

        Examples
        --------
        The SWESTR rate, for reference value date 6th Sep 2021, was published as
        2.375% and the RFR index for that date was 100.73350964. Below we calculate
        the value that was published for the RFR index on 7th Sep 2021 by the Riksbank.

        .. ipython:: python

           index_curve = Curve(
               nodes={
                   dt(2021, 9, 6): 1.0,
                   dt(2021, 9, 7): 1 / (1 + 2.375/36000)
               },
               index_base=100.73350964,
               convention="Act360",
               index_lag=0,
           )
           index_curve.rate(dt(2021, 9, 6), "1d")
           index_curve.index_value(dt(2021, 9, 7), 0)
        """  # noqa: E501
        if isinstance(self.meta.index_base, NoInput):
            raise ValueError(
                "Curve must be initialised with an `index_base` value to derive `index_value`."
            )

        lag_months = index_lag - self.meta.index_lag
        if interpolation.lower() == "curve":
            if lag_months != 0:
                raise ValueError(
                    "'curve' interpolation can only be used with `index_value` when the Curve "
                    "`index_lag` matches the input `index_lag`."
                )
            # use traditional discount factor from Index base to determine index value.
            if date < self.nodes.initial:
                warnings.warn(
                    "The date queried on the Curve for an `ìndex_value` is prior to the "
                    "initial node on the Curve.\nThis is returned as zero and likely "
                    f"causes downstream calculation error.\ndate queried: {date}"
                    "Either providing `index_fixings` to the object or extend the Curve backwards.",
                    UserWarning,
                )
                return 0.0
                # return zero for index dates in the past
                # the proper way for instruments to deal with this is to supply i_fixings
            elif date == self.nodes.initial:
                return self.meta.index_base
            else:
                return self.meta.index_base * 1.0 / self[date]
        elif interpolation.lower() == "monthly":
            date_ = add_tenor(date, f"-{lag_months}M", "none", NoInput(0), 1)
            return self.index_value(date_, self.meta.index_lag, "curve")
        elif interpolation.lower() == "daily":
            n = monthrange(date.year, date.month)[1]
            date_som = datetime(date.year, date.month, 1)
            date_sonm = add_tenor(date, "1M", "none", NoInput(0), 1)
            m1 = self.index_value(date_som, index_lag, "monthly")
            m2 = self.index_value(date_sonm, index_lag, "monthly")
            return m1 + (date.day - 1) / n * (m2 - m1)
        else:
            raise ValueError(
                "`interpolation` for `index_value` must be in {'curve', 'daily', 'monthly'}."
            )

    # Plotting

    def plot_index(
        self,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[Curve] | NoInput = NoInput(0),
        difference: bool = False,
        labels: list[str] | NoInput = NoInput(0),
        interpolation: str = "curve",
    ) -> PlotOutput:
        """
        Plot given index values on a  curve.

        Parameters
        ----------
        tenor : str
            The tenor of the forward rates to plot, e.g. "1D", "3M".
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the final node of the curve minus the ``tenor``.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the initial node of the curve.
        comparators: list[Curve]
            A list of curves which to include on the same plot as comparators.
        difference : bool
            Whether to plot as comparator minus base curve or outright curve levels in
            plot. Default is `False`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        interpolation : str in {"curve", "daily", "monthly"}
            The type of index interpolation method to use.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D

        """
        comparators = _drb([], comparators)
        labels = _drb([], labels)
        if left is NoInput.blank:
            left_: datetime = self.nodes.initial
        elif isinstance(left, str):
            left_ = add_tenor(self.nodes.initial, left, "NONE", NoInput(0))
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if right is NoInput.blank:
            right_: datetime = self.nodes.final
        elif isinstance(right, str):
            right_ = add_tenor(self.nodes.initial, right, "NONE", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        points: int = (right_ - left_).days + 1
        x = [left_ + timedelta(days=i) for i in range(points)]
        rates = [self.index_value(_, self.meta.index_lag, interpolation) for _ in x]
        if not difference:
            y = [rates]
            if not isinstance(comparators, NoInput) and len(comparators) > 0:
                for comparator in comparators:
                    y.append([comparator.index_value(_, self.meta.index_lag) for _ in x])
        elif difference and (isinstance(comparators, NoInput) or len(comparators) == 0):
            raise ValueError("If `difference` is True must supply at least one `comparators`.")
        else:
            y = []
            for comparator in comparators:
                diff = [
                    comparator.index_value(_, self.meta.index_lag, interpolation) - rates[i]
                    for i, _ in enumerate(x)
                ]
                y.append(diff)
        return plot([x] * len(y), y, labels)

    def plot(
        self,
        tenor: str,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[Curve] | NoInput = NoInput(0),
        difference: bool = False,
        labels: list[str] | NoInput = NoInput(0),
    ) -> PlotOutput:
        """
        Plot given forward tenor rates from the curve. See notes.

        Parameters
        ----------
        tenor : str
            The tenor of the forward rates to plot, e.g. "1D", "3M".
        right : datetime or str, optional
            The right bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the final node of the curve minus the ``tenor``.
        left : datetime or str, optional
            The left bound of the graph. If given as str should be a tenor format
            defining a point measured from the initial node date of the curve.
            Defaults to the initial node of the curve.
        comparators: list[Curve]
            A list of curves which to include on the same plot as comparators.
        difference : bool
            Whether to plot as comparator minus base curve or outright curve levels in
            plot. Default is `False`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D

        Notes
        ------
        This function plots single-period, **simple interest** curve rates, which are defined as:

        .. math::

           1 + r d = \\frac{v_{start}}{v_{end}}

        where *d* is the day count fraction determined using the ``convention`` associated
        with the *Curve*.

        This function does **not** plot swap rates,
        which is impossible since the *Curve* object contains no information regarding the
        parameters of the *'swap'* (e.g. its *frequency* or its *convention* etc.).
        If ``tenors`` longer than one year are sought results may start to deviate from those
        one might expect. See `Issue 246 <https://github.com/attack68/rateslib/issues/246>`_.

        """
        comparators_: list[Curve] = _drb([], comparators)
        labels = _drb([], labels)
        upper_tenor = tenor.upper()
        x, y = self._plot_rates(upper_tenor, left, right)
        y_ = [y] if not difference else []
        for _, comparator in enumerate(comparators_):
            if difference:
                y_.append(
                    [
                        self._plot_diff(_x, tenor, _y, comparator)
                        for _x, _y in zip(x, y, strict=False)
                    ]
                )
            else:
                pm_ = comparator._plot_modifier(tenor)
                y_.append([comparator._plot_rate(_x, tenor, pm_) for _x in x])

        return plot([x] * len(y_), y_, labels)

    def _plot_diff(
        self, date: datetime, tenor: str, rate: DualTypes | None, comparator: Curve
    ) -> DualTypes | None:
        if rate is None:
            return None
        rate2 = comparator._plot_rate(date, tenor, comparator._plot_modifier(tenor))
        if rate2 is None:
            return None
        return rate2 - rate

    def _plot_modifier(self, upper_tenor: str) -> str:
        """If tenor is in days do not allow modified for plot purposes"""
        if "B" in upper_tenor or "D" in upper_tenor or "W" in upper_tenor:
            if "F" in self.meta.modifier:
                return "F"
            elif "P" in self.meta.modifier:
                return "P"
        return self.meta.modifier

    def _plot_rates(
        self,
        upper_tenor: str,
        left: datetime | str | NoInput,
        right: datetime | str | NoInput,
    ) -> tuple[list[datetime], list[DualTypes | None]]:
        if isinstance(left, NoInput):
            left_: datetime = self.nodes.initial
        elif isinstance(left, str):
            left_ = add_tenor(self.nodes.initial, left, "F", self.meta.calendar)
        elif isinstance(left, datetime):
            left_ = left
        else:
            raise ValueError("`left` must be supplied as datetime or tenor string.")

        if isinstance(right, NoInput):
            # pre-adjust the end date to enforce business date.
            right_: datetime = add_tenor(
                self.meta.calendar.roll(self.nodes.final, Modifier.P, False),
                "-" + upper_tenor,
                "P",
                self.meta.calendar,
            )
        elif isinstance(right, str):
            right_ = add_tenor(self.nodes.initial, right, "P", NoInput(0))
        elif isinstance(right, datetime):
            right_ = right
        else:
            raise ValueError("`right` must be supplied as datetime or tenor string.")

        dates = self.meta.calendar.cal_date_range(start=left_, end=right_)
        rates = [self._plot_rate(_, upper_tenor, self._plot_modifier(upper_tenor)) for _ in dates]
        return dates, rates

    def _plot_rate(
        self,
        effective: datetime,
        termination: str,
        modifier: str,
    ) -> DualTypes | None:
        try:
            rate = self.rate(effective, termination, modifier)
        except ValueError:
            return None
        return rate

    def _plot_fx(
        self,
        curve_foreign: Curve,
        fx_rate: float | Dual,
        fx_settlement: datetime | NoInput = NoInput(0),
        # left: datetime = None,
        # right: datetime = None,
        # points: int = None,
    ) -> PlotOutput:  # pragma: no cover
        """
        Debugging method?
        """

        def forward_fx(
            date: datetime,
            curve_domestic: Curve,
            curve_foreign: Curve,
            fx_rate: DualTypes,
            fx_settlement: datetime | NoInput,
        ) -> DualTypes:
            _: DualTypes = self[date] / curve_foreign[date]
            if not isinstance(fx_settlement, NoInput):
                _ *= curve_foreign[fx_settlement] / curve_domestic[fx_settlement]
            _ *= fx_rate
            return _

        left, right = self.nodes.initial, self.nodes.final
        points = (right - left).days
        x = [left + timedelta(days=i) for i in range(points)]
        rates = [forward_fx(_, self, curve_foreign, fx_rate, fx_settlement) for _ in x]
        return plot([x], [rates])

    # Mutation
    @_new_state_post
    @_clear_cache_post
    def csolve(self) -> None:
        """
        Solves **and sets** the coefficients, ``c``, of the :class:`PPSpline`.

        Returns
        -------
        None

        Notes
        -----
        Only impacts curves which have a knot sequence, ``t``, and a ``PPSpline``.
        Only solves if ``c`` not given at curve initialisation.

        Uses the ``spline_endpoints`` attribute on the class to determine the solving
        method.
        """
        self.interpolator._csolve(self._base_type, self.nodes, self._ad)

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(self, vector: list[DualTypes], ad: int) -> None:
        """Used to update curve values during a Solver iteration. ``ad`` in {1, 2}."""
        self._set_node_vector_direct(vector, ad)

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        """
        Change the node values to float, Dual or Dual2 based on input parameter.
        """
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
        nodes_: dict[datetime, DualTypes] = {
            k: set_order_convert(v, order, [f"{self.id}{i}"])
            for i, (k, v) in enumerate(self.nodes.nodes.items())
        }
        self._nodes = _CurveNodes(nodes_)
        self.interpolator._csolve(self._base_type, self.nodes, self._ad)

    def _set_node_vector_direct(self, vector: list[DualTypes], ad: int) -> None:
        nodes_ = self.nodes.nodes.copy()
        if ad == 0:
            if self._ini_solve == 1 and self.nodes.n > 0:
                nodes_[self.nodes.initial] = _dual_float(nodes_[self.nodes.initial])
            for i, k in enumerate(self.nodes.keys[self._ini_solve :]):
                nodes_[k] = _dual_float(vector[i])
        else:
            DualType: type[Dual | Dual2] = Dual if ad == 1 else Dual2
            DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
                ([],) if ad == 1 else ([], [])
            )
            base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(self.nodes.n)], *DualArgs)
            ident: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.eye(
                self.nodes.n, dtype=np.float64
            )

            if self._ini_solve == 1:
                # then the first node on the Curve is not updated but
                # set it as a dual type with consistent vars.
                nodes_[self.nodes.initial] = DualType.vars_from(
                    base_obj,  # type: ignore[arg-type]
                    _dual_float(nodes_[self.nodes.initial]),
                    base_obj.vars,
                    ident[0, :].tolist(),
                    *DualArgs[1:],
                )

            for i, k in enumerate(self.nodes.keys[self._ini_solve :]):
                nodes_[k] = DualType.vars_from(
                    base_obj,  # type: ignore[arg-type]
                    _dual_float(vector[i]),
                    base_obj.vars,
                    ident[i + self._ini_solve, :].tolist(),
                    *DualArgs[1:],
                )
        self._ad = ad
        self._nodes = _CurveNodes(nodes_)
        self.interpolator._csolve(self._base_type, self.nodes, self._ad)

    @_new_state_post
    @_clear_cache_post
    def update(
        self,
        nodes: dict[datetime, DualTypes] | NoInput = NoInput(0),
    ) -> None:
        """
        Update a curves nodes with new, manually input values.

        For arguments see :class:`~rateslib.curves.curves.Curve`. Any value not given will not
        change the underlying *Curve*.

        Parameters
        ----------
        nodes: dict[datetime, DualTypes], optional
            New nodes to assign to the curve.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        if not isinstance(nodes, NoInput):
            self._nodes = _CurveNodes(nodes)

        self.interpolator._csolve(self._base_type, self.nodes, self._ad)

    @_new_state_post
    @_clear_cache_post
    def update_node(self, key: datetime, value: DualTypes) -> None:
        """
        Update a single node value on the *Curve*.

        Parameters
        ----------
        key: datetime
            The node date to update. Must exist in ``nodes``.
        value: float, Dual, Dual2, Variable
            Value to update on the *Curve*.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        if key not in self.nodes.nodes:
            raise KeyError("`key` is not in *Curve* ``nodes``.")

        nodes_ = self.nodes.nodes.copy()
        nodes_[key] = value
        self._nodes = _CurveNodes(nodes_)
        self.interpolator._csolve(self._base_type, self.nodes, self._ad)

    @_new_state_post
    @_clear_cache_post
    def update_meta(self, key: datetime, value: Any) -> None:
        """
        Update a single meta value on the *Curve*.

        Parameters
        ----------
        key: datetime
            The meta descriptor to update. Must be a documented attribute of
            :class:`~rateslib.curves.utils._CurveMeta`.
        value: Any
            Value to update on the *Curve*.

        Returns
        -------
        None
        """
        _key = f"_{key}"
        self._meta = replace(self._meta, **{_key: value})

    # Solver interaction

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(list(self.nodes.nodes.values())[self._ini_solve :])

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self.id}{i}" for i in range(self._ini_solve, self.nodes.n))

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

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__}:{self.id} at {hex(id(self))}>"

    def copy(self) -> Curve:
        """
        Create an identical copy of the curve object.

        Returns
        -------
        Curve or LineCurve
        """
        ret: Curve = pickle.loads(pickle.dumps(self, -1))  # noqa: S301
        return ret

        # from rateslib.serialization import from_json
        # return from_json(self.to_json())


class LineCurve(Curve):
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

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def rate(
        self,
        effective: datetime,
        *args: Any,
        **kwargs: Any,
    ) -> DualTypes | None:
        """
        Return the curve value for a given date.

        Note `LineCurve` s do not determine interest rates via DFs therefore do not
        have the concept of tenors or termination dates - the rate is simply the value
        attributed to that date on the curve.

        Parameters
        ----------
        effective : datetime
            The date of the curve value, which will be interpolated if necessary.

        Returns
        -------
        float, Dual, or Dual2
        """
        try:
            _: DualTypes = self._rate_with_raise(effective, *args, **kwargs)
        except ValueError as e:
            if "`effective` before initial LineCurve date" in str(e):
                return None
            raise e
        return _

    def _rate_with_raise(
        self,
        effective: datetime,
        *args: Any,
        **kwargs: Any,
    ) -> DualTypes:
        if effective < self.nodes.initial:  # Alternative solution to PR 172.
            raise ValueError("`effective` before initial LineCurve date.")
        return self[effective]

    def _translate_nodes(self, start: datetime) -> dict[datetime, DualTypes]:
        new_nodes = self.nodes.nodes.copy()

        # re-organise the nodes on the new curve
        del new_nodes[self.nodes.initial]
        flag, i = (start >= self.nodes.keys[1]), 1
        while flag:
            del new_nodes[self.nodes.keys[i]]
            flag, i = (start >= self.nodes.keys[i + 1]), i + 1

        new_nodes = {start: self[start], **new_nodes}
        return new_nodes

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def translate(self, start: datetime, t: bool = False) -> Curve:
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

           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = line_curve.translate(dt(2022, 12, 1))
           line_curve.plot("1d", comparators=[translated_curve], left=dt(2022, 12, 1))

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           translated_curve = line_curve.translate(dt(2022, 12, 1))
           fig, ax, line = line_curve.plot("1d", comparators=[translated_curve], labels=["orig", "translated"], left=dt(2022, 12,1))
           plt.show()
           plt.close()

        .. ipython:: python

           line_curve.nodes
           translated_curve.nodes

        When a line curve has a cubic spline the knot dates can be preserved or
        translated with the ``t`` argument. Preserving the knot dates preserves the
        interpolation of the curve. A knot sequence for a mixed curve which begins
        after ``start`` will not be affected in either case.

        .. ipython:: python

           curve = LineCurve(
               nodes={
                   dt(2022, 1, 1): 1.5,
                   dt(2022, 2, 1): 1.6,
                   dt(2022, 3, 1): 1.4,
                   dt(2022, 4, 1): 1.2,
                   dt(2022, 5, 1): 1.1,
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
           curve = LineCurve(
               nodes={
                   dt(2022, 1, 1): 1.5,
                   dt(2022, 2, 1): 1.6,
                   dt(2022, 3, 1): 1.4,
                   dt(2022, 4, 1): 1.2,
                   dt(2022, 5, 1): 1.1,
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
        return super().translate(start, t)

    def _roll_nodes(self, tenor: datetime, days: int) -> dict[datetime, DualTypes]:
        new_nodes = {k + timedelta(days=days): v for k, v in self.nodes.nodes.items()}
        if tenor > self.nodes.initial:
            new_nodes = {self.nodes.initial: self[self.nodes.initial], **new_nodes}
        return new_nodes

    def roll(self, tenor: datetime | str) -> Curve:
        """
        Create a new curve with its shape translated in time

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

           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = line_curve.roll("6m")
           rolled_curve2 = line_curve.roll("-6m")
           line_curve.plot(
               "1d",
               comparators=[rolled_curve, rolled_curve2],
               labels=["orig", "rolled", "rolled2"],
               right=dt(2026, 7, 1)
           )

        .. plot::

           from rateslib.curves import *
           import matplotlib.pyplot as plt
           from datetime import datetime as dt
           line_curve = LineCurve(
               nodes = {
                   dt(2022, 1, 1): 1.7,
                   dt(2023, 1, 1): 1.65,
                   dt(2024, 1, 1): 1.4,
                   dt(2025, 1, 1): 1.3,
                   dt(2026, 1, 1): 1.25,
                   dt(2027, 1, 1): 1.35
               },
               t = [
                   dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
                   dt(2025, 1, 1),
                   dt(2026, 1, 1),
                   dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
               ],
           )
           rolled_curve = line_curve.roll("6m")
           rolled_curve2 = line_curve.roll("-6m")
           fig, ax, line = line_curve.plot("1d", comparators=[rolled_curve, rolled_curve2], labels=["orig", "rolled", "rolled2"], right=dt(2026,6,30))
           plt.show()
           plt.close()

        """  # noqa: E501
        return super().roll(tenor)


class CompositeCurve(Curve):
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
        curves: list[Curve] | tuple[Curve, ...],
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

    def shift(
        self,
        spread: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
        collateral: str_ = NoInput(0),
        composite: bool = True,
    ) -> Curve:
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        See :meth:`Curve.shift()<rateslib.curves.Curve.shift>`
        """
        if composite is False:
            raise ValueError("`composite` must be set to `True` when shifting a CompositeCurve.")
        return self._shift(spread, id, collateral, composite, False)

    @_validate_states
    def _shift(
        self,
        spread: DualTypes,
        id: str_ = NoInput(0),  # noqa: A002
        collateral: str_ = NoInput(0),
        composite: bool = True,
        _no_validation: bool = False,
    ) -> Curve:
        """
        Create a new curve by vertically adjusting the curve by a set number of basis
        points.

        See :meth:`Curve.shift()<rateslib.curves.Curve.shift>`
        """
        return super()._shift(spread, id, collateral, composite, _no_validation)

    @_validate_states
    def translate(self, start: datetime, t: bool = False) -> CompositeCurve:
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
        CompositeCurve
        """
        # cache check unnecessary since translate is constructed from up-to-date objects directly
        return CompositeCurve(curves=[curve.translate(start, t) for curve in self.curves])

    @_validate_states
    def roll(self, tenor: datetime | str) -> CompositeCurve:
        """
        Create a new curve with its shape translated in time

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
        CompositeCurve
        """
        # cache check unnecessary since roll is constructed from up-to-date objects directly
        return CompositeCurve(curves=[curve.roll(tenor) for curve in self.curves])

    @_validate_states
    def index_value(
        self, date: datetime, index_lag: int, interpolation: str = "curve"
    ) -> DualTypes:
        """
        Calculate the accrued value of the index from the ``index_base``, which is taken
        as ``index_base`` of the *first* composited curve given.

        See :meth:`Curve.index_value()<rateslib.curves.Curve.index_value>`
        """
        return super().index_value(date, index_lag, interpolation)

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

    def _get_node_vector(self) -> Arr1dObj | Arr1dF64:
        raise NotImplementedError("Instances of CompositeCurve do not have solvable variables.")

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

    # Serialization

    # Overloads

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not provide update methods.")

    def update_node(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not provide update methods.")

    def update_meta(self, key: datetime, value: DualTypes) -> None:
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not provide update methods.")

    def to_json(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not provide serialization methods.")

    def from_json(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not provide update methods.")

    def csolve(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types does not set interpolation directly.")

    def copy(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Not implemented on CompositeCurve types."""
        raise NotImplementedError("CompositeCurve types do not currently have copy function.")


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
        curves: list[Curve] | tuple[Curve, ...],
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

    @_validate_states
    # unnecessary because up-to-date objects are referred to directly
    def translate(self, start: datetime, t: bool = False) -> MultiCsaCurve:
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
        MultiCsaCurve
        """
        return MultiCsaCurve(
            curves=[curve.translate(start, t) for curve in self.curves],
            multi_csa_max_step=self.multi_csa_max_step,
            multi_csa_min_step=self.multi_csa_min_step,
        )

    @_validate_states
    # unnecessary because up-to-date objects are referred to directly
    def roll(self, tenor: datetime | str) -> MultiCsaCurve:
        """
        Create a new curve with its shape translated in time

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
        MultiCsaCurve
        """
        return MultiCsaCurve(
            curves=[curve.roll(tenor) for curve in self.curves],
            multi_csa_max_step=self.multi_csa_max_step,
            multi_csa_min_step=self.multi_csa_min_step,
        )


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

    def translate(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override] # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types currently provide no translate operation.")

    def roll(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override] # pragma: no cover
        """Not implemented on *ProxyCurve* types."""
        raise NotImplementedError("ProxyCurve types currently provide no roll operation.")

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


def average_rate(
    effective: datetime,
    termination: datetime,
    convention: str,
    rate: DualTypes,
    dcf: float,
) -> tuple[Number, float, float]:
    """
    Return the geometric, 1-day, average simple rate for a given simple period rate.

    This is used for approximations usually in combination with floating periods.

    Parameters
    ----------
    effective : datetime
        The effective date of the rate.
    termination : datetime
        The termination date of the rate.
    convention : str
        The day count convention of the curve rate.
    rate : float, Dual, Dual2
        The simple period rate to decompose to average, in percentage terms, e.g. 4.00 = 4% rate.
    dcf : float
        The day count fraction of the period used to determine daily DCF.

    Returns
    -------
    tuple : The simple rate, the 1-day DCF, and the number of relevant days for the convention

    Notes
    -----
    This method operates in one of two modes to determine the value, :math:`\\bar{r}`.

    - Calendar day basis, where :math:`\\tilde{n}` is calendar days in period:

      .. math::

         1+\\tilde{n}\\bar{d}r = (1 + \\bar{d}\\bar{r})^{\\tilde{n}}

    - Business day basis (if ``convention`` is *'bus252'*), where :math:`n` is business days
      in period. *n* is approximated by a 252 business days per year rule and does not
      calculate the exact number of business days from any specific holiday calendar.

      .. math::

         1+n\\bar{d}r = (1 + \\bar{d}\\bar{r})^{n}

    :math:`\\bar{d}`, the 1-day DCF is estimated from a ``convention``. For certain conventions,
    e.g. *'act360'* and *'act365f'* this is explicit and exact, but for others, such as *'30360'*,
    this function will likely be lesser used and less accurate.
    """
    if convention.upper() == "BUS252":
        # business days are used
        n: float = dcf * 252.0
        d = 1.0 / 252.0
    else:  # calendar day mode
        n = (termination - effective).days
        d = dcf / n

    _: Number = ((1 + n * d * rate / 100) ** (1 / n) - 1) / d
    return _ * 100, d, n


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
    index_curve: Curve,
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
