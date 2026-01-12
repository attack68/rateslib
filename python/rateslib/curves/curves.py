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


from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from calendar import monthrange
from dataclasses import replace
from datetime import datetime, timedelta
from math import comb, prod
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

import numpy as np
from pandas import Series
from pytz import UTC

import rateslib.errors as err
from rateslib import defaults, fixings
from rateslib.curves.interpolation import InterpolationFunction
from rateslib.curves.utils import (
    _CreditImpliedType,
    _CurveInterpolator,
    _CurveMeta,
    _CurveNodes,
    _CurveType,
    _ProxyCurveInterpolator,
    average_rate,
)
from rateslib.data.loader import FixingMissingDataError, FixingRangeError
from rateslib.default import PlotOutput, plot
from rateslib.dual import Dual, Dual2, Variable, dual_exp, dual_log, set_order_convert
from rateslib.dual.utils import _dual_float, _get_order_of
from rateslib.enums.generics import Err, NoInput, Ok, _drb
from rateslib.enums.parameters import IndexMethod, _get_index_method
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _no_interior_validation,
    _validate_states,
    _WithCache,
    _WithState,
)
from rateslib.scheduling import Adjuster, Convention, add_tenor, dcf, get_calendar
from rateslib.scheduling.convention import _get_convention

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurveOption_,
        FXForwards,
        Number,
        Result,
        datetime_,
        float_,
        int_,
        str_,
    )

DualTypes: TypeAlias = (
    "Dual | Dual2 | Variable | float"  # required for non-cyclic import on _WithCache
)


class _WithOperations:
    """Provides automatic implementation of the curve operations required on a
    :class:`~rateslib.curves._BaseCurve`."""

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
    def roll(self, tenor: datetime | str | int, id: str_ = NoInput(0)) -> RolledCurve:  # noqa: A002
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
        _: _BaseCurve = self  # type: ignore[assignment]
        if isinstance(tenor, str):
            tenor_: datetime | int = add_tenor(_._nodes.initial, tenor, "NONE", NoInput(0))
        else:
            tenor_ = tenor

        if isinstance(tenor_, int):
            roll_days: int = tenor_
        else:
            roll_days = (tenor_ - _._nodes.initial).days

        return RolledCurve(curve=_, roll_days=roll_days, id=id)


class _BaseCurve(_WithState, _WithCache[datetime, DualTypes], _WithOperations, ABC):
    """
    An ABC defining the base methods of a *Curve*.

    Provided that the abstract base properties and methods of this class are implemented any
    custom curve can be used within *rateslib*. Often the default implementations for some of
    these, via ``super()`` are sufficient. The required base methods are:

    - ``_meta``: returns a :class:`~rateslib.curves._CurveMeta` class.
    - ``_interpolator``: returns a :class:`~rateslib.curves._CurveInterpolator` class.
    - ``_nodes``: returns a :class:`~rateslib.curves._CurveNodes` class.
    - ``_id``: returns a str representing the *Curve* id.
    - ``_ad``: returns an integer in {0, 1, 2} indicating the automatic differentiation state.
    - ``_base_type``: returns a :class:`~rateslib.curves._CurveType`.
    - ``__getitem__(date)``: returns a float, :class:`~rateslib.dual.Dual`,
      :class:`~rateslib.dual.Dual2`, or :class:`~rateslib.dual.Variable` given an input date.
    - ``_set_ad_order(ad)``: mutates the node values of the *Curve* to adopt new automatic
      differentiation states for facilitating other features, such as
      :class:`~rateslib.solver.Solver` calibration and risk sensitivity calculation.

    To automatically provide some of the operations the class
    :class:`~rateslib.curves._WithOperations` can, and is likely to always be, inherited, without
    the need for any additional implementation. In certain cases the `_base_type` will prevent
    some methods from calculating and will raise `TypeError`.

    To allow custom user curves to be calibrated by the :class:`~rateslib.solver.Solver` framework
    the :class:`~rateslib.curves._WithMutability` class can be inherited. This requires two
    additional implementation to allow a :class:`~rateslib.solver.Solver` to interact directly with
    it:

    - ``_get_node_vector()``: returns a NumPy array of the ordered node values consumed.
    - ``_get_node_vars()``: returns a tuple of ordered string variable names associated with
      each node of the *Curve*.
    - ``_set_node_vector(array)``: accepts a NumPy array of the ordered node values and sets
      these directly for the *Curve*.

    .. rubric:: Examples

    A demonstration of using this class to build a user custom *Curve* is presented at
    `Cookbook: Building Custom Curves with _BaseCurve (e.g. Nelson-Siegel) <../z_basecurve.html>`_
    """

    # Required properties

    @property
    @abstractmethod
    def _meta(self) -> _CurveMeta:
        return _CurveMeta(
            _calendar=get_calendar(NoInput(0)),
            _collateral=None,
            _convention=_get_convention(defaults.convention),
            _credit_discretization=defaults.cds_protection_discretization,
            _credit_recovery_rate=defaults.cds_recovery_rate,
            _index_base=NoInput(0),
            _index_lag=defaults.index_lag_curve,
            _modifier=defaults.modifier,
        )

    @property
    @abstractmethod
    def _interpolator(self) -> _CurveInterpolator:
        # create a default CurveInterpolator that is functionless
        # this is a placeholder obj that cannot be used for interpolation
        return _CurveInterpolator(
            local="log_linear",
            t=NoInput(0),
            endpoints=("natural", "natural"),
            node_dates=[],
            convention=defaults.convention.lower(),
            curve_type=_CurveType.dfs,
        )

    @property
    @abstractmethod
    def _nodes(self) -> _CurveNodes: ...

    @property
    @abstractmethod
    def _id(self) -> str:
        return uuid4().hex[:5]

    @property
    @abstractmethod
    def _ad(self) -> int: ...

    @property
    @abstractmethod
    def _base_type(self) -> _CurveType: ...

    # Required methods

    @abstractmethod
    def __getitem__(self, date: datetime) -> DualTypes:
        """
        The get item method for any *Curve* type will allow the inheritance of the below
        methods.
        """
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
                    f"{self.interpolator.spline.t[-1].strftime('%Y-%m-%d')}\n"
                    "This often occurs when a curve is constructed with a final node date "
                    "that aligns with the maturity of an instrument with a payment lag.\nIn the "
                    "case that the instrument has a payment lag (e.g. a SOFR swap or ESTR swap or "
                    "bond terminating on a non-business day) then a cashflow will occur after the "
                    "maturity of the instrument.\nThe solution is to ensure that the final node "
                    "date of the curve is changed to be beyond that expected payment date.",
                    UserWarning,
                )
            if self._base_type == _CurveType.dfs:
                val = dual_exp(self.interpolator.spline.spline.ppev_single(date_posix))  # type: ignore[union-attr]
            else:  # self._base_type == _CurveType.values:
                val = self.interpolator.spline.spline.ppev_single(date_posix)  # type: ignore[union-attr]

        return self._cached_value(date, val)

    @abstractmethod
    def _set_ad_order(self, order: int) -> None: ...

    # Properties

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

    # Rate Calculation

    def rate(
        self,
        effective: datetime,
        termination: datetime | str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(1),
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
                return None  # TODO (low): is this an unreachable line?
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
        if self._base_type == _CurveType.dfs:
            return self._rate_with_raise_dfs(
                effective, termination, modifier, float_spread, spread_compound_method
            )
        else:  # is _CurveType.values
            return self._rate_with_raise_values(
                effective, termination, modifier, float_spread, spread_compound_method
            )

    def _rate_with_raise_values(
        self,
        effective: datetime,
        *args: Any,
        **kwargs: Any,
    ) -> DualTypes:
        if effective < self.nodes.initial:  # Alternative solution to PR 172.
            raise ValueError(
                "`effective` date for rate period is before the initial node date of the Curve.\n"
                "If you are trying to calculate a rate for an historical FloatPeriod have you "
                "neglected to supply appropriate `fixings`?\n"
                "See Documentation > Cookbook > Working with Fixings."
            )
        return self.__getitem__(effective)

    def _rate_with_raise_dfs(
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

        df_ratio = self.__getitem__(effective) / self.__getitem__(termination)
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

    # Index Calculations

    def _try_index_value(
        self, index_date: datetime, index_lag: int, index_method: IndexMethod = IndexMethod.Curve
    ) -> Result[DualTypes]:
        if self._base_type == _CurveType.values:
            return Err(TypeError("A 'values' type Curve cannot be used to forecast index values."))

        if isinstance(self.meta.index_base, NoInput):
            return Err(
                ValueError(
                    "Curve must be initialised with an `index_base` value to derive `index_value`."
                )
            )

        lag_months = index_lag - self.meta.index_lag
        if index_method == IndexMethod.Curve:
            if lag_months != 0:
                return Err(
                    ValueError(
                        "'curve' interpolation can only be used with `index_value` when the Curve "
                        "`index_lag` matches the input `index_lag`."
                    )
                )
            # use traditional discount factor from Index base to determine index value.
            if index_date < self.nodes.initial:
                warnings.warn(
                    "The date queried on the Curve for an `index_value` is prior to the "
                    "initial node on the Curve.\nThis is returned as zero and likely "
                    f"causes downstream calculation error.\ndate queried: {index_date}"
                    "Either providing `index_fixings` to the object or extend the Curve backwards.",
                    UserWarning,
                )
                return Ok(0.0)
                # return zero for index dates in the past
                # the proper way for instruments to deal with this is to supply i_fixings
            elif index_date == self.nodes.initial:
                return Ok(self.meta.index_base)
            else:
                return Ok(self.meta.index_base * 1.0 / self.__getitem__(index_date))
        elif index_method == IndexMethod.Monthly:
            index_date_ = add_tenor(index_date, f"{lag_months * -1}M", "none", NoInput(0), 1)
            return self._try_index_value(
                index_date=index_date_,
                index_lag=self.meta.index_lag,
                index_method=IndexMethod.Curve,
            )
        elif index_method == IndexMethod.Daily:
            n = monthrange(index_date.year, index_date.month)[1]
            date_som = datetime(index_date.year, index_date.month, 1)
            date_sonm = add_tenor(index_date, "1M", "none", NoInput(0), 1)
            m1 = self._try_index_value(
                index_date=date_som, index_lag=index_lag, index_method=IndexMethod.Monthly
            )
            m2 = self._try_index_value(
                index_date=date_sonm, index_lag=index_lag, index_method=IndexMethod.Monthly
            )
            if m1.is_err:
                return m1
            if m2.is_err:
                return m2
            m1_, m2_ = m1.unwrap(), m2.unwrap()
            return Ok(m1_ + (index_date.day - 1) / n * (m2_ - m1_))
        else:
            return Err(  # pragma: no cover
                ValueError(
                    "`interpolation` for `index_value` must be in {'curve', 'daily', 'monthly'}."
                )
            )

    def index_value(
        self,
        index_date: datetime,
        index_lag: int,
        index_method: IndexMethod | str = IndexMethod.Curve,
    ) -> DualTypes:
        """
        Calculate the accrued value of the index from the ``index_base``.

        This method will raise if performed on a *'values'* type *Curve*.

        Parameters
        ----------
        index_date : datetime
            The reference date for which the index value will be returned.
        index_lag : int
            The number of months by which to lag the index when determining the value.
        index_method : IndexMethod or str in {"curve", "monthly", "daily"}
            The interpolation method for returning the index value. Monthly returns the index value
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
           :suppress:

           from rateslib import Curve, dt

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
        return self._try_index_value(
            index_date=index_date,
            index_lag=index_lag,
            index_method=_get_index_method(index_method),
        ).unwrap()

    # Rate Plotting

    def plot(
        self,
        tenor: str,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[_BaseCurve] | NoInput = NoInput(0),
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
        comparators_: list[_BaseCurve] = _drb([], comparators)
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
                if upper_tenor == "Z":
                    y_.append([comparator._plot_zero_rate(_x) for _x in x])
                else:
                    y_.append([comparator._plot_rate(_x, tenor, pm_) for _x in x])

        return plot([x] * len(y_), y_, labels)

    def _plot_diff(
        self, date: datetime, tenor: str, rate: DualTypes | None, comparator: _BaseCurve
    ) -> DualTypes | None:  # pragma: no cover
        if rate is None:
            return None
        if tenor == "Z" or tenor == "z":
            rate2 = comparator._plot_zero_rate(date)
        else:
            rate2 = comparator._plot_rate(date, tenor, comparator._plot_modifier(tenor))
        if rate2 is None:
            return None
        return rate2 - rate

    def _plot_modifier(self, upper_tenor: str) -> str:
        """If tenor is in days do not allow modified for plot purposes"""
        if "B" in upper_tenor or "D" in upper_tenor or "W" in upper_tenor:
            if "F" in self.meta.modifier:
                return "F"
            elif "P" in self.meta.modifier:  # pragma: no cover
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
            if upper_tenor == "Z":
                # then plotting zero rates just use the last date
                right_: datetime = self.nodes.final
            else:
                # pre-adjust the end date to enforce business date.
                right_ = add_tenor(
                    self.meta.calendar.adjust(self.nodes.final, Adjuster.Previous()),
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
        if upper_tenor == "Z":
            rates = [self._plot_zero_rate(_) for _ in dates]
        else:
            rates = [
                self._plot_rate(_, upper_tenor, self._plot_modifier(upper_tenor)) for _ in dates
            ]
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

    def _plot_zero_rate(
        self,
        maturity: datetime,
    ) -> DualTypes | None:
        """plotting a continuously compounded zero rate is done using the ActActISDA convention"""
        if self._base_type != _CurveType.dfs:
            raise ValueError(
                "To plot continuously compounded zero rates ('Z') the Curve `_base_type` must be "
                f"discount factor based. Got: '{self._base_type}'."
            )

        if maturity <= self.nodes.initial:
            return None
        else:
            t = dcf(self.nodes.initial, maturity, Convention.ActActISDA)
            return (dual_log(self[maturity]) / -t) * 100.0

    # Index Plotting

    def plot_index(
        self,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[_BaseCurve] | NoInput = NoInput(0),
        difference: bool = False,
        labels: list[str] | NoInput = NoInput(0),
        interpolation: str = "curve",
    ) -> PlotOutput:
        """
        Plot given index values on a *Curve*.

        Parameters
        ----------
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

    # Dunder operators

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

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__}:{self._id} at {hex(id(self))}>"

    def copy(self) -> _BaseCurve:
        """
        Create an identical copy of the curve object.

        Returns
        -------
        Self
        """
        ret: _BaseCurve = pickle.loads(pickle.dumps(self, -1))  # noqa: S301
        return ret

        # from rateslib.serialization import from_json
        # return from_json(self.to_json())


class ShiftedCurve(_BaseCurve):
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
    def _ad(self) -> int:
        return self.obj.ad

    @property
    def _meta(self) -> _CurveMeta:
        return self.obj.meta

    @property
    def _id(self) -> str:
        return self.obj.id

    @property
    def _nodes(self) -> _CurveNodes:
        return self.obj.nodes

    @property
    def _interpolator(self) -> _CurveInterpolator:
        return self.obj.interpolator

    @property
    def _base_type(self) -> _CurveType:
        return self.obj._base_type


class TranslatedCurve(_BaseCurve):
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
    def _ad(self) -> int:
        return self.obj.ad

    @property
    def _interpolator(self) -> _CurveInterpolator:
        return self.obj.interpolator

    @property
    def _meta(self) -> _CurveMeta:
        if self._base_type == _CurveType.dfs and not isinstance(self.obj.meta.index_base, NoInput):
            return replace(
                self.obj.meta,
                _index_base=self.obj.index_value(self.nodes.initial, self.obj.meta.index_lag),  # type: ignore[arg-type]
            )
        else:
            return self.obj.meta

    @property
    def _base_type(self) -> _CurveType:
        return self.obj._base_type


class RolledCurve(_BaseCurve):
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
    def _ad(self) -> int:
        return self.obj.ad

    @property
    def _interpolator(self) -> _CurveInterpolator:
        return self.obj.interpolator

    @property
    def _meta(self) -> _CurveMeta:
        return self.obj.meta

    @property
    def _nodes(self) -> _CurveNodes:
        return self.obj.nodes

    @property
    def _base_type(self) -> _CurveType:
        return self.obj._base_type


class _WithMutability:
    """
    This class is designed as a mixin for the methods for *Curve Pricing Objects*, i.e.
    the :class:`~rateslib.curves.Curve` and :class:`~rateslib.curves.LineCurve`.

    It permits initialization, configuration of ``nodes`` and ``meta`` and
    mutability when interacting with a :class:`~rateslib.solver.Solver`, when
    getting and setting nodes, as well as user update methods, spline interpolation solving and
    state validation.
    """

    _ini_solve: int
    _base_type: _CurveType
    _nodes: _CurveNodes
    _interpolator: _CurveInterpolator
    _ad: int
    _meta: _CurveMeta
    _id: str

    @_new_state_post
    def __init__(  # type: ignore[no-untyped-def]
        self,
        nodes: dict[datetime, DualTypes],
        *,
        interpolation: str | InterpolationFunction | NoInput = NoInput(0),
        t: list[datetime] | NoInput = NoInput(0),
        endpoints: str | tuple[str, str] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        convention: Convention | str | NoInput = NoInput(0),
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
            _convention=_get_convention(_drb(defaults.convention, convention)),
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

        temp: str | tuple[str, str] = _drb(defaults.endpoints, endpoints)
        if isinstance(temp, str):
            endpoints_: tuple[str, str] = (temp.lower(), temp.lower())
        else:
            endpoints_ = (temp[0].lower(), temp[1].lower())

        self._interpolator = _CurveInterpolator(
            local=interpolation,
            t=t,
            endpoints=endpoints_,
            node_dates=self._nodes.keys,
            convention=self._meta.convention,
            curve_type=self._base_type,
        )
        self._set_ad_order(order=ad)  # will also clear and initialise the cache

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
            k: set_order_convert(v, order, [f"{self._id}{i}"])
            for i, (k, v) in enumerate(self._nodes.nodes.items())
        }
        self._nodes = _CurveNodes(nodes_)
        self._interpolator._csolve(self._base_type, self._nodes, self._ad)

    # Solver interaction

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array(list(self._nodes.nodes.values())[self._ini_solve :])

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self._id}{i}" for i in range(self._ini_solve, self._nodes.n))

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
        self._interpolator._csolve(self._base_type, self._nodes, self._ad)

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

        self._interpolator._csolve(self._base_type, self._nodes, self._ad)

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
        if key not in self._nodes.nodes:
            raise KeyError("`key` is not in *Curve* ``nodes``.")

        nodes_ = self._nodes.nodes.copy()
        nodes_[key] = value
        self._nodes = _CurveNodes(nodes_)
        self._interpolator._csolve(self._base_type, self._nodes, self._ad)

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

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(self, vector: list[DualTypes], ad: int) -> None:
        """Used to update curve values during a Solver iteration. ``ad`` in {1, 2}."""
        self._set_node_vector_direct(vector, ad)

    def _set_node_vector_direct(self, vector: list[DualTypes], ad: int) -> None:
        nodes_ = self._nodes.nodes.copy()
        if ad == 0:
            if self._ini_solve == 1 and self._nodes.n > 0:
                nodes_[self._nodes.initial] = _dual_float(nodes_[self._nodes.initial])
            for i, k in enumerate(self._nodes.keys[self._ini_solve :]):
                nodes_[k] = _dual_float(vector[i])
        else:
            DualType: type[Dual | Dual2] = Dual if ad == 1 else Dual2
            DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
                ([],) if ad == 1 else ([], [])
            )
            base_obj = DualType(0.0, [f"{self._id}{i}" for i in range(self._nodes.n)], *DualArgs)
            ident: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.eye(
                self._nodes.n, dtype=np.float64
            )

            if self._ini_solve == 1:
                # then the first node on the Curve is not updated but
                # set it as a dual type with consistent vars.
                nodes_[self._nodes.initial] = DualType.vars_from(
                    base_obj,  # type: ignore[arg-type]
                    _dual_float(nodes_[self._nodes.initial]),
                    base_obj.vars,
                    ident[0, :].tolist(),
                    *DualArgs[1:],
                )

            for i, k in enumerate(self._nodes.keys[self._ini_solve :]):
                nodes_[k] = DualType.vars_from(
                    base_obj,  # type: ignore[arg-type]
                    _dual_float(vector[i]),
                    base_obj.vars,
                    ident[i + self._ini_solve, :].tolist(),
                    *DualArgs[1:],
                )
        self._ad = ad
        self._nodes = _CurveNodes(nodes_)
        self._interpolator._csolve(self._base_type, self._nodes, self._ad)

    # Serialization

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> _BaseCurve:
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

        _: _BaseCurve = cls(  # type: ignore[assignment]
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
        return _

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


class Curve(_WithMutability, _BaseCurve):
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


class LineCurve(_WithMutability, _BaseCurve):
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


class CompositeCurve(_BaseCurve):
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


class MultiCsaCurve(_BaseCurve):
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


class ProxyCurve(_BaseCurve):
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
    def _ad(self) -> int:
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


class CreditImpliedCurve(_BaseCurve):
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
    def _base_type(self) -> _CurveType:
        return self.obj._base_type

    @property
    def _id(self) -> str:
        return self.obj.id

    @property
    def _ad(self) -> int:
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
    def _nodes(self) -> _CurveNodes:
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
    index_method: str | IndexMethod,
    index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    index_date: datetime_ = NoInput(0),
    index_curve: CurveOption_ = NoInput(0),
) -> DualTypes:
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
    index_fixings: float, Dual, Dual2, Variable, Series[DualTypes], str, optional
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
    DualTypes

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
    index_method_ = _get_index_method(index_method)
    iv_result = _try_index_value(
        index_lag=index_lag,
        index_method=index_method_,
        index_fixings=index_fixings,
        index_date=index_date,
        index_curve=index_curve,
    )
    return iv_result.unwrap()


def _try_index_value(
    index_lag: int,
    index_method: IndexMethod,
    index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    index_date: datetime_ = NoInput(0),
    index_curve: CurveOption_ = NoInput(0),
) -> Result[DualTypes]:
    if isinstance(index_fixings, int | float | Dual | Dual2 | Variable):
        # i_fixings is a given value, probably aligned with an ``index_base``: return directly
        return Ok(index_fixings)

    if isinstance(index_curve, dict):
        return Err(
            NotImplementedError(
                "`index_curve` cannot currently be supplied as dict. Use a Curve type or "
                "NoInput(0)."
            )
        )

    if isinstance(index_date, NoInput):
        return Err(
            ValueError(
                "Must supply an `index_date` from which to forecast if `index_fixings` is "
                "not a value."
            )
        )

    if isinstance(index_fixings, NoInput | None):
        # forecast from curve if available
        if isinstance(index_curve, NoInput):
            return Err(
                ValueError(
                    "`index_value` must be forecast from a `index_curve` but no such argument "
                    "was provided."
                )
            )
        return index_curve._try_index_value(
            index_date=index_date,
            index_lag=index_lag,
            index_method=index_method,
        )
    elif isinstance(index_fixings, str):
        try:
            fixings_series = fixings.__getitem__(index_fixings)
        except Exception as e:
            return Err(e)
        if isinstance(index_curve, NoInput):
            return _index_value_from_series_no_curve(
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=fixings_series[1],
                index_date=index_date,
                index_fixings_boundary=fixings_series[2],
            )
        else:
            return _index_value_from_mixed_series_and_curve(
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=fixings_series[1],
                index_date=index_date,
                index_curve=index_curve,
            )
    elif isinstance(index_fixings, Series):
        if isinstance(index_curve, NoInput):
            return _index_value_from_series_no_curve(
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings,
                index_date=index_date,
            )
        else:
            return _index_value_from_mixed_series_and_curve(
                index_lag=index_lag,
                index_method=index_method,
                index_fixings=index_fixings,
                index_date=index_date,
                index_curve=index_curve,
            )
    else:
        return Err(
            TypeError(
                "`index_fixings` must be of type: Str, Series, DualTypes or NoInput.\n"
                f"{type(index_fixings)} was given."
            )
        )


def _index_value_from_mixed_series_and_curve(
    index_lag: int,
    index_method: IndexMethod,
    index_fixings: Series[DualTypes],  # type: ignore[type-var]
    index_date: datetime,
    index_curve: _BaseCurve,
) -> Result[DualTypes]:
    """
    Iterate through possibilities assuming a Curve and fixings as series exists.

    For returning a value from the Series the ``index_lag`` must be zero.
    If the lag is not zero then a Curve method will be used instead which will omit the Series.
    """
    if index_method == IndexMethod.Curve:
        if index_date in index_fixings.index:
            # simplest case returns Series value if all checks pass.
            if index_lag == 0:
                return Ok(index_fixings.loc[index_date])
            else:
                return Err(
                    ValueError(
                        "`index_lag` must be zero when using a 'curve' `index_method`.\n"
                        f"`index_date`: {index_date}, is in Series but got "
                        f"`index_lag`: {index_lag}."
                    )
                )
        elif len(index_fixings.index) == 0:
            # recall with the curve
            return index_curve._try_index_value(
                index_date=index_date, index_lag=index_lag, index_method=index_method
            )
        elif index_lag == 0 and (index_fixings.index[0] < index_date < index_fixings.index[-1]):
            # index date is within the Series index range but not found and the index lag is
            # zero so this should be available
            return Err(
                ValueError(
                    f"The Series given for `index_fixings` requires, but does not contain, "
                    f"the value for date: {index_date}.\n"
                    "For inflation indexes using 'monthly' or 'daily' `index_method` the "
                    "values associated for a month should be assigned "
                    "to the first day of that month."
                )
            )
        else:
            return index_curve._try_index_value(
                index_date=index_date, index_lag=index_lag, index_method=index_method
            )
    elif index_method == IndexMethod.Monthly:
        date_ = add_tenor(index_date, f"-{index_lag}M", "none", NoInput(0), 1)
        # a monthly value can only be derived from one source.
        # make separate determinations to avoid the issue of mis-matching index lags
        value_from_fixings = _try_index_value(
            index_lag=0,
            index_method=IndexMethod.Curve,
            index_fixings=index_fixings,
            index_date=date_,
            index_curve=NoInput(0),
        )
        if value_from_fixings.is_ok:
            return value_from_fixings
        else:
            value_from_curve = _try_index_value(
                index_lag=index_lag,
                index_method=IndexMethod.Monthly,
                index_fixings=NoInput(0),
                index_date=index_date,
                index_curve=index_curve,
            )
            return value_from_curve
    else:  # i_method == IndexMethod.Daily:
        n = monthrange(index_date.year, index_date.month)[1]
        date_som = datetime(index_date.year, index_date.month, 1)
        date_sonm = add_tenor(index_date, "1M", "none", NoInput(0), 1)
        m1 = _try_index_value(
            index_lag=index_lag,
            index_method=IndexMethod.Monthly,
            index_fixings=index_fixings,
            index_date=date_som,
            index_curve=index_curve,
        )
        if index_date == date_som:
            return m1
        m2 = _try_index_value(
            index_lag=index_lag,
            index_method=IndexMethod.Monthly,
            index_fixings=index_fixings,
            index_date=date_sonm,
            index_curve=index_curve,
        )
        if m2.is_err or m1.is_err:
            return Err(
                ValueError(
                    "The `index_value` could not be determined.\nThe period may be 'future' based "
                    "and there is no `index_fixing` available, or an `index_curve` has not be "
                    "able to forecast it."
                )
            )
            # this line cannot be hit when a curve returns DualTypes and not a NoInput
            # will raise a warning when the curve returns 0.0
        m1_, m2_ = m1.unwrap(), m2.unwrap()
        return Ok(m1_ + (index_date.day - 1) / n * (m2_ - m1_))


def _index_value_from_series_no_curve(
    index_lag: int,
    index_method: IndexMethod,
    index_fixings: Series[DualTypes],  # type: ignore[type-var]
    index_date: datetime,
    index_fixings_boundary: tuple[datetime, datetime] | None = None,
) -> Result[DualTypes]:
    """
    Derive a value from a Series only, detecting cases where the errors might be raised.
    """
    fixings_series = index_fixings

    if index_method == IndexMethod.Curve:
        if index_lag != 0:
            return Err(ValueError(err.VE_INDEX_LAG_MUST_BE_ZERO.format(index_date, index_lag)))

        if len(fixings_series.index) == 0:
            return Err(ValueError(err.VE_EMPTY_SERIES))

        if index_fixings_boundary is not None:
            left, right = index_fixings_boundary
            if index_date < left or index_date > right:
                return Err(FixingRangeError(index_date, index_fixings_boundary))
        else:
            right = fixings_series.index[-1]
            if index_date > right:
                return Err(FixingRangeError(index_date, (datetime(1, 1, 1), right)))
            left = fixings_series.index[0]
            if index_date < left:
                return Err(FixingRangeError(index_date, (left, right)))

        if index_date in fixings_series.index:
            # simplest case returns Series value if all checks pass.
            return Ok(fixings_series.loc[index_date])

        # date falls inside the dates of the Series but does not exist.
        return Err(FixingMissingDataError(index_date, (left, right)))
    elif index_method == IndexMethod.Monthly:
        date_ = add_tenor(index_date, f"-{index_lag}M", "none", NoInput(0), 1)
        return _index_value_from_series_no_curve(
            index_lag=0,
            index_method=IndexMethod.Curve,
            index_fixings=index_fixings,
            index_date=date_,
            index_fixings_boundary=index_fixings_boundary,
        )
    else:  # i_method == IndexMethod.Daily:
        n = monthrange(index_date.year, index_date.month)[1]
        date_som = datetime(index_date.year, index_date.month, 1)
        date_sonm = add_tenor(index_date, "1M", "none", NoInput(0), 1)
        m1 = _index_value_from_series_no_curve(
            index_lag=index_lag,
            index_method=IndexMethod.Monthly,
            index_fixings=index_fixings,
            index_date=date_som,
            index_fixings_boundary=index_fixings_boundary,
        )
        if index_date == date_som:
            return m1
        m2 = _index_value_from_series_no_curve(
            index_lag=index_lag,
            index_method=IndexMethod.Monthly,
            index_fixings=index_fixings,
            index_date=date_sonm,
            index_fixings_boundary=index_fixings_boundary,
        )
        if m1.is_err:
            return m1
        if m2.is_err:
            return m2
        m1_, m2_ = m1.unwrap(), m2.unwrap()
        return Ok(m1_ + (index_date.day - 1) / n * (m2_ - m1_))


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
