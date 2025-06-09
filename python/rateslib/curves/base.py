from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb
from typing import TYPE_CHECKING
from datetime import timedelta, datetime
from calendar import monthrange
import warnings

from rateslib.calendars import add_tenor, dcf
from rateslib.curves.utils import (
    _CurveInterpolator,
    _CurveMeta,
    _CurveNodes,
    _CurveType,
    average_rate,
)
from rateslib.default import NoInput, _drb, plot, PlotOutput
from rateslib.rs import Modifier

if TYPE_CHECKING:
    from rateslib.typing import Any, DualTypes


class BaseCurve(ABC):
    _ad: int
    _id: str
    _base_type: _CurveType
    _meta: _CurveMeta
    _nodes: _CurveNodes
    _interpolator: _CurveInterpolator

    @abstractmethod
    def __getitem__(self, value: datetime) -> DualTypes:
        """The get item method for any *Curve* type will allow the inheritance of the below
        methods.
        """
        ...

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
        termination: datetime | str | NoInput,
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
            raise ValueError("`effective` before initial LineCurve date.")
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

    def index_value(
        self, date: datetime, index_lag: int, interpolation: str = "curve"
    ) -> DualTypes:
        """
        Calculate the accrued value of the index from the ``index_base``.

        This method will raise if performed on a *'values'* type *Curve*.
        
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
        if self._base_type == _CurveType.values:
            raise TypeError(
                "A 'values' type Curve cannot be used to forecast index values."
            )

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
                return self.meta.index_base * 1.0 / self.__getitem__(date)
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

    # Rate Plotting

    def plot(
        self,
        tenor: str,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[BaseCurve] | NoInput = NoInput(0),
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
        comparators_: list[BaseCurve] = _drb([], comparators)
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
        self, date: datetime, tenor: str, rate: DualTypes | None, comparator: BaseCurve
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

    # Index Plotting

    def plot_index(
        self,
        right: datetime | str | NoInput = NoInput(0),
        left: datetime | str | NoInput = NoInput(0),
        comparators: list[BaseCurve] | NoInput = NoInput(0),
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
